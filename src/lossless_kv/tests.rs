use super::*;

fn normal(exponent: u16, index: usize) -> u16 {
    (exponent << 10) | ((index as u16) & 1023) | (((index as u16) & 1) << 15)
}

/// Every half bit pattern survives integer reconstruction, including signed
/// zeros and NaN payloads; permutation also exercises broad-range raw blocks.
#[test]
fn test_all_half_patterns_roundtrip() -> Result<()> {
    for multiplier in [1_u16, 40503] {
        let words: Vec<_> = (0..=u16::MAX).map(|v| v.wrapping_mul(multiplier)).collect();
        let encoded = LosslessKvBlocks::encode(&words)?;
        assert_eq!(encoded.decode()?, words);
        assert_eq!(
            LosslessKvBlocks::from_bytes(encoded.get_bytes().to_vec())?.decode()?,
            words
        );
        assert_eq!(encoded.get_stats().values, 65536);
        assert_eq!(encoded.get_stats().original_bytes, 131072);
    }
    Ok(())
}

/// Each exponent width has an independently calculated payload size, and the
/// first code's sign/fraction/offset bits match the documented byte ordering.
#[test]
fn test_widths_and_bit_layout() -> Result<()> {
    for (difference, width) in [(0, 0), (1, 1), (2, 2), (4, 3), (8, 4), (16, 5)] {
        let words: Vec<_> = (0..512)
            .map(|index| normal(5 + if index % 2 == 0 { difference } else { 0 }, index))
            .collect();
        let encoded = LosslessKvBlocks::encode(&words)?;
        assert_eq!(encoded.decode()?, words);
        let stats = encoded.get_stats();
        assert_eq!(stats.metadata_bytes, 16);
        if width < 5 {
            assert_eq!(stats.packed_blocks_by_width[width], 1);
            assert_eq!(stats.payload_bytes, 512 * (11 + width) / 8);
            let first = u32::from_le_bytes(encoded.get_bytes()[16..20].try_into().unwrap());
            assert_eq!(
                first & ((1 << (11 + width)) - 1),
                u32::from(difference) << 11
            );
        } else {
            assert_eq!(stats.raw_blocks, 1);
            assert_eq!(stats.encoded_bytes, 1040);
        }
    }
    Ok(())
}

/// Exceptional values force raw storage, preserving their exact payloads and
/// exposing directory overhead instead of reporting false compression savings.
#[test]
fn test_exceptional_raw_blocks() -> Result<()> {
    for special in [0, 0x8000, 1, 0x83ff, 0x7c00, 0xfc00, 0x7d55, 0xfeaa] {
        let mut words = vec![0x3c00; 512];
        words[137] = special;
        let encoded = LosslessKvBlocks::encode(&words)?;
        assert_eq!(encoded.decode()?, words);
        assert_eq!(encoded.get_stats().raw_blocks, 1);
        assert_eq!(encoded.get_stats().encoded_bytes, 1040);
    }
    Ok(())
}

/// Tail counts, byte padding and per-block eligibility include all metadata;
/// tiny blocks remain raw even though their exponent payload could shrink.
#[test]
fn test_tails_and_size_accounting() -> Result<()> {
    for count in [1, 2, 14, 15, 16, 17, 127, 511, 512, 513, 1024, 1025] {
        let words: Vec<_> = (0..count).map(|i| normal(15, i)).collect();
        let encoded = LosslessKvBlocks::encode(&words)?;
        let stats = encoded.get_stats();
        assert_eq!(encoded.decode()?, words);
        assert_eq!(
            stats.encoded_bytes,
            stats.metadata_bytes + stats.payload_bytes
        );
        assert_eq!(stats.metadata_bytes, 8 + count.div_ceil(512) * 8);
        assert_eq!(
            stats.raw_blocks + stats.packed_blocks_by_width.iter().sum::<usize>(),
            count.div_ceil(512)
        );
        if count <= 14 {
            assert_eq!(stats.raw_blocks, 1);
        } else if count == 15 {
            assert_eq!(stats.packed_blocks_by_width[0], 1);
            assert_eq!(stats.payload_bytes, 21);
            assert_eq!(stats.encoded_bytes, 37);
        } else if count == 17 {
            assert_eq!(stats.packed_blocks_by_width[0], 1);
            assert_eq!(stats.payload_bytes, 24);
            assert_eq!(stats.encoded_bytes, 40);
        }
    }
    assert!(LosslessKvBlocks::encode(&[]).is_err());
    Ok(())
}

/// Malformed directory fields, impossible exponents, oversized counts and
/// nonzero tail padding are rejected before any decoded data is returned.
#[test]
fn test_rejects_malformed_streams() -> Result<()> {
    let source = LosslessKvBlocks::encode(&vec![0x3c00; 511])?;
    let bytes = source.get_bytes();
    for (position, value) in [
        (0, b'X'),
        (8, 0),
        (12, 0),
        (14, 0),
        (14, 31),
        (15, 5),
        (15, RAW_WIDTH),
    ] {
        let mut changed = bytes.to_vec();
        changed[position] = value;
        assert!(
            LosslessKvBlocks::from_bytes(changed).is_err(),
            "accepted changed byte {position}"
        );
    }
    let mut huge = bytes.to_vec();
    huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(LosslessKvBlocks::from_bytes(huge).is_err());
    let mut padding = bytes.to_vec();
    *padding.last_mut().unwrap() |= 0x80;
    assert!(LosslessKvBlocks::from_bytes(padding).is_err());
    let mut trailing = bytes.to_vec();
    trailing.push(0);
    assert!(LosslessKvBlocks::from_bytes(trailing).is_err());
    for count in [0, 1, 7, 15, bytes.len() - 1] {
        assert!(LosslessKvBlocks::from_bytes(bytes[..count].to_vec()).is_err());
    }
    // Valid base and width can still yield a reserved exponent code.
    let mut words = vec![normal(15, 0); 512];
    words[0] = normal(16, 0);
    let mut invalid = LosslessKvBlocks::encode(&words)?.get_bytes().to_vec();
    invalid[14] = 30;
    assert!(LosslessKvBlocks::from_bytes(invalid).is_err());
    Ok(())
}
