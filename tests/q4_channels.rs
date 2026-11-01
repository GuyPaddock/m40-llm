use anyhow::Result;
use m40_llm::infer::q4_channels::Q4ChannelWeights;

/// Every scale bit pattern survives packing, including NaNs and signed zeros.
#[test]
fn test_preserves_all_scale_bits_and_payloads() -> Result<()> {
    let mut source = Vec::new();
    for scale in 0..=u16::MAX {
        source.extend_from_slice(&scale.to_le_bytes());
        for byte in 0..16u16 {
            source.push(scale.wrapping_add(byte * 17) as u8);
        }
    }
    let packed = Q4ChannelWeights::pack(&source, 64, 32768)?;
    assert_eq!(packed.unpack(), source);
    assert_eq!(packed.bytes(), source.len());
    Ok(())
}

/// Scattered channels decode to native weights across scale groups and odd tails.
#[test]
fn test_selected_channels_preserve_numerical_values() -> Result<()> {
    let mut source = Vec::new();
    for output in 0..5 {
        for group in 0..3 {
            let scale =
                half::f16::from_f32((output + 1) as f32 * if group == 1 { -0.25 } else { 0.5 });
            source.extend_from_slice(&scale.to_bits().to_le_bytes());
            for pair in 0..16 {
                source.push(((pair + output) % 16) as u8 | (((15 - pair + group) % 16) as u8) << 4);
            }
        }
    }
    let packed = Q4ChannelWeights::pack(&source, 96, 5)?;
    assert_eq!(packed.input_dim(), 96);
    assert_eq!(packed.output_dim(), 5);
    assert_eq!(packed.unpack(), source);
    assert_eq!(packed.bytes(), source.len() + 48);
    assert!(packed.planes()[1]
        .chunks_exact(3)
        .all(|row| row[2] & 0xf0 == 0));
    for channel in [0, 15, 16, 31, 32, 47, 63, 64, 95] {
        for output in 0..5 {
            let block = (output * 3 + channel / 32) * 18;
            let scale = half::f16::from_le_bytes([source[block], source[block + 1]]).to_f32();
            let byte = source[block + 2 + channel % 16];
            let code = if channel % 32 < 16 {
                byte & 15
            } else {
                byte >> 4
            };
            let expected = scale * (i32::from(code) - 8) as f32;
            assert_eq!(
                packed.decode_weight(channel, output)?.to_bits(),
                expected.to_bits()
            );
        }
    }
    assert!(packed.decode_weight(96, 0).is_err());
    assert!(packed.decode_weight(0, 5).is_err());
    Ok(())
}

/// Invalid dimensions or byte spans fail before allocation or indexing.
#[test]
fn test_rejects_invalid_sources() {
    for (input, output, bytes) in [
        (0, 1, 0),
        (31, 1, 18),
        (32, 0, 0),
        (32, 1, 17),
        (32, 1, 19),
        (usize::MAX - 31, usize::MAX, 0),
    ] {
        assert!(Q4ChannelWeights::pack(&vec![0; bytes], input, output).is_err());
    }
}
