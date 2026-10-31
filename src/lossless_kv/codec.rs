use super::{LosslessKvStats, BLOCK_VALUES, ENTRY_BYTES, HEADER_BYTES, MAGIC, RAW_WIDTH};
use anyhow::{ensure, Context, Result};

fn read_u32(bytes: &[u8]) -> usize {
    u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize
}

fn push_bits(output: &mut Vec<u8>, accumulator: &mut u32, used: &mut u32, value: u32, width: u32) {
    *accumulator |= value << *used;
    *used += width;
    while *used >= 8 {
        output.push(*accumulator as u8);
        *accumulator >>= 8;
        *used -= 8;
    }
}

pub(super) fn encode(words: &[u16]) -> Result<Vec<u8>> {
    ensure!(
        !words.is_empty() && words.len() <= u32::MAX as usize,
        "invalid lossless KV value count"
    );
    let block_count = words.len().div_ceil(BLOCK_VALUES);
    let directory_bytes = block_count
        .checked_mul(ENTRY_BYTES)
        .and_then(|v| v.checked_add(HEADER_BYTES))
        .context("directory overflow")?;
    let maximum_bytes = words
        .len()
        .checked_mul(2)
        .and_then(|v| v.checked_add(directory_bytes))
        .context("lossless KV size overflow")?;
    ensure!(
        maximum_bytes <= u32::MAX as usize,
        "lossless KV stream exceeds u32 offsets"
    );
    let mut output = vec![0; directory_bytes];
    output[..4].copy_from_slice(MAGIC);
    output[4..8].copy_from_slice(&(words.len() as u32).to_le_bytes());
    for (index, block) in words.chunks(BLOCK_VALUES).enumerate() {
        let mut minimum = 31_u16;
        let mut maximum = 0_u16;
        let mut has_exception = false;
        for &word in block {
            let exponent = (word >> 10) & 31;
            minimum = minimum.min(exponent);
            maximum = maximum.max(exponent);
            has_exception |= exponent == 0 || exponent == 31;
        }
        let difference = maximum - minimum;
        let width = (u16::BITS - difference.leading_zeros()) as u8;
        let packed_bytes = (block.len() * (11 + usize::from(width))).div_ceil(8);
        let is_packed =
            !has_exception && width <= 4 && packed_bytes + ENTRY_BYTES < block.len() * 2;
        let offset = output.len() as u32;
        let entry = HEADER_BYTES + index * ENTRY_BYTES;
        output[entry..entry + 4].copy_from_slice(&offset.to_le_bytes());
        output[entry + 4..entry + 6].copy_from_slice(&(block.len() as u16).to_le_bytes());
        output[entry + 6] = if is_packed { minimum as u8 } else { 0 };
        output[entry + 7] = if is_packed { width } else { RAW_WIDTH };
        if is_packed {
            let mut accumulator = 0;
            let mut used = 0;
            for &word in block {
                let exponent = ((word >> 10) & 31) - minimum;
                let code = u32::from(word & 1023)
                    | (u32::from(word >> 15) << 10)
                    | (u32::from(exponent) << 11);
                push_bits(
                    &mut output,
                    &mut accumulator,
                    &mut used,
                    code,
                    11 + u32::from(width),
                );
            }
            if used > 0 {
                output.push(accumulator as u8);
            }
        } else {
            for word in block {
                output.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    Ok(output)
}

/// Read fields independently by bit position rather than sharing the encoder's
/// streaming accumulator. Reconstruction uses integer operations only.
fn unpack(payload: &[u8], index: usize, width: u8) -> u16 {
    let field_bits = 11 + usize::from(width);
    let bit_start = index * field_bits;
    let mut code = 0_u16;
    for bit in 0..field_bits {
        let location = bit_start + bit;
        code |= u16::from((payload[location / 8] >> (location % 8)) & 1) << bit;
    }
    code
}

pub(super) fn visit(bytes: &[u8], mut consume: impl FnMut(u16)) -> Result<LosslessKvStats> {
    ensure!(
        bytes.len() >= HEADER_BYTES && bytes.len() <= u32::MAX as usize && &bytes[..4] == MAGIC,
        "invalid lossless KV header"
    );
    let values = read_u32(&bytes[4..8]);
    let original_bytes = values
        .checked_mul(2)
        .context("original byte count overflow")?;
    ensure!(
        values > 0 && values <= bytes.len().saturating_mul(8) / 11,
        "invalid lossless KV count"
    );
    let blocks = values.div_ceil(BLOCK_VALUES);
    let metadata = blocks
        .checked_mul(ENTRY_BYTES)
        .and_then(|v| v.checked_add(HEADER_BYTES))
        .context("directory overflow")?;
    ensure!(metadata <= bytes.len(), "truncated lossless KV directory");
    let mut expected_offset = metadata;
    let mut raw_blocks = 0;
    let mut packed = [0; 5];
    for index in 0..blocks {
        let entry = &bytes[HEADER_BYTES + index * ENTRY_BYTES..][..ENTRY_BYTES];
        let offset = read_u32(entry);
        let count = usize::from(u16::from_le_bytes([entry[4], entry[5]]));
        let base = entry[6];
        let width = entry[7];
        ensure!(
            offset == expected_offset && count == (values - index * BLOCK_VALUES).min(BLOCK_VALUES),
            "invalid lossless KV block span"
        );
        let payload_bytes = if width == RAW_WIDTH {
            ensure!(base == 0, "raw lossless KV base is nonzero");
            raw_blocks += 1;
            count * 2
        } else {
            ensure!(
                width <= 4 && (1..=30).contains(&base),
                "invalid lossless KV packed fields"
            );
            let length = (count * (11 + usize::from(width))).div_ceil(8);
            ensure!(
                length + ENTRY_BYTES < count * 2,
                "packed lossless KV block loses space"
            );
            packed[usize::from(width)] += 1;
            length
        };
        let end = offset
            .checked_add(payload_bytes)
            .context("payload end overflow")?;
        let payload = bytes
            .get(offset..end)
            .context("truncated lossless KV payload")?;
        if width == RAW_WIDTH {
            for pair in payload.chunks_exact(2) {
                consume(u16::from_le_bytes([pair[0], pair[1]]));
            }
        } else {
            for position in 0..count {
                let code = unpack(payload, position, width);
                let exponent = u16::from(base) + (code >> 11);
                ensure!(
                    (1..=30).contains(&exponent),
                    "packed lossless KV exponent is exceptional"
                );
                consume((code & 1023) | ((code & 1024) << 5) | (exponent << 10));
            }
            let bits = count * (11 + usize::from(width));
            let tail = bits % 8;
            ensure!(
                tail == 0 || payload[payload.len() - 1] >> tail == 0,
                "nonzero lossless KV padding"
            );
        }
        expected_offset = end;
    }
    ensure!(expected_offset == bytes.len(), "trailing lossless KV bytes");
    Ok(LosslessKvStats {
        values,
        original_bytes,
        encoded_bytes: bytes.len(),
        metadata_bytes: metadata,
        payload_bytes: bytes.len() - metadata,
        raw_blocks,
        packed_blocks_by_width: packed,
    })
}

pub(super) fn validate(bytes: &[u8]) -> Result<LosslessKvStats> {
    visit(bytes, |_| {})
}
