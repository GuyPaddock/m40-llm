use anyhow::Result;
use m40_llm::decode::{decode_loop_with, greedy_sampler, StoppingCriteria};
use m40_llm::tokenizer::Tokenizer;

#[test]
fn decode_loop_respects_eos_and_max_tokens() -> Result<()> {
    let tokenizer = Tokenizer::byte_level();
    let eos_id = Some(b'!' as u32);
    let stopping = StoppingCriteria::new(Some(5), eos_id);
    let sampler = greedy_sampler(123);

    // logits function that points strongly to next byte of "Hi!" cycle
    let seq = [b'H' as u32, b'i' as u32, b'!' as u32];
    let mut pos = 0usize;
    let logits_fn = move |_ids: &[u32]| -> Result<Vec<f32>> {
        let mut logits = vec![0.0f32; 256];
        let idx = seq[pos % seq.len()] as usize;
        logits[idx] = 100.0; // very likely
        pos += 1;
        Ok(logits)
    };

    let ids = decode_loop_with(&tokenizer, "", true, sampler, &stopping, logits_fn)?;
    // Expect BOS if present? byte_level has no BOS, so sequence should be exactly sampled tokens until EOS hit or max_tokens
    // Our greedy logits emit H,i,!, then stop on ! because eos_id matches
    assert!(ids.ends_with(&[b'H' as u32, b'i' as u32, b'!' as u32]));
    Ok(())
}
