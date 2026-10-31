pub const USER_PROMPT: &str = "Explain everything that happens between the time a user types a web address into their browser and the time the page is fully loaded and displayed on their screen in as much detail as possible";
pub const MAX_TOKENS: usize = 512;
pub const CONTEXT_TOKENS: u32 = 1024;

pub fn render_user_prompt(user_prompt: &str) -> String {
    format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n")
}
