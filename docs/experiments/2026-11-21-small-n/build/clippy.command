env M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo clippy --locked --features cuda\,server --all-targets -- -D warnings 
