env M40LLM_ENABLE_NVCC=0 M40LLM_ENABLE_CUBLAS=0 RUSTFLAGS=-D\ warnings cargo test --no-default-features --locked -- --test-threads=1 
