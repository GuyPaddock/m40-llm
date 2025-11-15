use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Compile CUDA into static lib
    cc::Build::new()
        .cuda(true)
        .file("cuda/kernels.cu")
        .flag("-std=c++14")
        .flag("-O3")
        .flag("-Xcompiler").flag("-fPIC")
        .flag("-gencode=arch=compute_52,code=sm_52") // Tesla M40
        .compile("fastllm_kernels");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=fastllm_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
}
