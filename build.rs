use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Check if nvcc is available
    let nvcc_available = Command::new("which")
        .arg("nvcc")
        .status()
        .map(|status| status.success())
        .unwrap_or(false);

    if nvcc_available {
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
    } else {
        // Create empty cubin file to allow compilation to proceed
        let _ = File::create(out_dir.join("kernels.cubin")).unwrap();
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        // Skip linking with CUDA libraries when nvcc is not available
    }
}
