use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/stub.c");

    // Always declare this cfg so we can gate tests without warnings
    println!("cargo::rustc-check-cfg=cfg(has_nvcc)");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Is the crate built with the `cuda` feature?
    let cuda_feature_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

    // Check if nvcc is available
    let nvcc_available = Command::new("which")
        .arg("nvcc")
        .status()
        .map(|status| status.success())
        .unwrap_or(false);

    if cuda_feature_enabled {
        if nvcc_available {
            // Expose cfg so tests/examples can detect real CUDA
            println!("cargo:rustc-cfg=has_nvcc");

            // Compile CUDA into static lib
            cc::Build::new()
                .cuda(true)
                .file("cuda/kernels.cu")
                .flag("-std=c++14")
                .flag("-O3")
                .flag("-Xcompiler")
                .flag("-fPIC")
                .flag("-gencode=arch=compute_52,code=sm_52") // Tesla M40
                .compile("m40llm_kernels");

            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=m40llm_kernels");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cublas");
        } else {
            // Build stub library that defines all required symbols so linking succeeds
            cc::Build::new()
                .file("cuda/stub.c")
                .compile("m40llm_kernels");
            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=m40llm_kernels");
        }
    } else {
        // No CUDA feature: nothing to build/link
        let _ = File::create(out_dir.join("kernels.cubin")).unwrap();
        println!("cargo:rustc-link-search=native={}", out_dir.display());
    }
}
