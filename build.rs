use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/stub.c");

    // Always declare this cfg so we can gate tests without warnings
    println!("cargo::rustc-check-cfg=cfg(nvcc)");

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
            println!("cargo:rustc-cfg=nvcc");

            // If available, use conda CUDA include/lib paths
            let conda_prefix = env::var("CONDA_PREFIX").ok();
            let conda_include = conda_prefix.as_ref().map(|p| format!("{}/include", p));
            let conda_lib = conda_prefix.as_ref().map(|p| format!("{}/lib", p));

            // Detect cuBLAS header availability
            let mut have_cublas_header = false;
            if let Some(ref inc) = conda_include {
                let hdr = std::path::Path::new(inc).join("cublas_v2.h");
                if hdr.exists() {
                    have_cublas_header = true;
                }
            }
            // Fallback common locations
            for p in ["/usr/local/cuda/include", "/usr/include"] {
                if !have_cublas_header {
                    let hdr = std::path::Path::new(p).join("cublas_v2.h");
                    if hdr.exists() {
                        have_cublas_header = true;
                    }
                }
            }

            // Compile CUDA into static lib
            let mut build = cc::Build::new();
            build
                .cuda(true)
                .include("cuda")
                .file("cuda/kernels.cu")
                .flag("-std=c++14")
                .flag("-O3")
                .flag("-Xcompiler")
                .flag("-fPIC")
                .flag("-gencode=arch=compute_52,code=sm_52") // Tesla M40
                .flag("-allow-unsupported-compiler");
            if let Some(ref inc) = conda_include {
                build.include(inc);
            }
            if have_cublas_header {
                build.define("M40LLM_HAVE_CUBLAS", None);
            }
            build.compile("m40llm_kernels");

            println!("cargo:rustc-link-search=native={}", out_dir.display());
            if let Some(ref libp) = conda_lib {
                println!("cargo:rustc-link-search=native={}", libp);
            }
            println!("cargo:rustc-link-lib=static=m40llm_kernels");
            println!("cargo:rustc-link-lib=cudart");
            if have_cublas_header {
                println!("cargo:rustc-link-lib=cublas");
            }
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
