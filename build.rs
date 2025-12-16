use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process::Command;

#[cfg(feature = "cuda")]
use cc;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/stub.c");

    // Configuration directives
    println!("cargo::rustc-check-cfg=cfg(nvcc)");
    println!("cargo::rustc-check-cfg=cfg(have_cublas_header)");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_feature_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

    // Host compiler detection (only needed for CUDA)
    #[cfg(feature = "cuda")]
    let host_cxx =
        which_preferring(&["g++-12", "g++-11", "g++", "c++"]).unwrap_or_else(|| "c++".to_string());
    #[cfg(not(feature = "cuda"))]
    let _host_cxx = ""; // Placeholder when CUDA disabled

    if cuda_feature_enabled {
        #[cfg(feature = "cuda")]
        {
            // CUDA compilation and linking
            let mut build = cc::Build::new();
            build
                .cuda(true)
                .include("cuda")
                .file("cuda/kernels.cu")
                .flag(&format!("-ccbin={}", host_cxx))
                .flag("-std=c++17")
                .flag("-Xcompiler")
                .flag("-std=gnu++17")
                .flag("-O3")
                .flag("-Xcompiler")
                .flag("-fPIC")
                .flag("-gencode=arch=compute_52,code=sm_52")
                .compile("m40llm_kernels");

            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=m40llm_kernels");
            println!("cargo:rustc-link-lib=cudart");
        }
    } else {
        // Non-CUDA fallback
        let _ = File::create(out_dir.join("kernels.cubin")).unwrap();
        println!("cargo:rustc-link-search=native={}", out_dir.display());
    }
}

fn which_preferring(candidates: &[&str]) -> Option<String> {
    candidates
        .iter()
        .find(|&&c| {
            Command::new("which")
                .arg(c)
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        })
        .map(|s| s.to_string())
}
