use std::env;
use std::path::PathBuf;
use std::process::Command;

fn which(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn pick_host_cxx() -> String {
    for c in ["g++-12", "g++-11", "g++-10", "g++", "c++"] {
        if which(c) {
            return c.to_string();
        }
    }
    "c++".to_string()
}

fn main() {
    // Declare allowed cfgs (prevents check-cfg warnings)
    println!("cargo:rustc-check-cfg=cfg(nvcc)");
    println!("cargo:rustc-check-cfg=cfg(have_cublas_header)");

    // Re-run triggers
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/stub.c");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by Cargo"));

    let cuda_feature_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    let have_nvcc = which("nvcc");

    if have_nvcc {
        println!("cargo:rustc-cfg=nvcc");
    }

    if cuda_feature_enabled {
        if !have_nvcc {
            panic!(
                "Feature `cuda` enabled but `nvcc` not found in PATH. \
                 Install CUDA or build without `--features cuda`."
            );
        }

        let host_cxx = pick_host_cxx();

        let mut build = cc::Build::new();
        build
            .cuda(true)
            .include("cuda")
            .file("cuda/kernels.cu")
            .flag(&format!("-ccbin={}", host_cxx))
            .flag("-std=c++17")
            .flag("-O3")
            .flag("-Xcompiler")
            .flag("-fPIC")
            .flag("-gencode=arch=compute_52,code=sm_52")
            .compile("m40llm_kernels");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=m40llm_kernels");
        println!("cargo:rustc-link-lib=cudart");
    } else {
        let mut build = cc::Build::new();
        build
            .include("cuda")
            .file("cuda/stub.c")
            .flag_if_supported("-O3")
            .flag_if_supported("-fPIC")
            .compile("m40llm_stub");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=m40llm_stub");
    }
}
