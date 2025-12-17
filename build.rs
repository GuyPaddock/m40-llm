use std::env;
use std::path::PathBuf;
use std::process::Command;

fn have_cmd(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn pick_host_cxx() -> String {
    for c in ["g++-12", "g++-11", "g++-10", "g++", "c++"] {
        if have_cmd(c) {
            return c.to_string();
        }
    }
    "c++".to_string()
}

fn main() {
    // Declare allowed cfgs (prevents check-cfg warnings)
    println!("cargo:rustc-check-cfg=cfg(nvcc)");
    println!("cargo:rustc-check-cfg=cfg(have_cublas)");

    // Rebuild triggers
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/stub.c");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

    if cuda_enabled {
        // ─────────────────────────────────────────────
        // CUDA BUILD
        // ─────────────────────────────────────────────

        if !have_cmd("nvcc") {
            panic!(
                "Feature `cuda` is enabled, but `nvcc` was not found in PATH.\n\
                 Install the CUDA toolkit or build without `--features cuda`."
            );
        }

        // This cfg means: "this build actually used nvcc"
        println!("cargo:rustc-cfg=nvcc");
        println!("cargo:rustc-cfg=have_cublas");

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
            // Tesla M40 (Maxwell)
            .flag("-gencode=arch=compute_52,code=sm_52")
            .compile("m40llm_native");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=m40llm_native");

        // CUDA runtime + cuBLAS
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    } else {
        // ─────────────────────────────────────────────
        // CPU / STUB BUILD
        // ─────────────────────────────────────────────

        let mut build = cc::Build::new();
        build
            .include("cuda")
            .file("cuda/stub.c")
            .flag_if_supported("-O3")
            .flag_if_supported("-fPIC")
            .compile("m40llm_native");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=m40llm_native");
    }
}
