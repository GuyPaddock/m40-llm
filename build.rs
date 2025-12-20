use std::env;
use std::path::{Path, PathBuf};
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

struct CublasPaths {
    includes: Vec<PathBuf>,
    rpaths: Vec<PathBuf>,
    detected: bool,
}

fn detect_cublas_paths() -> CublasPaths {
    let conda_prefix = env::var("CONDA_PREFIX").ok();
    let conda_include = conda_prefix.as_ref().map(|p| PathBuf::from(p).join("include"));
    let conda_lib = conda_prefix.as_ref().map(|p| PathBuf::from(p).join("lib"));
    let conda_targets_include = conda_prefix
        .as_ref()
        .map(|p| PathBuf::from(p).join("targets/x86_64-linux/include"));
    let conda_targets_lib = conda_prefix
        .as_ref()
        .map(|p| PathBuf::from(p).join("targets/x86_64-linux/lib"));

    let mut include_paths: Vec<PathBuf> = vec![];
    include_paths.extend(conda_targets_include.clone());
    include_paths.extend(conda_include.clone());
    include_paths.push(PathBuf::from("/usr/local/cuda/include"));
    include_paths.push(PathBuf::from("/usr/include"));

    let mut lib_paths: Vec<PathBuf> = vec![];
    lib_paths.extend(conda_targets_lib.clone());
    lib_paths.extend(conda_lib.clone());
    lib_paths.push(PathBuf::from("/usr/local/cuda/lib64"));
    lib_paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    lib_paths.push(PathBuf::from("/usr/lib64"));

    let have_header = include_paths.iter().any(|p| p.join("cublas_v2.h").exists());
    let have_lib = lib_paths.iter().any(|p| {
        ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
            .iter()
            .any(|name| p.join(name).exists())
    });

    let mut rpaths = vec![];
    rpaths.extend(conda_targets_lib);
    rpaths.extend(conda_lib);
    rpaths.push(PathBuf::from("/usr/local/cuda/lib64"));
    rpaths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    rpaths.push(PathBuf::from("/usr/lib64"));

    let detected = have_header && have_lib;

    CublasPaths {
        includes: include_paths,
        rpaths,
        detected,
    }
}

fn build_stub(out_dir: &Path) {
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
    let nvcc_available = have_cmd("nvcc");

    if cuda_enabled && !nvcc_available {
        panic!(
            "Feature `cuda` is enabled, but `nvcc` was not found in PATH.\n\
            Install the CUDA toolkit or build without `--features cuda`."
        );
    }

    if cuda_enabled {
        // ─────────────────────────────────────────────
        // CUDA BUILD
        // ─────────────────────────────────────────────

        // This cfg means: "this build actually used nvcc"
        println!("cargo:rustc-cfg=nvcc");

        let host_cxx = pick_host_cxx();
        let cublas_paths = detect_cublas_paths();
        let cublas_enabled = cublas_paths.detected
            && env::var("M40LLM_ENABLE_CUBLAS").ok().as_deref() == Some("1");

        let mut build = cc::Build::new();
        build
            .cuda(true)
            .include("cuda")
            .file("cuda/kernels.cu")
            .flag(&format!("-ccbin={}", host_cxx))
            .flag("-std=c++17")
            .flag("-Xcompiler")
            .flag("-std=gnu++17")
            .flag("-cudart=shared")
            .flag("-O3")
            .flag("-Xcompiler")
            .flag("-fPIC")
            // Tesla M40 (Maxwell)
            .flag("-gencode=arch=compute_52,code=sm_52")
            // Also embed PTX so newer GPUs can JIT
            .flag("-gencode=arch=compute_52,code=compute_52")
            .flag("-allow-unsupported-compiler");

        for inc in cublas_paths.includes.iter() {
            build.include(inc);
        }

        if cublas_enabled {
            build.define("M40LLM_HAVE_CUBLAS", None);
            println!("cargo:rustc-cfg=have_cublas");
        }

        build.compile("m40llm_native");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=m40llm_native");

        for p in cublas_paths.rpaths.iter() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", p.display());
            println!("cargo:rustc-link-search=native={}", p.display());
        }

        // CUDA runtime + optional cuBLAS
        println!("cargo:rustc-link-lib=cudart");
        if cublas_enabled {
            println!("cargo:rustc-link-lib=cublas");
        }
    } else {
        // ─────────────────────────────────────────────
        // CPU / STUB BUILD
        // ─────────────────────────────────────────────
        build_stub(&out_dir);
    }
}
