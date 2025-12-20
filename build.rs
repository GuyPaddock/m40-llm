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

struct CublasPaths {
    includes: Vec<PathBuf>,
    rpaths: Vec<PathBuf>,
    detected: bool,
}

fn detect_cublas_paths() -> CublasPaths {
    let mut prefixes: Vec<PathBuf> = vec![];

    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        prefixes.push(PathBuf::from(conda_prefix));
    }

    if let Ok(mamba_root) = env::var("MAMBA_ROOT_PREFIX") {
        prefixes.push(PathBuf::from(mamba_root));
    }

    let default_mamba = PathBuf::from("/root/.local/share/mamba");
    if default_mamba.exists() {
        prefixes.push(default_mamba);
    }

    let mut include_paths: Vec<PathBuf> = vec![];
    let mut lib_paths: Vec<PathBuf> = vec![];

    for prefix in prefixes.iter() {
        include_paths.push(prefix.join("targets/x86_64-linux/include"));
        include_paths.push(prefix.join("include"));

        lib_paths.push(prefix.join("targets/x86_64-linux/lib"));
        lib_paths.push(prefix.join("lib"));
    }

    include_paths.push(PathBuf::from("/usr/local/cuda/include"));
    include_paths.push(PathBuf::from("/usr/include"));
    include_paths.sort();
    include_paths.dedup();

    lib_paths.push(PathBuf::from("/usr/local/cuda/lib64"));
    lib_paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    lib_paths.push(PathBuf::from("/usr/lib64"));
    lib_paths.sort();
    lib_paths.dedup();

    let have_header = include_paths.iter().any(|p| p.join("cublas_v2.h").exists());
    let have_lib = lib_paths.iter().any(|p| {
        ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
            .iter()
            .any(|name| p.join(name).exists())
    });

    let mut rpaths = lib_paths.clone();
    rpaths.sort();
    rpaths.dedup();

    let detected = have_header && have_lib;

    CublasPaths {
        includes: include_paths,
        rpaths,
        detected,
    }
}

fn nvcc_host_compiler() -> Option<String> {
    if let Ok(ccbin) = env::var("NVCC_CCBIN").or_else(|_| env::var("CUDAHOSTCXX")) {
        return Some(ccbin);
    }

    if let Ok(cxx) = env::var("CXX") {
        return Some(cxx);
    }

    for candidate in ["/usr/bin/g++", "g++-13", "g++-12", "g++-11", "g++"] {
        if have_cmd(candidate) {
            return Some(candidate.to_string());
        }
    }

    None
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

        let cublas_paths = detect_cublas_paths();
        let cublas_enabled = cublas_paths.detected
            && env::var("M40LLM_ENABLE_CUBLAS").ok().as_deref() == Some("1");

        let mut build = cc::Build::new();
        build
            .cuda(true)
            .include("cuda")
            .file("cuda/kernels.cu")
            .flag("-std=c++17")
            .flag("-Xcompiler")
            .flag("-std=c++17")
            // Disable GNU-only math prototypes (e.g., cospi/sinpi) to avoid
            // conflicting exception specs with CUDA's math_functions.h
            .define("__STRICT_ANSI__", None)
            .flag("-Xcompiler")
            .flag("-D__STRICT_ANSI__")
            .flag("-cudart=shared")
            .flag("-O3")
            .flag("-Xcompiler")
            .flag("-fPIC")
            // Tesla M40 (Maxwell)
            .flag("-gencode=arch=compute_52,code=sm_52")
            // Also embed PTX so newer GPUs can JIT
            .flag("-gencode=arch=compute_52,code=compute_52")
            .flag("-allow-unsupported-compiler");

        if let Some(ccbin) = nvcc_host_compiler() {
            if env::var("CXX").is_err() {
                env::set_var("CXX", &ccbin);
            }

            if env::var("CUDAHOSTCXX").is_err() {
                env::set_var("CUDAHOSTCXX", &ccbin);
            }
        }

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
