use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/stub.c");

    // Always declare these cfgs so we can gate tests without warnings
    println!("cargo::rustc-check-cfg=cfg(nvcc)");
    println!("cargo::rustc-check-cfg=cfg(have_cublas_header)");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Is the crate built with the `cuda` feature?
    let cuda_feature_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

    // Check if nvcc is available
    let nvcc_available = Command::new("which")
        .arg("nvcc")
        .status()
        .map(|status| status.success())
        .unwrap_or(false);

        // Prefer a CUDA-supported host compiler; CUDA 12.x accepts gcc <= 12.
        let host_cxx = env::var("CXX")
            .ok()
            .filter(|cxx| Command::new(cxx).arg("--version").status().map(|s| s.success()).unwrap_or(false))
            .or_else(|| which_preferring(&["g++-12", "gcc-12", "g++-11", "gcc-11"]))
            .or_else(|| which_preferring(&["g++", "c++"]))
            .unwrap_or_else(|| "c++".to_string());

    if cuda_feature_enabled {
        if !nvcc_available {
            panic!(
                "nvcc is required when building with the `cuda` feature; install the CUDA toolkit or disable the feature",
            );
        }

        // Expose cfg so tests/examples can detect real CUDA
        println!("cargo:rustc-cfg=nvcc");

        // If available, use conda CUDA include/lib paths
        let conda_prefix = env::var("CONDA_PREFIX").ok();
        let conda_include = conda_prefix.as_ref().map(|p| format!("{}/include", p));
        let conda_lib = conda_prefix.as_ref().map(|p| format!("{}/lib", p));
        // Conda CUDA layout also uses targets/x86_64-linux
        let conda_targets_include = conda_prefix
            .as_ref()
            .map(|p| format!("{}/targets/x86_64-linux/include", p));
        let conda_targets_lib = conda_prefix
            .as_ref()
            .map(|p| format!("{}/targets/x86_64-linux/lib", p));

        // Detect cuBLAS availability: require both header and shared library
        let mut have_cublas_header = false;
        for inc in [conda_targets_include.as_deref(), conda_include.as_deref()]
            .into_iter()
            .flatten()
        {
            let hdr = std::path::Path::new(inc).join("cublas_v2.h");
            if hdr.exists() {
                have_cublas_header = true;
                break;
            }
        }
        // Fallback common locations for headers
        for p in ["/usr/local/cuda/include", "/usr/include"] {
            if !have_cublas_header {
                let hdr = std::path::Path::new(p).join("cublas_v2.h");
                if hdr.exists() {
                    have_cublas_header = true;
                }
            }
        }
        // Detect shared library presence
        let mut have_cublas_lib = false;
        for libp in [conda_targets_lib.as_deref(), conda_lib.as_deref()]
            .into_iter()
            .flatten()
        {
            for name in ["libcublas.so", "libcublas.so.12", "libcublas.so.11"] {
                if std::path::Path::new(libp).join(name).exists() {
                    have_cublas_lib = true;
                    break;
                }
            }
            if have_cublas_lib {
                break;
            }
        }
        for p in [
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
        ] {
            if !have_cublas_lib {
                for name in ["libcublas.so", "libcublas.so.12", "libcublas.so.11"] {
                    if std::path::Path::new(p).join(name).exists() {
                        have_cublas_lib = true;
                    }
                }
            }
        }
        let have_cublas_detected = have_cublas_header && have_cublas_lib;
        let have_cublas = if std::env::var("M40LLM_ENABLE_CUBLAS").ok().as_deref() == Some("1") {
            have_cublas_detected
        } else {
            false
        };

        // Compile CUDA into static lib
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .include("cuda")
            .file("cuda/kernels.cu")
            // Use the detected host compiler while staying within CUDA's
            // supported GCC range.
            .flag(&format!("-ccbin={}", host_cxx))
            // Compile as C++17 and ask the host compiler for the GNU dialect so
            // glibc exposes the _Float* typedefs needed by newer headers.
            .flag("-std=c++17")
            .flag("-Xcompiler")
            .flag("-std=gnu++17")
            .flag("-O3")
            .flag("-Xcompiler")
            .flag("-fPIC")
            .flag("-gencode=arch=compute_52,code=sm_52") // Tesla M40
            .flag("-gencode=arch=compute_52,code=compute_52") // include PTX for JIT on newer GPUs
            .flag("-allow-unsupported-compiler");
        if let Some(ref inc) = conda_include {
            build.include(inc);
        }
        if let Some(ref inc) = conda_targets_include {
            build.include(inc);
        }
        if have_cublas {
            build.define("M40LLM_HAVE_CUBLAS", None);
            // Expose to Rust to allow tests to gate on cuBLAS availability
            println!("cargo:rustc-cfg=have_cublas_header");
        }
        build.compile("m40llm_kernels");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        if let Some(ref libp) = conda_lib {
            println!("cargo:rustc-link-search=native={}", libp);
        }
        if let Some(ref libp) = conda_targets_lib {
            println!("cargo:rustc-link-search=native={}", libp);
        }
        // Embed RPATH so test binaries can locate CUDA/cuBLAS without external LD_LIBRARY_PATH
        // Prefer Conda paths when present
        if have_cublas {
            if let Some(ref libp) = conda_targets_lib {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libp);
            }
            if let Some(ref libp) = conda_lib {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libp);
            }
            // Common system locations as fallback RPATHs
            for p in [
                "/usr/local/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib64",
            ] {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", p);
            }
        }
        println!("cargo:rustc-link-lib=static=m40llm_kernels");
        println!("cargo:rustc-link-lib=cudart");
        if have_cublas {
        println!("cargo:rustc-link-lib=cublas");
        }
    } else {
        // No CUDA feature: nothing to build/link
        let _ = File::create(out_dir.join("kernels.cubin")).unwrap();
        println!("cargo:rustc-link-search=native={}", out_dir.display());
    }
}

fn which_preferring(candidates: &[&str]) -> Option<String> {
    for candidate in candidates {
        if Command::new("which")
            .arg(candidate)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Some(candidate.to_string());
        }
    }
    None
}
