use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/inference/rocm/kernels.cpp");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=PYTORCH_ROCM_ARCH");
    println!("cargo:rerun-if-env-changed=MESHNET_ROCM_ARCH");

    if env::var_os("CARGO_FEATURE_ROCM").is_none() {
        return;
    }
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "linux" {
        panic!("the Mesh ROCm backend is only supported on Linux targets");
    }

    let rocm_path = env::var_os("ROCM_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"));
    let hipcc = rocm_path.join("bin").join("hipcc");
    let hipcc = if hipcc.exists() {
        hipcc
    } else {
        PathBuf::from("hipcc")
    };
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is required"));
    let object = out_dir.join("meshnet_rocm_kernels.o");
    let archive = out_dir.join("libmeshnet_rocm_kernels.a");
    let arch = resolve_rocm_arch();

    let status = Command::new(&hipcc)
        .arg("-x")
        .arg("hip")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-ffast-math")
        .arg(format!("--offload-arch={arch}"))
        .arg("-D__AMDGCN_WAVEFRONT_SIZE=32")
        .arg("-I")
        .arg(rocm_path.join("include"))
        .arg("-c")
        .arg("src/inference/rocm/kernels.cpp")
        .arg("-o")
        .arg(&object)
        .status()
        .expect("failed to invoke hipcc for Mesh ROCm kernels");
    if !status.success() {
        panic!("hipcc failed to compile Mesh ROCm kernels for {arch}");
    }

    let status = Command::new("ar")
        .arg("crus")
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("failed to invoke ar for Mesh ROCm kernels");
    if !status.success() {
        panic!("failed to archive Mesh ROCm kernels");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=meshnet_rocm_kernels");
    println!(
        "cargo:rustc-link-search=native={}",
        rocm_path.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        rocm_path.join("lib").display()
    );
}

fn resolve_rocm_arch() -> String {
    if let Some(arch) = env::var_os("MESHNET_ROCM_ARCH")
        .or_else(|| env::var_os("PYTORCH_ROCM_ARCH"))
        .and_then(|value| value.into_string().ok())
        .and_then(|value| value.split(';').next().map(str::trim).map(str::to_owned))
        .filter(|value| !value.is_empty())
    {
        return arch;
    }

    command_output("rocm_agent_enumerator")
        .and_then(first_gfx_line)
        .or_else(|| command_output("rocminfo").and_then(first_gfx_token))
        .unwrap_or_else(|| "gfx1151".to_string())
}

fn command_output(program: &str) -> Option<String> {
    Command::new(program)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
}

fn first_gfx_line(output: String) -> Option<String> {
    output
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("gfx"))
        .map(str::to_owned)
}

fn first_gfx_token(output: String) -> Option<String> {
    output
        .split(|ch: char| ch.is_ascii_whitespace() || ch == ',')
        .map(str::trim)
        .find(|token| token.starts_with("gfx"))
        .map(clean_gfx_token)
}

fn clean_gfx_token(token: &str) -> String {
    token
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric())
        .collect()
}
