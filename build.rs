use std::{
	borrow::Cow,
	env, fs,
	io::{self, Read, Write},
	path::{Path, PathBuf},
	str::FromStr
};

const ORT_VERSION: &str = "1.11.0";
const ORT_RELEASE_BASE_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download";
const ORT_ENV_STRATEGY: &str = "ORT_STRATEGY";
const ORT_ENV_SYSTEM_LIB_LOCATION: &str = "ORT_LIB_LOCATION";
const ORT_PREBUILD_EXTRACT_DIR: &str = "onnxruntime";

trait OnnxPrebuiltArchive {
	fn as_onnx_str(&self) -> Cow<str>;
}

#[derive(Debug)]
enum Architecture {
	X86,
	X86_64,
	Arm,
	Arm64
}

impl FromStr for Architecture {
	type Err = String;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s {
			"x86" => Ok(Architecture::X86),
			"x86_64" => Ok(Architecture::X86_64),
			"arm" => Ok(Architecture::Arm),
			"aarch64" => Ok(Architecture::Arm64),
			_ => Err(format!("Unsupported architecture: {}", s))
		}
	}
}

impl OnnxPrebuiltArchive for Architecture {
	fn as_onnx_str(&self) -> Cow<str> {
		match self {
			Architecture::X86 => "x86".into(),
			Architecture::X86_64 => "x64".into(),
			Architecture::Arm => "arm".into(),
			Architecture::Arm64 => "arm64".into()
		}
	}
}

#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum Os {
	Windows,
	Linux,
	MacOS
}

impl Os {
	fn archive_extension(&self) -> &'static str {
		match self {
			Os::Windows => "zip",
			Os::Linux => "tgz",
			Os::MacOS => "tgz"
		}
	}
}

impl FromStr for Os {
	type Err = String;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s {
			"windows" => Ok(Os::Windows),
			"linux" => Ok(Os::Linux),
			"macos" => Ok(Os::MacOS),
			_ => Err(format!("Unsupported OS: {}", s))
		}
	}
}

impl OnnxPrebuiltArchive for Os {
	fn as_onnx_str(&self) -> Cow<str> {
		match self {
			Os::Windows => "win".into(),
			Os::Linux => "linux".into(),
			Os::MacOS => "osx".into()
		}
	}
}

#[derive(Debug)]
enum Accelerator {
	None,
	Gpu
}

impl OnnxPrebuiltArchive for Accelerator {
	fn as_onnx_str(&self) -> Cow<str> {
		match self {
			Accelerator::None => "unaccelerated".into(),
			Accelerator::Gpu => "gpu".into()
		}
	}
}

#[derive(Debug)]
struct Triplet {
	os: Os,
	arch: Architecture,
	accelerator: Accelerator
}

impl OnnxPrebuiltArchive for Triplet {
	fn as_onnx_str(&self) -> Cow<str> {
		match (&self.os, &self.arch, &self.accelerator) {
			(Os::Windows, Architecture::X86, Accelerator::None)
			| (Os::Windows, Architecture::X86_64, Accelerator::None)
			| (Os::Windows, Architecture::Arm, Accelerator::None)
			| (Os::Windows, Architecture::Arm64, Accelerator::None)
			| (Os::Linux, Architecture::X86_64, Accelerator::None)
			| (Os::MacOS, Architecture::Arm64, Accelerator::None) => format!("{}-{}", self.os.as_onnx_str(), self.arch.as_onnx_str()).into(),
			// for some reason, arm64/Linux uses `aarch64` instead of `arm64`
			(Os::Linux, Architecture::Arm64, Accelerator::None) => format!("{}-{}", self.os.as_onnx_str(), "aarch64").into(),
			// for another odd reason, x64/macOS uses `x86_64` instead of `x64`
			(Os::MacOS, Architecture::X86_64, Accelerator::None) => format!("{}-{}", self.os.as_onnx_str(), "x86_64").into(),
			(Os::Windows, Architecture::X86_64, Accelerator::Gpu) | (Os::Linux, Architecture::X86_64, Accelerator::Gpu) => {
				format!("{}-{}-{}", self.os.as_onnx_str(), self.arch.as_onnx_str(), self.accelerator.as_onnx_str()).into()
			}
			_ => panic!(
				"Microsoft does not provide ONNX Runtime downloads for triplet: {}-{}-{}; you may have to use the `system` strategy instead",
				self.os.as_onnx_str(),
				self.arch.as_onnx_str(),
				self.accelerator.as_onnx_str()
			)
		}
	}
}

fn prebuilt_archive_url() -> (PathBuf, String) {
	let accelerator = if cfg!(feature = "cuda") { Accelerator::Gpu } else { Accelerator::None };

	let triplet = Triplet {
		os: env::var("CARGO_CFG_TARGET_OS").expect("unable to get target OS").parse().unwrap(),
		arch: env::var("CARGO_CFG_TARGET_ARCH").expect("unable to get target arch").parse().unwrap(),
		accelerator
	};

	let prebuilt_archive = format!("onnxruntime-{}-{}.{}", triplet.as_onnx_str(), ORT_VERSION, triplet.os.archive_extension());
	let prebuilt_url = format!("{}/v{}/{}", ORT_RELEASE_BASE_URL, ORT_VERSION, prebuilt_archive);

	(PathBuf::from(prebuilt_archive), prebuilt_url)
}

fn download<P>(source_url: &str, target_file: P)
where
	P: AsRef<Path>
{
	let resp = ureq::get(source_url)
		.timeout(std::time::Duration::from_secs(300))
		.call()
		.unwrap_or_else(|err| panic!("ERROR: Failed to download {}: {:?}", source_url, err));

	let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
	let mut reader = resp.into_reader();
	// FIXME: Save directly to the file
	let mut buffer = vec![];
	let read_len = reader.read_to_end(&mut buffer).unwrap();
	assert_eq!(buffer.len(), len);
	assert_eq!(buffer.len(), read_len);

	let f = fs::File::create(&target_file).unwrap();
	let mut writer = io::BufWriter::new(f);
	writer.write_all(&buffer).unwrap();
}

fn extract_archive(filename: &Path, output: &Path) {
	match filename.extension().map(|e| e.to_str()) {
		#[cfg(target_os = "windows")]
		Some(Some("zip")) => extract_zip(filename, output),
		#[cfg(not(target_os = "windows"))]
		Some(Some("tgz")) => extract_tgz(filename, output),
		_ => unimplemented!()
	}
}

#[cfg(not(target_os = "windows"))]
fn extract_tgz(filename: &Path, output: &Path) {
	let file = fs::File::open(&filename).unwrap();
	let buf = io::BufReader::new(file);
	let tar = flate2::read::GzDecoder::new(buf);
	let mut archive = tar::Archive::new(tar);
	archive.unpack(output).unwrap();
}

#[cfg(target_os = "windows")]
fn extract_zip(filename: &Path, outpath: &Path) {
	let file = fs::File::open(&filename).unwrap();
	let buf = io::BufReader::new(file);
	let mut archive = zip::ZipArchive::new(buf).unwrap();
	for i in 0..archive.len() {
		let mut file = archive.by_index(i).unwrap();
		#[allow(deprecated)]
		let outpath = outpath.join(file.sanitized_name());
		if !(&*file.name()).ends_with('/') {
			println!("File {} extracted to \"{}\" ({} bytes)", i, outpath.as_path().display(), file.size());
			if let Some(p) = outpath.parent() {
				if !p.exists() {
					fs::create_dir_all(&p).unwrap();
				}
			}
			let mut outfile = fs::File::create(&outpath).unwrap();
			io::copy(&mut file, &mut outfile).unwrap();
		}
	}
}

fn prepare_libort_dir() -> PathBuf {
	let strategy = env::var(ORT_ENV_STRATEGY);
	println!("[glide] strategy: {:?}", strategy.as_ref().map(String::as_str).unwrap_or_else(|_| "unknown"));
	match strategy.as_ref().map(String::as_str) {
		Ok("download") | Err(_) => {
			let (prebuilt_archive, prebuilt_url) = prebuilt_archive_url();

			let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
			let extract_dir = out_dir.join(ORT_PREBUILD_EXTRACT_DIR);
			let downloaded_file = out_dir.join(&prebuilt_archive);

			println!("cargo:rerun-if-changed={}", downloaded_file.display());

			if !downloaded_file.exists() {
				fs::create_dir_all(&out_dir).unwrap();
				download(&prebuilt_url, &downloaded_file);
			}

			if !extract_dir.exists() {
				extract_archive(&downloaded_file, &extract_dir);
			}

			extract_dir.join(prebuilt_archive.file_stem().unwrap())
		}
		Ok("system") => PathBuf::from(match env::var(ORT_ENV_SYSTEM_LIB_LOCATION) {
			Ok(p) => p,
			Err(e) => {
				panic!("[glide] system strategy requires ORT_LIB_LOCATION env var to be set: {}", e);
			}
		}),
		Ok("compile") => unimplemented!(),
		_ => panic!("[glide] unknown strategy: {} (valid options are `download` or `system`)", strategy.unwrap_or_else(|_| "unknown".to_string()))
	}
}

#[cfg(not(feature = "generate-bindings"))]
fn generate_bindings(_include_dir: &Path) {
	println!("[glide] bindings not generated automatically; using committed bindings instead.");
	println!("[glide] enable the `generate-bindings` feature to generate fresh bindings.");
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings(include_dir: &Path) {
	let clang_args = &[
		format!("-I{}", include_dir.display()),
		format!("-I{}", include_dir.join("onnxruntime").join("core").join("session").display())
	];

	println!("cargo:rerun-if-changed=src/onnx/wrapper.h");
	println!("cargo:rerun-if-changed=src/generated/bindings.rs");

	let bindings = bindgen::Builder::default()
        .header("src/onnx/wrapper.h")
        .clang_args(clang_args)
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Set `size_t` to be translated to `usize` for win32 compatibility.
        .size_t_is_usize(true)
        // Format using rustfmt
        .rustfmt_bindings(true)
        .rustified_enum("*")
        .generate()
        .expect("Unable to generate bindings");

	// Write the bindings to (source controlled) src/onnx/bindings/<os>/<arch>/bindings.rs
	let generated_file = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
		.join("src")
		.join("onnx")
		.join("bindings")
		.join(env::var("CARGO_CFG_TARGET_OS").unwrap())
		.join(env::var("CARGO_CFG_TARGET_ARCH").unwrap())
		.join("bindings.rs");
	println!("cargo:rerun-if-changed={:?}", generated_file);
	fs::create_dir_all(generated_file.parent().unwrap()).unwrap();
	bindings.write_to_file(&generated_file).expect("Couldn't write bindings!");
}

#[cfg(feature = "disable-sys-build-script")]
fn main() {}

#[cfg(not(feature = "disable-sys-build-script"))]
fn main() {
	let install_dir = prepare_libort_dir();

	let include_dir = install_dir.join("include");
	let lib_dir = install_dir.join("lib");

	println!("cargo:rustc-link-lib=onnxruntime");
	println!("cargo:rustc-link-search=native={}", lib_dir.display());

	println!("cargo:rerun-if-env-changed={}", ORT_ENV_STRATEGY);
	println!("cargo:rerun-if-env-changed={}", ORT_ENV_SYSTEM_LIB_LOCATION);

	generate_bindings(&include_dir);
}
