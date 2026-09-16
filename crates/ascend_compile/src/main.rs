use std::path::PathBuf;
use std::process;

use ascend_compile::compiler::CompileConfig;
use ascend_compile::target::{AscendTarget, FlagStyle, OutputFormat};
use ascend_compile::validate::Severity;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        process::exit(0);
    }

    let opts = match parse_args(&args[1..]) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    };

    // Parse target
    let target = match AscendTarget::from_soc_version(&opts.soc) {
        Some(t) => t,
        None => {
            eprintln!("error: unknown SoC version: {}", opts.soc);
            process::exit(1);
        }
    };

    let mut config = CompileConfig::new(target);
    config.validate = !opts.no_validate;
    config.auto_sync = !opts.no_auto_sync;
    config.opt_level = opts.opt_level;
    config.extra_includes = opts.includes;
    config.extra_defines = opts.defines;
    config.extra_libs = opts.libs;

    if opts.shared {
        config.output_format = OutputFormat::SharedLib;
    }

    if let Some(style) = opts.flag_style {
        config.flag_style = style;
    }

    // Validate-only mode
    if opts.validate_only {
        let source = match std::fs::read_to_string(&opts.input) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot read {}: {}", opts.input.display(), e);
                process::exit(1);
            }
        };
        let diags = ascend_compile::validate_kernel(&source, target);
        if diags.is_empty() {
            println!("ok: no issues found");
        } else {
            for d in &diags {
                eprintln!("{}", d);
            }
            let has_errors = diags.iter().any(|d| d.severity == Severity::Error);
            process::exit(if has_errors { 1 } else { 0 });
        }
        return;
    }

    // Determine output path
    let output = opts.output.unwrap_or_else(|| {
        let ext = match config.output_format {
            OutputFormat::Object => "o",
            OutputFormat::SharedLib => "so",
        };
        opts.input.with_extension(ext)
    });

    match ascend_compile::compile_kernel_file(&opts.input, &output, &config) {
        Ok(()) => {
            eprintln!("compiled: {}", output.display());
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

struct Opts {
    input: PathBuf,
    output: Option<PathBuf>,
    soc: String,
    shared: bool,
    opt_level: u8,
    includes: Vec<String>,
    defines: Vec<String>,
    libs: Vec<String>,
    no_validate: bool,
    no_auto_sync: bool,
    validate_only: bool,
    flag_style: Option<FlagStyle>,
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts {
        input: PathBuf::new(),
        output: None,
        soc: "Ascend910B3".to_string(),
        shared: false,
        opt_level: 2,
        includes: Vec::new(),
        defines: Vec::new(),
        libs: Vec::new(),
        no_validate: false,
        no_auto_sync: false,
        validate_only: false,
        flag_style: None,
    };

    let mut i = 0;
    let mut input_set = false;

    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                opts.output = Some(PathBuf::from(args.get(i).ok_or("-o requires an argument")?));
            }
            "--soc" => {
                i += 1;
                opts.soc = args.get(i).ok_or("--soc requires an argument")?.clone();
            }
            "--shared" => {
                opts.shared = true;
            }
            "--opt-level" => {
                i += 1;
                opts.opt_level = args
                    .get(i)
                    .ok_or("--opt-level requires an argument")?
                    .parse::<u8>()
                    .map_err(|_| "opt-level must be 0-3")?
                    .min(3);
            }
            "--no-validate" => {
                opts.no_validate = true;
            }
            "--no-auto-sync" => {
                opts.no_auto_sync = true;
            }
            "--validate-only" => {
                opts.validate_only = true;
            }
            "--flag-style" => {
                i += 1;
                let s = args.get(i).ok_or("--flag-style requires an argument")?;
                opts.flag_style = Some(match s.as_str() {
                    "cce" => FlagStyle::CceAicore,
                    "npu" => FlagStyle::NpuArch,
                    _ => return Err(format!("unknown flag style: {} (use 'cce' or 'npu')", s)),
                });
            }
            s if s.starts_with("-I") => {
                if s.len() > 2 {
                    opts.includes.push(s[2..].to_string());
                } else {
                    i += 1;
                    opts.includes
                        .push(args.get(i).ok_or("-I requires an argument")?.clone());
                }
            }
            s if s.starts_with("-D") => {
                if s.len() > 2 {
                    opts.defines.push(s[2..].to_string());
                } else {
                    i += 1;
                    opts.defines
                        .push(args.get(i).ok_or("-D requires an argument")?.clone());
                }
            }
            s if s.starts_with("-l") => {
                if s.len() > 2 {
                    opts.libs.push(s[2..].to_string());
                } else {
                    i += 1;
                    opts.libs
                        .push(args.get(i).ok_or("-l requires an argument")?.clone());
                }
            }
            s if s.starts_with('-') => {
                return Err(format!("unknown option: {}", s));
            }
            _ => {
                if input_set {
                    return Err("multiple input files not supported".to_string());
                }
                opts.input = PathBuf::from(&args[i]);
                input_set = true;
            }
        }
        i += 1;
    }

    if !input_set {
        return Err("no input file specified".to_string());
    }

    Ok(opts)
}

fn print_usage() {
    eprintln!(
        "Usage: ascend-compile [OPTIONS] <INPUT>

Compile AscendC C++ kernels for Ascend NPU targets.

Options:
  -o, --output <PATH>    Output file path
  --soc <SOC>            Target SoC (default: Ascend910B3)
  --shared               Produce shared library instead of object
  --opt-level <N>        Optimization level 0-3 (default: 2)
  -I <PATH>              Additional include path (repeatable)
  -D <DEFINE>            Preprocessor define (repeatable)
  -l <LIB>               Link library (repeatable, SharedLib only)
  --no-validate          Skip validation passes
  --no-auto-sync         Disable --cce-auto-sync
  --validate-only        Only validate, don't compile
  --flag-style <STYLE>   Force flag style: cce or npu"
    );
}
