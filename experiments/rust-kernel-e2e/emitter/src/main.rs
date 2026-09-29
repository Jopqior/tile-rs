use std::{env, fs, process};
use tile_codegen::{EmitOpts, TargetRegistry};

fn main() {
    let mut args = env::args().skip(1);
    let input = args.next().expect("usage: emitter <real.mlir> <output.metal>");
    let output = args.next().expect("missing MSL output path");
    assert!(args.next().is_none(), "unexpected extra argument");
    let mlir = fs::read_to_string(&input).expect("read actual frontend MLIR");
    let registry = TargetRegistry::with_builtin();
    let target = registry.select("msl").expect("fixed-source msl not registered");
    match target.emit(&mlir, &EmitOpts::default()) {
        Ok(result) if !result.source.trim().is_empty() && result.ext == "metal" => {
            fs::write(&output, &result.source).expect("write MSL");
            println!("accepted {} bytes MLIR; wrote {} bytes to {} from registry msl", mlir.len(), result.source.len(), output);
        }
        Ok(result) => {
            eprintln!("invalid emitter output: extension={} bytes={}", result.ext, result.source.len());
            process::exit(2);
        }
        Err(error) => {
            eprintln!("fixed-source registry msl rejected actual frontend MLIR: {error}");
            process::exit(1);
        }
    }
}
