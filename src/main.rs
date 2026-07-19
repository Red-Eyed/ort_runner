use anyhow::{Context, Result};
use clap::Parser;

use ort_runner::cli::Cli;
use ort_runner::{dylib, info, model, session};

fn main() {
    if let Err(err) = run() {
        // `{err:#}` renders the whole anyhow context chain, so a failure deep in the stack still
        // explains which step it happened in.
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    let dylib_path = dylib::resolve(cli.ort_dylib.as_deref())?;
    ort::init_from(&dylib_path)
        .with_context(|| format!("loading ONNX Runtime from {}", dylib_path.display()))?
        .commit();

    if cli.version {
        println!("ort_runner {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    if cli.info {
        info::print(&dylib_path)?;
        return Ok(());
    }

    let session = session::build(&cli)?;
    let inputs = model::describe_inputs(&session, &cli.dim_overrides(), cli.default_dim)?;
    let outputs = model::describe_outputs(&session)?;

    for (name, value) in model::unmatched_dim_overrides(&inputs, &cli.dim_overrides()) {
        eprintln!(
            "warning: --dim {name}={value} does not match any symbolic dimension name in this \
             model's inputs"
        );
    }

    if cli.list_io {
        print_io(&inputs, &outputs);
        return Ok(());
    }

    println!("(benchmark not wired up yet -- phase 4)");
    Ok(())
}

fn print_io(inputs: &[model::InputSpec], outputs: &[model::OutputSpec]) {
    println!("inputs:");
    for spec in inputs {
        println!(
            "  {}: shape={} dtype={} symbolic_dims={}",
            spec.name,
            format_shape(&spec.declared_shape),
            model::element_type_name(spec.element_type),
            format_symbolic_dims(&spec.symbolic_dims),
        );
    }
    println!("outputs:");
    for spec in outputs {
        println!(
            "  {}: shape={} dtype={}",
            spec.name,
            format_shape(&spec.shape),
            model::element_type_name(spec.element_type),
        );
    }
}

fn format_shape(shape: &[i64]) -> String {
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    format!("[{}]", dims.join(", "))
}

/// Anonymous axes carry an empty name, so they are dropped rather than printed as blanks --
/// the list exists to show which names `--dim` can target.
fn format_symbolic_dims(symbolic_dims: &[String]) -> String {
    let named: Vec<&str> =
        symbolic_dims.iter().filter(|n| !n.is_empty()).map(String::as_str).collect();
    if named.is_empty() {
        "(none)".to_string()
    } else {
        named.join(", ")
    }
}
