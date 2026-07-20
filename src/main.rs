use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use ort_runner::cli::Cli;
use ort_runner::{
    bench, config, dylib, host, info, model, profile, report, session, stats, tensors,
};

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
        let system_info = info::gather(info::platform::probe().as_ref(), &dylib_path)?;
        info::render::render(&system_info, cli.output_format)?;
        return Ok(());
    }

    let model_path = cli.model_path()?;

    // Resolved before the session exists because ONNX Runtime takes the destination at build time
    // and creates no directory of its own. Absent unless asked for: no --profile, no directory.
    let profile_destination = if cli.profile {
        Some(profile::destination(model_path)?)
    } else {
        None
    };
    let run_config = config::RunConfig::new(&cli, profile_destination);

    // Taken before the session exists, so the difference from the next reading isolates what the
    // model costs from the constant the runtime and libc already occupy.
    let baseline_memory = host::snapshot();

    let load_started = Instant::now();
    let mut session = session::build(model_path, &run_config)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let session_memory = host::snapshot();

    let input_specs = model::describe_inputs(&session, &cli.dim_overrides(), cli.default_dim)?;
    let outputs = model::describe_outputs(&session)?;

    for (name, value) in model::unmatched_dim_overrides(&input_specs, &cli.dim_overrides()) {
        eprintln!(
            "warning: --dim {name}={value} does not match any symbolic dimension name in this \
             model's inputs"
        );
    }

    if cli.list_io {
        print_io(&input_specs, &outputs);
        return Ok(());
    }

    let prepared = tensors::prepare_inputs(
        &input_specs,
        cli.inputs.as_deref(),
        tensors::SynthOptions {
            fill: cli.fill,
            seed: cli.seed,
            int_max: cli.int_fill_max,
        },
    )?;

    let bench_config = config::BenchConfig::from(&cli);
    let timings = bench::run(&mut session, &prepared.inputs, bench_config)?;
    let complete_memory = host::snapshot();

    // After the snapshot: flushing the trace allocates, and that cost belongs to the profiler
    // rather than to the model the report is about.
    let profile_path = match &run_config.profile {
        Some(_) => Some(profile::finish(&mut session)?),
        None => None,
    };

    // Non-empty by construction: --iterations is rejected at zero by the parser, so this is the
    // "cannot happen" branch rather than a case the CLI can reach.
    let measured = stats::summarize(&timings.measured_ms)
        .context("the measured iterations produced no samples")?;

    let report = report::BenchReport {
        created_at: jiff::Timestamp::now()
            .strftime("%Y-%m-%dT%H:%M:%SZ")
            .to_string(),
        command_line: std::env::args().collect(),
        model_path: model_path.display().to_string(),
        model: model::describe_model(&session, model_path),
        input_source: prepared.source,
        inputs: input_specs,
        outputs,
        config: run_config,
        bench_config,
        system: info::gather(info::platform::probe().as_ref(), &dylib_path)?,
        load_ms,
        measured,
        warmup: stats::summarize(&timings.warmup_ms),
        timings,
        memory: report::MemoryReport::new(
            baseline_memory,
            session_memory,
            complete_memory,
            host::peak_rss_bytes(),
        ),
        profile_path: profile_path.map(|path| path.display().to_string()),
    };

    // Both destinations always run: stdout for the person watching, JSON for everything else.
    // Ordering matters only in that the JSON reporter prints where it wrote.
    for reporter in [
        &report::human::HumanReporter as &dyn report::Reporter,
        &report::json::JsonReporter::beside_executable()?,
    ] {
        reporter.report(&report)?;
    }

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
    let named: Vec<&str> = symbolic_dims
        .iter()
        .filter(|n| !n.is_empty())
        .map(String::as_str)
        .collect();
    if named.is_empty() {
        "(none)".to_string()
    } else {
        named.join(", ")
    }
}
