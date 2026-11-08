mod datasets;

use rnbw::{PlainEf, Rnbw};
use std::env;
use std::error::Error;
use std::hint::black_box;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        return usage();
    };
    if command == "--help" || command == "help" {
        return usage();
    }
    if command != "bench" {
        return Err(format!("unknown command {command:?}; use `rnbw --help`").into());
    }

    let mut workload = "uniform".to_string();
    let mut operation = "build".to_string();
    let mut engine = "all".to_string();
    let mut size = 1_000_000usize;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--workload" => workload = args.next().ok_or("--workload needs a value")?,
            "--operation" => operation = args.next().ok_or("--operation needs a value")?,
            "--engine" => engine = args.next().ok_or("--engine needs a value")?,
            "--size" => size = args.next().ok_or("--size needs a value")?.parse()?,
            "--quick" => size = 20_000,
            "-h" | "--help" => return usage(),
            other => return Err(format!("unknown option {other:?}").into()),
        }
    }
    if !["build", "contains", "decode"].contains(&operation.as_str()) {
        return Err("--operation must be build, contains, or decode".into());
    }
    let values = datasets::make_dataset(&workload, size)?;
    if values.is_empty() {
        return Err("workload generated no values".into());
    }
    println!(
        "workload={workload} operation={operation} requested={} actual={}",
        size,
        values.len()
    );

    let engines: Vec<&str> = if engine == "all" {
        vec!["rnbw", "plain-ef", "raw"]
    } else {
        vec![engine.as_str()]
    };
    for name in engines {
        run_engine(name, &operation, &values)?;
    }
    Ok(())
}

fn run_engine(engine: &str, operation: &str, values: &[u32]) -> Result<(), Box<dyn Error>> {
    let probes = if operation == "contains" {
        make_probes(values)
    } else {
        Vec::new()
    };
    let expected_hits = probes
        .iter()
        .filter(|&&p| values.binary_search(&p).is_ok())
        .count();
    let (bytes, runtime_index_bytes, hits, decoded) = match engine {
        "rnbw" => {
            let bitmap = Rnbw::encode(values)?;
            let bytes = bitmap.as_bytes().len();
            let hits = if operation == "contains" {
                Some(
                    probes
                        .iter()
                        .filter(|&&p| black_box(bitmap.contains(black_box(p))))
                        .count(),
                )
            } else {
                None
            };
            let decoded = if operation == "decode" {
                Some(bitmap.decode())
            } else {
                None
            };
            (bytes, bitmap.runtime_index_bytes(), hits, decoded)
        }
        "plain-ef" => {
            let ef = PlainEf::encode(values)?;
            let bytes = ef.encoded_size();
            let hits = if operation == "contains" {
                Some(
                    probes
                        .iter()
                        .filter(|&&p| black_box(ef.contains(black_box(p))))
                        .count(),
                )
            } else {
                None
            };
            let decoded = if operation == "decode" {
                Some(ef.decode())
            } else {
                None
            };
            (bytes, 0, hits, decoded)
        }
        "raw" => {
            let raw = values.to_vec();
            let bytes = raw.len() * std::mem::size_of::<u32>();
            let hits = if operation == "contains" {
                Some(
                    probes
                        .iter()
                        .filter(|&&p| black_box(raw.binary_search(&black_box(p)).is_ok()))
                        .count(),
                )
            } else {
                None
            };
            let decoded = if operation == "decode" {
                Some(raw.clone())
            } else {
                None
            };
            (bytes, 0, hits, decoded)
        }
        other => {
            return Err(format!("unknown engine {other:?}; choose rnbw, plain-ef, or raw").into());
        }
    };

    if let Some(decoded) = decoded
        && decoded.as_slice() != values
    {
        return Err(format!("{engine} decode did not match input").into());
    }
    if let Some(hits) = hits
        && hits != expected_hits
    {
        return Err(format!("{engine} returned {hits} hits; expected {expected_hits}").into());
    }
    println!(
        "engine={engine} encoded_bytes={bytes} runtime_index_bytes={runtime_index_bytes} bytes_per_key={:.4} probes={} hits={}",
        bytes as f64 / values.len() as f64,
        probes.len(),
        expected_hits
    );
    Ok(())
}

fn make_probes(values: &[u32]) -> Vec<u32> {
    let sample_count = values.len().min(4096);
    let mut probes = Vec::with_capacity(sample_count * 2);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..sample_count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        probes.push(values[(state as usize) % values.len()]);
    }
    while probes.len() < sample_count * 2 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let candidate = state as u32;
        if values.binary_search(&candidate).is_err() {
            probes.push(candidate);
        }
    }
    probes
}

fn usage() -> Result<(), Box<dyn Error>> {
    println!(
        "RNBW adaptive u32 set codec\n\nUsage:\n  rnbw bench [--workload uniform|clustered|dense|ecs|morton-shell] [--size N] [--engine all|rnbw|plain-ef|raw] [--operation build|contains|decode]\n  rnbw bench --quick\n\nThe CLI reports encoded size and verifies requested contains/decode work. Hyperfine measures full process time; Criterion isolates operations. External codecs are benchmark-only."
    );
    Ok(())
}
