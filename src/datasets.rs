//! Deterministic benchmark data generators shared by the CLI and Criterion.

use std::collections::HashSet;

pub fn make_dataset(name: &str, requested: usize) -> Result<Vec<u32>, String> {
    let mut values = match name {
        "uniform" => uniform(requested),
        "clustered" => clustered(requested),
        "dense" => dense(requested),
        "ecs" => ecs(requested),
        "morton-shell" => morton_shell(requested),
        _ => return Err(format!("unknown workload {name:?}")),
    };
    values.sort_unstable();
    values.dedup();
    Ok(values)
}

fn uniform(count: usize) -> Vec<u32> {
    let mut state = 0x4d59_5df4_d0f3_3173u64;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 16) as u32
        })
        .collect()
}

fn clustered(count: usize) -> Vec<u32> {
    let mut values = Vec::with_capacity(count);
    let mut cluster = 0u32;
    while values.len() < count {
        let base = cluster << 16;
        let in_cluster = (count - values.len()).min(64);
        values.extend((0..in_cluster as u32).map(|offset| base + offset));
        cluster += 1;
    }
    values
}

fn dense(count: usize) -> Vec<u32> {
    let base = 0x4000_0000u32;
    (0..count).map(|i| base.wrapping_add(i as u32)).collect()
}

fn ecs(count: usize) -> Vec<u32> {
    const BASE: u32 = 0x4000_0000;
    let universe = count.saturating_add(count.div_ceil(9)).saturating_add(1);
    (0..universe)
        .filter(|id| id % 10 != 9)
        .take(count)
        .map(|id| BASE.wrapping_add(id as u32))
        .collect()
}

fn morton_shell(count: usize) -> Vec<u32> {
    let mut state = 0x726e_6277_7368_656cu64;
    let mut values = HashSet::with_capacity(count);
    let mut attempts = 0usize;
    while values.len() < count && attempts < count.saturating_mul(64).max(1024) {
        attempts += 1;
        let x = next(&mut state) & 1023;
        let y = next(&mut state) & 1023;
        let z = next(&mut state) & 1023;
        let dx = x as i32 - 512;
        let dy = y as i32 - 512;
        let dz = z as i32 - 512;
        let radius_squared = dx * dx + dy * dy + dz * dz;
        if (190_000..=225_000).contains(&radius_squared) {
            values.insert(morton3(x, y, z));
        }
    }
    values.into_iter().collect()
}

fn next(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 16) as u32
}

fn morton3(x: u32, y: u32, z: u32) -> u32 {
    fn spread(mut value: u32) -> u32 {
        value &= 0x3ff;
        value = (value | value << 16) & 0x0300_00ff;
        value = (value | value << 8) & 0x0300_f00f;
        value = (value | value << 4) & 0x030c_30c3;
        value = (value | value << 2) & 0x0924_9249;
        value
    }
    spread(x) | (spread(y) << 1) | (spread(z) << 2)
}

#[cfg(test)]
mod tests {
    #[test]
    fn workloads_are_repeatable_sorted_and_unique() {
        for name in ["uniform", "clustered", "dense", "ecs", "morton-shell"] {
            let first = super::make_dataset(name, 20_000).unwrap();
            let second = super::make_dataset(name, 20_000).unwrap();
            assert_eq!(first, second, "{name}");
            assert!(!first.is_empty(), "{name}");
            assert!(first.windows(2).all(|pair| pair[0] < pair[1]), "{name}");
            if name != "morton-shell" {
                assert_eq!(first.len(), 20_000, "{name}");
            }
        }
    }

    #[test]
    fn ecs_workload_is_dense_with_ten_percent_holes() {
        let values = super::make_dataset("ecs", 20_000).unwrap();
        let span = (values.last().unwrap() - values[0] + 1) as usize;
        let density = values.len() as f64 / span as f64;
        assert!((0.899..=0.901).contains(&density), "density={density}");
    }

    #[test]
    fn unknown_workload_is_an_error() {
        assert!(super::make_dataset("missing", 10).is_err());
    }
}
