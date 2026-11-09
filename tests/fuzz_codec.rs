use rnbw::{LaneLz, LaneLzEncoder, Rnbw, RnbwView};

const FUZZ_CASES: usize = 12_000;

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn bounded(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next() as usize) % bound
        }
    }
}

#[test]
fn deterministic_mutation_fuzzer_checks_raw_and_entropy_parsers() {
    let mut rng = Rng(0x8f3c_2a17_d451_6b09);
    let mut corpus = Vec::new();

    for width in (2..=32).step_by(2) {
        let mask = if width == 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let target = (1usize << width.min(10)).min(512);
        let mut values = Vec::with_capacity(target);
        for _ in 0..target.saturating_mul(4) {
            values.push((rng.next() as u32) & mask);
        }
        values.sort_unstable();
        values.dedup();

        let encoded = Rnbw::encode_width(&values, width).expect("valid fuzz seed");
        assert_eq!(encoded.decode(), values, "width={width}");
        for &value in &values {
            assert!(encoded.contains(value), "width={width}, value={value}");
        }
        let compressed = encoded.to_compressed_bytes().expect("entropy seed encodes");
        let restored = Rnbw::from_compressed_bytes(&compressed).expect("entropy seed parses");
        assert_eq!(restored.decode(), values, "compressed width={width}");
        corpus.push(encoded.as_bytes().to_vec());
        corpus.push(compressed);
    }

    let edge_sets = [
        Vec::new(),
        vec![u32::MAX],
        (0..=u16::MAX as u32).collect(),
        (0..=u16::MAX as u32).step_by(3).collect(),
    ];
    for values in edge_sets {
        let encoded = Rnbw::encode(&values).expect("edge seed encodes");
        corpus.push(encoded.as_bytes().to_vec());
        corpus.push(
            encoded
                .to_compressed_bytes()
                .expect("edge entropy seed encodes"),
        );
    }

    let interleaved: Vec<u8> = (0..4096)
        .map(|index| ((index / 4) as u8).wrapping_add((index % 4) as u8))
        .collect();
    let lane_stream = LaneLz::encode_interleaved(&interleaved, 4).expect("lane fuzz seed encodes");
    assert_eq!(lane_stream.decode_interleaved().unwrap(), interleaved);
    corpus.push(lane_stream.as_bytes().to_vec());
    let lane_stream =
        LaneLz::encode_interleaved(&interleaved, 16).expect("sixteen-lane SIMD fuzz seed encodes");
    assert_eq!(lane_stream.decode_interleaved().unwrap(), interleaved);
    corpus.push(lane_stream.as_bytes().to_vec());
    let lane_stream = LaneLz::encode(&[&interleaved[..2048], &interleaved[2048..]])
        .expect("multi-lane fuzz seed encodes");
    assert_eq!(
        lane_stream.decode_lanes().unwrap(),
        [&interleaved[..2048], &interleaved[2048..]]
    );
    corpus.push(lane_stream.as_bytes().to_vec());

    let mut exercised = 0usize;
    let mut directed_header_mutations = 0usize;
    for seed in &corpus {
        check_candidate(seed);
        exercised += 1;
        for _ in 0..192 {
            let candidate = mutate(seed, &mut rng);
            check_candidate(&candidate);
            exercised += 1;
        }
        if seed.len() >= 32 && seed.get(0..4) == Some(b"RNBW") {
            for offset in [6usize, 12, 16, 20, 28] {
                let mut candidate = seed.clone();
                if offset + 4 <= candidate.len() {
                    candidate[offset..offset + 4].fill(u8::MAX);
                    check_candidate(&candidate);
                    exercised += 1;
                    directed_header_mutations += 1;
                }
            }
        }
        if seed.len() >= 20 && seed.get(0..4) == Some(b"RNLZ") {
            for offset in [8usize, 12, 16, 20, 24, 28, 32, 36] {
                let mut candidate = seed.clone();
                if offset + 4 <= candidate.len() {
                    candidate[offset..offset + 4].fill(u8::MAX);
                    check_candidate(&candidate);
                    exercised += 1;
                    directed_header_mutations += 1;
                }
            }
        }
    }

    for _ in 0..FUZZ_CASES {
        let len = rng.bounded(1025);
        let mut candidate = vec![0u8; len];
        for byte in &mut candidate {
            *byte = rng.next() as u8;
        }
        check_candidate(&candidate);
        exercised += 1;
    }

    assert_eq!(
        exercised,
        corpus.len() * 193 + FUZZ_CASES + directed_header_mutations
    );
    println!(
        "fuzzed {exercised} deterministic raw, entropy, and multi-lane streams, including {directed_header_mutations} directed header mutations"
    );
}

#[test]
fn randomized_lane_widths_reuse_and_simd_roundtrip() {
    let mut rng = Rng(0x32f1_87d9_6b0a_c45e);
    let mut encoder = LaneLzEncoder::default();
    let widths = [1, 2, 4, 8, 16, 17, 32];
    for case in 0..96 {
        let width = if case % 3 == 0 {
            16
        } else {
            widths[rng.bounded(widths.len())]
        };
        let records = match case % 8 {
            0 => 0,
            1 => 31,
            2 => 32,
            3 => 33,
            _ => rng.bounded(4097),
        };
        let offset = rng.bounded(32);
        let mut input = vec![0u8; offset + width * records];
        for (i, byte) in input[offset..].iter_mut().enumerate() {
            *byte = match case % 4 {
                0 => 0,
                1 => (i % width) as u8,
                2 => (rng.next() & 7) as u8,
                _ => rng.next() as u8,
            };
        }
        let input = &input[offset..];
        let lanes: Vec<Vec<u8>> = (0..width)
            .map(|lane| {
                input
                    .chunks_exact(width)
                    .map(|record| record[lane])
                    .collect()
            })
            .collect();
        let expected =
            LaneLz::encode(&lanes.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        let serial = encoder.encode_interleaved(input, width).unwrap();
        assert_eq!(
            serial.as_bytes(),
            expected.as_bytes(),
            "case={case}, width={width}"
        );
        assert_eq!(serial.decode_interleaved().unwrap(), input);
        let parallel = encoder.encode_interleaved_parallel(input, width).unwrap();
        assert_eq!(parallel.decode_interleaved().unwrap(), input);
    }
}

fn mutate(seed: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut bytes = seed.to_vec();
    match rng.bounded(7) {
        0 => {
            if !bytes.is_empty() {
                let index = rng.bounded(bytes.len());
                bytes[index] ^= 1 << rng.bounded(8);
            }
        }
        1 => {
            if !bytes.is_empty() {
                let index = rng.bounded(bytes.len());
                bytes.truncate(index);
            }
        }
        2 => {
            let count = 1 + rng.bounded(16);
            bytes.reserve(count);
            for _ in 0..count {
                bytes.push(rng.next() as u8);
            }
        }
        3 => {
            if !bytes.is_empty() {
                let index = rng.bounded(bytes.len());
                bytes[index] = 0;
            }
        }
        4 => {
            if !bytes.is_empty() {
                let index = rng.bounded(bytes.len());
                bytes[index] = u8::MAX;
            }
        }
        5 => {
            if !bytes.is_empty() {
                let index = rng.bounded(bytes.len());
                bytes.insert(index, rng.next() as u8);
            }
        }
        _ => {
            if bytes.len() >= 4 {
                let start = rng.bounded(bytes.len() - 3);
                for byte in &mut bytes[start..start + 4] {
                    *byte = rng.next() as u8;
                }
            }
        }
    }
    bytes
}

fn check_candidate(bytes: &[u8]) {
    if let Ok(view) = RnbwView::from_bytes(bytes) {
        let decoded = view.decode();
        assert!(decoded.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(decoded.len(), view.len());
        for value in decoded.iter().step_by(7) {
            assert!(view.contains(*value));
        }
        let mut probe = 0x9e37_79b9u32;
        for _ in 0..8 {
            probe = probe.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            assert_eq!(view.contains(probe), decoded.binary_search(&probe).is_ok());
        }
    }

    if let Ok(decoded) = Rnbw::from_compressed_bytes(bytes) {
        let values = decoded.decode();
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(values.len(), decoded.len());
        for value in values.iter().step_by(7) {
            assert!(decoded.contains(*value));
        }
        let raw = decoded.as_bytes().to_vec();
        assert_eq!(RnbwView::from_bytes(&raw).unwrap().decode(), values);
    }

    if let Ok(lanes) = LaneLz::from_bytes(bytes)
        && lanes.raw_bytes() <= 1 << 20
        && let Ok(decoded) = lanes.decode_lanes()
    {
        assert_eq!(
            decoded.iter().map(Vec::len).sum::<usize>(),
            lanes.raw_bytes()
        );
        if let Some(first) = decoded.first()
            && decoded.iter().all(|lane| lane.len() == first.len())
        {
            let expected: Vec<u8> = (0..first.len())
                .flat_map(|record| decoded.iter().map(move |lane| lane[record]))
                .collect();
            assert_eq!(lanes.decode_interleaved().unwrap(), expected);
        }
    }
}
