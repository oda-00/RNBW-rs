//! Vulkan compute backend for the fixed-width per-lane LZ stage.
//!
//! Entropy coding stays in the shared RNBW codec so CPU and GPU paths write
//! the same file format. One workgroup processes one semantic lane, matching
//! the specified one-window-per-lane model.

use super::*;
use std::sync::mpsc;

const ENCODE_SHADER: &str = include_str!("encode.wgsl");
const DECODE_SHADER: &str = include_str!("decode.wgsl");
const HASH_SIZE: usize = 1 << 16;
const RESTART_MAGIC: &[u8; 4] = b"RNRB";
const RESTART_VERSION: u16 = 1;
const RESTART_HEADER_BYTES: usize = 16;

/// Independently restartable RNLZ blocks. Each source lane is divided into
/// fixed-size chunks; the envelope records the original lane boundaries.
#[derive(Clone, Debug)]
pub struct LaneLzRestarted {
    image: Vec<u8>,
}

impl LaneLzRestarted {
    /// Builds a restart-blocked CPU codec image with the same framing used by
    /// the GPU encoder. Each block has an independent 64 KiB LZ window.
    pub fn encode_cpu(lanes: &[&[u8]], block_size: usize) -> Result<Self, CodecError> {
        let (blocks, lengths) = split_restart_blocks(lanes, block_size)?;
        let codec = LaneLz::encode(&blocks)?;
        Self::from_codec(&codec, &lengths, block_size)
    }

    /// Parses and validates the restart envelope and its embedded RNLZ image.
    pub fn from_bytes(image: &[u8]) -> Result<Self, CodecError> {
        let (codec, lengths, block_size) = parse_restart_image(image)?;
        validate_restart_descriptors(&codec, &lengths, block_size)?;
        Ok(Self {
            image: image.to_vec(),
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.image
    }

    pub fn raw_bytes(&self) -> Result<usize, CodecError> {
        parse_restart_lengths(&self.image)?
            .iter()
            .try_fold(0usize, |sum, &len| {
                sum.checked_add(len).ok_or(CodecError::LengthOverflow)
            })
    }

    pub fn decode_lanes(&self) -> Result<Vec<Vec<u8>>, CodecError> {
        let (codec, lengths, block_size) = parse_restart_image(&self.image)?;
        let blocks = codec.decode_lanes()?;
        reassemble_restart_blocks(blocks, &lengths, block_size)
    }

    fn from_codec(
        codec: &LaneLz,
        lengths: &[usize],
        block_size: usize,
    ) -> Result<Self, CodecError> {
        let block_size = u32::try_from(block_size).map_err(|_| CodecError::LengthOverflow)?;
        let lane_count = u32::try_from(lengths.len()).map_err(|_| CodecError::LengthOverflow)?;
        let capacity = RESTART_HEADER_BYTES
            .checked_add(
                lengths
                    .len()
                    .checked_mul(8)
                    .ok_or(CodecError::LengthOverflow)?,
            )
            .and_then(|n| n.checked_add(codec.as_bytes().len()))
            .ok_or(CodecError::LengthOverflow)?;
        let mut image = Vec::with_capacity(capacity);
        image.extend_from_slice(RESTART_MAGIC);
        image.extend_from_slice(&RESTART_VERSION.to_le_bytes());
        image.extend_from_slice(&0u16.to_le_bytes());
        image.extend_from_slice(&lane_count.to_le_bytes());
        image.extend_from_slice(&block_size.to_le_bytes());
        for &len in lengths {
            image.extend_from_slice(
                &u64::try_from(len)
                    .map_err(|_| CodecError::LengthOverflow)?
                    .to_le_bytes(),
            );
        }
        image.extend_from_slice(codec.as_bytes());
        Ok(Self { image })
    }
}

fn split_restart_blocks<'a>(
    lanes: &[&'a [u8]],
    block_size: usize,
) -> Result<(Vec<&'a [u8]>, Vec<usize>), CodecError> {
    if block_size == 0 || block_size > u32::MAX as usize || lanes.is_empty() {
        return Err(CodecError::InvalidFormat(
            "invalid RNBW restart block configuration",
        ));
    }
    let mut blocks: Vec<&'a [u8]> = Vec::new();
    let lengths: Vec<usize> = lanes.iter().map(|lane| lane.len()).collect();
    for lane in lanes {
        if lane.is_empty() {
            blocks.push(&lane[..]);
        } else {
            for block in lane.chunks(block_size) {
                blocks.push(block);
            }
        }
    }
    Ok((blocks, lengths))
}

fn parse_restart_lengths(image: &[u8]) -> Result<Vec<usize>, CodecError> {
    if image.len() < RESTART_HEADER_BYTES || &image[..4] != RESTART_MAGIC {
        return Err(CodecError::InvalidFormat("invalid RNBW restart image"));
    }
    if read_u16(image, 4)? != RESTART_VERSION || read_u16(image, 6)? != 0 {
        return Err(CodecError::InvalidFormat(
            "unsupported RNBW restart image version",
        ));
    }
    let count = read_u32(image, 8)? as usize;
    let table_end = RESTART_HEADER_BYTES
        .checked_add(count.checked_mul(8).ok_or(CodecError::LengthOverflow)?)
        .ok_or(CodecError::LengthOverflow)?;
    if table_end > image.len() || count == 0 {
        return Err(CodecError::InvalidFormat(
            "truncated RNBW restart lane table",
        ));
    }
    let mut lengths = Vec::with_capacity(count);
    for index in 0..count {
        let offset = RESTART_HEADER_BYTES + index * 8;
        let len = u64::from_le_bytes(image[offset..offset + 8].try_into().unwrap());
        lengths.push(usize::try_from(len).map_err(|_| CodecError::LengthOverflow)?);
    }
    Ok(lengths)
}

fn parse_restart_image(image: &[u8]) -> Result<(LaneLz, Vec<usize>, usize), CodecError> {
    let lengths = parse_restart_lengths(image)?;
    let block_size = read_u32(image, 12)? as usize;
    if block_size == 0 {
        return Err(CodecError::InvalidFormat("zero RNBW restart block size"));
    }
    let inner_offset = RESTART_HEADER_BYTES + lengths.len() * 8;
    let codec = LaneLz::from_bytes(&image[inner_offset..])?;
    Ok((codec, lengths, block_size))
}

fn validate_restart_blocks(
    blocks: &[Vec<u8>],
    lengths: &[usize],
    block_size: usize,
) -> Result<(), CodecError> {
    let expected = lengths.iter().try_fold(0usize, |sum, &len| {
        sum.checked_add(len.div_ceil(block_size).max(1))
            .ok_or(CodecError::LengthOverflow)
    })?;
    if blocks.len() != expected {
        return Err(CodecError::InvalidFormat(
            "RNBW restart block count mismatch",
        ));
    }
    let mut index = 0;
    for &len in lengths {
        let count = len.div_ceil(block_size).max(1);
        let mut remaining = len;
        for block in &blocks[index..index + count] {
            let expected_len = remaining.min(block_size);
            if block.len() != expected_len {
                return Err(CodecError::InvalidFormat(
                    "RNBW restart block length mismatch",
                ));
            }
            remaining -= expected_len;
        }
        index += count;
    }
    Ok(())
}

fn validate_restart_descriptors(
    codec: &LaneLz,
    lengths: &[usize],
    block_size: usize,
) -> Result<(), CodecError> {
    let descriptors = parse_image(&codec.image)?;
    let expected_count = lengths.iter().try_fold(0usize, |sum, &len| {
        sum.checked_add(len.div_ceil(block_size).max(1))
            .ok_or(CodecError::LengthOverflow)
    })?;
    if descriptors.len() != expected_count {
        return Err(CodecError::InvalidFormat(
            "RNBW restart block count mismatch",
        ));
    }
    let mut index = 0;
    for &len in lengths {
        let count = len.div_ceil(block_size).max(1);
        let mut remaining = len;
        for descriptor in &descriptors[index..index + count] {
            let expected_len = remaining.min(block_size);
            if descriptor.raw_len != expected_len {
                return Err(CodecError::InvalidFormat(
                    "RNBW restart block length mismatch",
                ));
            }
            remaining -= expected_len;
        }
        index += count;
    }
    let expected_total = lengths.iter().try_fold(0usize, |sum, &len| {
        sum.checked_add(len).ok_or(CodecError::LengthOverflow)
    })?;
    if codec.raw_bytes() != expected_total {
        return Err(CodecError::InvalidFormat(
            "RNBW restart raw byte count mismatch",
        ));
    }
    Ok(())
}

fn reassemble_restart_blocks(
    blocks: Vec<Vec<u8>>,
    lengths: &[usize],
    block_size: usize,
) -> Result<Vec<Vec<u8>>, CodecError> {
    validate_restart_blocks(&blocks, lengths, block_size)?;
    let mut blocks = blocks.into_iter();
    lengths
        .iter()
        .map(|&len| {
            let count = len.div_ceil(block_size).max(1);
            let mut lane = Vec::with_capacity(len);
            for _ in 0..count {
                lane.extend_from_slice(
                    &blocks
                        .next()
                        .ok_or(CodecError::InvalidFormat("missing RNBW restart block"))?,
                );
            }
            Ok(lane)
        })
        .collect()
}

/// Persistent Vulkan device and pipelines for RNBW's per-lane LZ stage.
///
/// Construct once and reuse it; device setup and pipeline compilation are
/// intentionally outside the per-buffer encode/decode operations.
pub struct LaneLzGpu {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    encode_pipeline: wgpu::ComputePipeline,
    decode_pipeline: wgpu::ComputePipeline,
    adapter_info: wgpu::AdapterInfo,
    max_storage_bytes: u64,
}

impl LaneLzGpu {
    /// Requests a Vulkan adapter, preferring a discrete GPU over integrated or
    /// software adapters. This is async so callers can initialize it in their
    /// existing runtime without a nested executor.
    pub async fn new_vulkan() -> Result<Self, CodecError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN).await;
        adapters.sort_by_key(|adapter| match adapter.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        });
        let adapter = adapters
            .into_iter()
            .filter(|adapter| adapter.limits().max_storage_buffers_per_shader_stage >= 7)
            .next()
            .ok_or_else(|| {
                CodecError::Gpu(
                    "no Vulkan adapter supports the seven storage buffers required by the encode kernel".into(),
                )
            })?;
        let adapter_info = adapter.get_info();
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_storage_buffers_per_shader_stage = 7;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("RNBW lane LZ"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|error| CodecError::Gpu(format!("request Vulkan device: {error}")))?;
        let encode_pipeline = make_pipeline(&device, "RNBW lane LZ encode", ENCODE_SHADER);
        let decode_pipeline = make_pipeline(&device, "RNBW lane LZ decode", DECODE_SHADER);
        let max_storage_bytes = u64::from(device.limits().max_storage_buffer_binding_size);
        Ok(Self {
            _instance: instance,
            device,
            queue,
            encode_pipeline,
            decode_pipeline,
            adapter_info,
            max_storage_bytes,
        })
    }

    /// Adapter selected at initialization. Check this to distinguish the
    /// discrete GPU from an integrated or software Vulkan adapter.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Encodes semantic lanes with GPU LZ search and the shared CPU entropy
    /// back end. The result is the same RNLZ v2 container accepted by the CPU
    /// decoder. The per-lane 64 KiB hash and previous-link tables are scratch.
    pub fn encode(&self, lanes: &[&[u8]]) -> Result<LaneLz, CodecError> {
        let raw_total = validate_lanes(lanes)?;
        let gpu_lz = self.encode_lz_lanes(lanes)?;
        let stored = lanes
            .iter()
            .zip(gpu_lz)
            .map(|(input, (stream, token_count, literal_len))| {
                encode_lane_from_lz(input, &stream, token_count, literal_len)
            })
            .collect::<Result<Vec<_>, _>>()?;
        LaneLz::from_lanes(stored, raw_total)
    }

    /// Encodes independent restart blocks on the GPU. Smaller blocks expose
    /// more parallel jobs; `block_size` is persisted in the RNRB envelope.
    pub fn encode_restart_blocks(
        &self,
        lanes: &[&[u8]],
        block_size: usize,
    ) -> Result<LaneLzRestarted, CodecError> {
        let (blocks, lengths) = split_restart_blocks(lanes, block_size)?;
        let encoded = self.encode(&blocks)?;
        LaneLzRestarted::from_codec(&encoded, &lengths, block_size)
    }

    /// Decodes and rejoins the independent blocks in an RNRB envelope.
    pub fn decode_restart_blocks(&self, image: &[u8]) -> Result<Vec<Vec<u8>>, CodecError> {
        let (codec, lengths, block_size) = parse_restart_image(image)?;
        let blocks = self.decode_lanes(&codec)?;
        reassemble_restart_blocks(blocks, &lengths, block_size)
    }

    /// Encodes a fixed-width interleaved stream using one GPU LZ window for
    /// each structure-derived field lane.
    pub fn encode_interleaved(
        &self,
        input: &[u8],
        lane_width: usize,
    ) -> Result<LaneLz, CodecError> {
        if lane_width == 1 {
            return self.encode(&[input]);
        }
        let mut lanes = Vec::new();
        let raw_total = split_interleaved(input, lane_width, &mut lanes)?;
        let lane_refs: Vec<&[u8]> = lanes.iter().take(lane_width).map(Vec::as_slice).collect();
        let encoded = self.encode(&lane_refs)?;
        debug_assert_eq!(encoded.raw_bytes(), raw_total);
        Ok(encoded)
    }

    /// Decodes lane streams with GPU LZ expansion and the shared CPU entropy
    /// decoder. Lanes identified as incompressible use the raw fallback.
    pub fn decode_lanes(&self, codec: &LaneLz) -> Result<Vec<Vec<u8>>, CodecError> {
        let version = read_u16(&codec.image, 4)?;
        if version == LEGACY_VERSION {
            return codec.decode_lanes();
        }
        let descriptors = parse_image(&codec.image)?;
        let mut output: Vec<Option<Vec<u8>>> = vec![None; descriptors.len()];
        let mut gpu_lanes = Vec::new();
        for (index, descriptor) in descriptors.iter().copied().enumerate() {
            let payload = &codec.image
                [descriptor.payload_offset..descriptor.payload_offset + descriptor.payload_len];
            let model = if descriptor.model_len == 0 {
                &[][..]
            } else {
                &codec.image
                    [descriptor.model_offset..descriptor.model_offset + descriptor.model_len]
            };
            let decoded = rans::entropy_decode(
                descriptor.entropy_method,
                descriptor.token_len,
                model,
                payload,
            )?;
            if descriptor.mode == MODE_LZ_ENTROPY {
                gpu_lanes.push((index, descriptor, decoded));
            } else {
                if decoded.len() != descriptor.raw_len {
                    return Err(CodecError::InvalidFormat("RNBW lane length mismatch"));
                }
                output[index] = Some(decoded);
            }
        }
        if !gpu_lanes.is_empty() {
            let decoded = self.decode_lz_lanes(&gpu_lanes)?;
            for ((index, _, _), lane) in gpu_lanes.into_iter().zip(decoded) {
                output[index] = Some(lane);
            }
        }
        output
            .into_iter()
            .map(|lane| lane.ok_or(CodecError::InvalidFormat("missing decoded RNBW lane")))
            .collect()
    }

    /// Decodes lanes and restores their fixed-width interleaved structure.
    pub fn decode_interleaved(&self, codec: &LaneLz) -> Result<Vec<u8>, CodecError> {
        let lanes = self.decode_lanes(codec)?;
        if lanes.is_empty() || lanes.iter().any(|lane| lane.len() != lanes[0].len()) {
            return Err(CodecError::InvalidFormat("unequal RNBW interleaved lanes"));
        }
        let output_len = lanes[0]
            .len()
            .checked_mul(lanes.len())
            .ok_or(CodecError::LengthOverflow)?;
        let mut output = Vec::with_capacity(output_len);
        for record_index in 0..lanes[0].len() {
            for lane in &lanes {
                output.push(lane[record_index]);
            }
        }
        Ok(output)
    }

    fn encode_lz_lanes(&self, lanes: &[&[u8]]) -> Result<Vec<(Vec<u8>, usize, usize)>, CodecError> {
        let mut input = Vec::new();
        let mut descriptors = Vec::with_capacity(lanes.len());
        let mut token_words = 0usize;
        let mut literal_bytes = 0usize;
        for (index, lane) in lanes.iter().enumerate() {
            align_vec(&mut input, 4);
            align_size(&mut literal_bytes, 4)?;
            let input_offset = u32_len(input.len())?;
            let literal_offset = u32_len(literal_bytes)?;
            let token_offset = u32_len(token_words)?;
            let scratch_offset = index
                .checked_mul(HASH_SIZE)
                .ok_or(CodecError::LengthOverflow)?;
            descriptors.extend_from_slice(&[
                input_offset,
                u32_len(lane.len())?,
                token_offset,
                literal_offset,
                u32_len(scratch_offset)?,
                u32_len(scratch_offset)?,
                u32_len(index)?,
                0,
            ]);
            input.extend_from_slice(lane);
            token_words = token_words
                .checked_add(lane.len())
                .ok_or(CodecError::LengthOverflow)?;
            literal_bytes = literal_bytes
                .checked_add(lane.len())
                .ok_or(CodecError::LengthOverflow)?;
        }
        align_vec(&mut input, 4);
        align_size(&mut literal_bytes, 4)?;
        let input_words = bytes_as_u32(&input);
        let lane_bytes = u32s_as_bytes(&descriptors);
        let token_size = bytes_for_words(token_words)?;
        let literal_size = bytes_for_words(literal_bytes / 4)?;
        let scratch_size = bytes_for_words(
            lanes
                .len()
                .checked_mul(HASH_SIZE)
                .ok_or(CodecError::LengthOverflow)?,
        )?;
        let result_size = bytes_for_words(
            lanes
                .len()
                .checked_mul(2)
                .ok_or(CodecError::LengthOverflow)?,
        )?;
        self.check_storage_sizes(&[
            bytes_for_words(input_words.len())?,
            lane_bytes.len() as u64,
            token_size,
            literal_size,
            scratch_size,
            scratch_size,
            result_size,
        ])?;

        let input_buffer = self.upload("RNBW lane input", &u32s_as_bytes(&input_words), true);
        let lane_buffer = self.upload("RNBW encode lane descriptors", &lane_bytes, true);
        let token_buffer = self.zero_buffer("RNBW fixed tokens", token_size);
        let literal_buffer = self.zero_buffer("RNBW literal bytes", literal_size);
        let heads_buffer = self.storage_buffer("RNBW hash heads", scratch_size, false);
        let previous_buffer = self.storage_buffer("RNBW hash links", scratch_size, false);
        let results_buffer = self.storage_buffer("RNBW lane results", result_size, true);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RNBW encode bindings"),
            layout: &self.encode_pipeline.get_bind_group_layout(0),
            entries: &[
                binding(0, &input_buffer),
                binding(1, &lane_buffer),
                binding(2, &token_buffer),
                binding(3, &literal_buffer),
                binding(4, &heads_buffer),
                binding(5, &previous_buffer),
                binding(6, &results_buffer),
            ],
        });
        self.dispatch(&self.encode_pipeline, &bind_group, lanes.len())?;
        let readbacks = self.readback(&[
            (&token_buffer, token_size),
            (&literal_buffer, literal_size),
            (&results_buffer, result_size),
        ])?;
        let tokens = &readbacks[0];
        let literals = &readbacks[1];
        let results = &readbacks[2];
        let mut output = Vec::with_capacity(lanes.len());
        let mut token_offset = 0usize;
        let mut literal_offset = 0usize;
        for (index, lane) in lanes.iter().enumerate() {
            align_size(&mut literal_offset, 4)?;
            let token_count = read_u32(results, index * 8)? as usize;
            let literal_len = read_u32(results, index * 8 + 4)? as usize;
            if token_count > lane.len() || literal_len > lane.len() {
                return Err(CodecError::Gpu(
                    "LZ encode kernel returned invalid lengths".into(),
                ));
            }
            let token_end = token_offset
                .checked_add(
                    token_count
                        .checked_mul(4)
                        .ok_or(CodecError::LengthOverflow)?,
                )
                .ok_or(CodecError::LengthOverflow)?;
            let literal_end = literal_offset
                .checked_add(literal_len)
                .ok_or(CodecError::LengthOverflow)?;
            if token_end > tokens.len() || literal_end > literals.len() {
                return Err(CodecError::Gpu(
                    "LZ encode kernel output exceeds its buffers".into(),
                ));
            }
            let mut stream = Vec::with_capacity(token_end - token_offset + literal_len);
            stream.extend_from_slice(&tokens[token_offset..token_end]);
            stream.extend_from_slice(&literals[literal_offset..literal_end]);
            output.push((stream, token_count, literal_len));
            token_offset = token_offset
                .checked_add(
                    lane.len()
                        .checked_mul(4)
                        .ok_or(CodecError::LengthOverflow)?,
                )
                .ok_or(CodecError::LengthOverflow)?;
            literal_offset = literal_offset
                .checked_add(lane.len())
                .ok_or(CodecError::LengthOverflow)?;
        }
        Ok(output)
    }

    fn decode_lz_lanes(
        &self,
        lanes: &[(usize, LaneDescriptor, Vec<u8>)],
    ) -> Result<Vec<Vec<u8>>, CodecError> {
        let mut stream = Vec::new();
        let mut output_size = 0usize;
        let mut descriptors = Vec::with_capacity(lanes.len());
        let mut output_offsets = Vec::with_capacity(lanes.len());
        for (index, descriptor, decoded) in lanes {
            let token_bytes = descriptor
                .token_count
                .checked_mul(4)
                .ok_or(CodecError::LengthOverflow)?;
            if token_bytes.checked_add(descriptor.literal_len) != Some(decoded.len()) {
                return Err(CodecError::InvalidFormat(
                    "invalid fixed-width RNBW token stream",
                ));
            }
            align_vec(&mut stream, 4);
            align_size(&mut output_size, 4)?;
            let stream_offset = u32_len(stream.len())?;
            let output_offset = u32_len(output_size)?;
            descriptors.extend_from_slice(&[
                stream_offset,
                u32_len(descriptor.raw_len)?,
                u32_len(descriptor.token_count)?,
                u32_len(descriptor.literal_len)?,
                stream_offset
                    .checked_add(u32_len(token_bytes)?)
                    .ok_or(CodecError::LengthOverflow)?,
                output_offset,
                u32_len(*index)?,
                0,
            ]);
            stream.extend_from_slice(decoded);
            output_offsets.push((output_size, descriptor.raw_len));
            output_size = output_size
                .checked_add(descriptor.raw_len)
                .ok_or(CodecError::LengthOverflow)?;
        }
        align_vec(&mut stream, 4);
        align_size(&mut output_size, 4)?;
        let lane_bytes = u32s_as_bytes(&descriptors);
        let stream_words = bytes_as_u32(&stream);
        let stream_size = bytes_for_words(stream_words.len())?;
        let output_words = bytes_for_words(output_size / 4)?;
        let result_size = bytes_for_words(
            lanes
                .len()
                .checked_mul(2)
                .ok_or(CodecError::LengthOverflow)?,
        )?;
        self.check_storage_sizes(&[
            stream_size,
            lane_bytes.len() as u64,
            output_words,
            result_size,
        ])?;
        let stream_buffer = self.upload(
            "RNBW fixed token input",
            &u32s_as_bytes(&stream_words),
            true,
        );
        let lane_buffer = self.upload("RNBW decode lane descriptors", &lane_bytes, true);
        let output_buffer = self.zero_buffer("RNBW decoded lane output", output_words);
        let results_buffer = self.storage_buffer("RNBW decode results", result_size, true);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RNBW decode bindings"),
            layout: &self.decode_pipeline.get_bind_group_layout(0),
            entries: &[
                binding(0, &stream_buffer),
                binding(1, &lane_buffer),
                binding(2, &output_buffer),
                binding(3, &results_buffer),
            ],
        });
        self.dispatch(&self.decode_pipeline, &bind_group, lanes.len())?;
        let readbacks = self.readback(&[
            (&output_buffer, output_words),
            (&results_buffer, result_size),
        ])?;
        let output = &readbacks[0];
        let results = &readbacks[1];
        let mut decoded_lanes = Vec::with_capacity(lanes.len());
        for (lane_index, (offset, len)) in output_offsets.into_iter().enumerate() {
            if read_u32(results, lane_index * 8)? != 1
                || read_u32(results, lane_index * 8 + 4)? as usize != len
            {
                return Err(CodecError::InvalidFormat(
                    "GPU rejected an invalid RNBW LZ lane",
                ));
            }
            let end = offset.checked_add(len).ok_or(CodecError::LengthOverflow)?;
            decoded_lanes.push(
                output
                    .get(offset..end)
                    .ok_or(CodecError::Gpu(
                        "decoded lane is outside the output buffer".into(),
                    ))?
                    .to_vec(),
            );
        }
        Ok(decoded_lanes)
    }

    fn check_storage_sizes(&self, sizes: &[u64]) -> Result<(), CodecError> {
        if sizes.iter().any(|&size| size > self.max_storage_bytes) {
            return Err(CodecError::Gpu(format!(
                "required storage buffer exceeds adapter limit of {} bytes",
                self.max_storage_bytes
            )));
        }
        Ok(())
    }

    fn storage_buffer(&self, label: &str, size: u64, copy_src: bool) -> wgpu::Buffer {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if copy_src {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage,
            mapped_at_creation: false,
        })
    }

    fn zero_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        let buffer = self.storage_buffer(label, size, true);
        let zeros = vec![0u8; size.max(4) as usize];
        self.queue.write_buffer(&buffer, 0, &zeros);
        buffer
    }

    fn upload(&self, label: &str, bytes: &[u8], copy_src: bool) -> wgpu::Buffer {
        let size = aligned_buffer_size(bytes.len());
        let buffer = self.storage_buffer(label, size, copy_src);
        if !bytes.is_empty() {
            self.queue.write_buffer(&buffer, 0, bytes);
        }
        buffer
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: usize,
    ) -> Result<(), CodecError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RNBW lane LZ dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RNBW lane LZ pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(
                u32::try_from(workgroups).map_err(|_| CodecError::LengthOverflow)?,
                1,
                1,
            );
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    fn readback(&self, sources: &[(&wgpu::Buffer, u64)]) -> Result<Vec<Vec<u8>>, CodecError> {
        let staging: Vec<wgpu::Buffer> = sources
            .iter()
            .enumerate()
            .map(|(index, (_, size))| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if index == 0 {
                        "RNBW readback"
                    } else {
                        "RNBW readback extra"
                    }),
                    size: (*size).max(4),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RNBW copy GPU output"),
            });
        for ((source, size), target) in sources.iter().zip(&staging) {
            encoder.copy_buffer_to_buffer(source, 0, target, 0, (*size).max(4));
        }
        let submission = self.queue.submit([encoder.finish()]);
        let receivers: Vec<_> = staging
            .iter()
            .map(|buffer| {
                let (sender, receiver) = mpsc::channel();
                buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = sender.send(result);
                    });
                receiver
            })
            .collect();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| CodecError::Gpu(format!("wait for Vulkan work: {error}")))?;
        let mut output = Vec::with_capacity(staging.len());
        for (buffer, receiver) in staging.iter().zip(receivers) {
            receiver
                .recv()
                .map_err(|error| CodecError::Gpu(format!("Vulkan map callback: {error}")))?
                .map_err(|error| CodecError::Gpu(format!("map Vulkan output: {error}")))?;
            let view = buffer
                .slice(..)
                .get_mapped_range()
                .map_err(|error| CodecError::Gpu(format!("read Vulkan output: {error}")))?;
            output.push(view.to_vec());
            drop(view);
            buffer.unmap();
        }
        Ok(output)
    }
}

fn make_pipeline(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_envelope_roundtrips_empty_and_multiple_lanes() {
        let a = b"abcdabcdabcdabcdabcd";
        let b = b"a different lane with its own history";
        let empty = b"";
        let lanes: [&[u8]; 3] = [a, empty, b];
        let expected = vec![a.to_vec(), empty.to_vec(), b.to_vec()];
        for block_size in [1, 5, 64] {
            let image = LaneLzRestarted::encode_cpu(&lanes, block_size).unwrap();
            assert_eq!(image.raw_bytes().unwrap(), a.len() + b.len());
            assert_eq!(image.decode_lanes().unwrap(), expected);
            let parsed = LaneLzRestarted::from_bytes(image.as_bytes()).unwrap();
            assert_eq!(parsed.decode_lanes().unwrap(), expected);
        }
    }

    #[test]
    fn restart_envelope_rejects_bad_metadata() {
        let source = b"restart frame validation";
        let image = LaneLzRestarted::encode_cpu(&[source], 8).unwrap();
        let mut corrupt = image.as_bytes().to_vec();
        corrupt[12..16].copy_from_slice(&0u32.to_le_bytes());
        assert!(LaneLzRestarted::from_bytes(&corrupt).is_err());
        assert!(LaneLzRestarted::encode_cpu(&[source], 0).is_err());
    }
}

fn binding(index: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
}

fn align_vec(bytes: &mut Vec<u8>, alignment: usize) {
    bytes.resize(bytes.len().div_ceil(alignment) * alignment, 0);
}

fn align_size(value: &mut usize, alignment: usize) -> Result<(), CodecError> {
    *value = align_up(*value, alignment).ok_or(CodecError::LengthOverflow)?;
    Ok(())
}

fn u32_len(value: usize) -> Result<u32, CodecError> {
    u32::try_from(value).map_err(|_| CodecError::LengthOverflow)
}

fn bytes_for_words(words: usize) -> Result<u64, CodecError> {
    u64::try_from(words)
        .ok()
        .and_then(|words| words.checked_mul(4))
        .ok_or(CodecError::LengthOverflow)
}

fn aligned_buffer_size(bytes: usize) -> u64 {
    bytes.max(4).div_ceil(4) as u64 * 4
}

fn bytes_as_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn u32s_as_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}
