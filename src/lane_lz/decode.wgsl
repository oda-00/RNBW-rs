struct Lane {
    stream_offset: u32,
    raw_len: u32,
    token_count: u32,
    literal_len: u32,
    literal_offset: u32,
    output_offset: u32,
    result_index: u32,
    reserved: u32,
}

@group(0) @binding(0) var<storage, read> stream_words: array<u32>;
@group(0) @binding(1) var<storage, read> lanes: array<Lane>;
@group(0) @binding(2) var<storage, read_write> output_words: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<vec2<u32>>;

fn get_byte(byte_offset: u32) -> u32 {
    return (stream_words[byte_offset >> 2u] >> ((byte_offset & 3u) * 8u)) & 255u;
}

fn get_output_byte(byte_offset: u32) -> u32 {
    return (output_words[byte_offset >> 2u] >> ((byte_offset & 3u) * 8u)) & 255u;
}

fn set_output_byte(byte_offset: u32, value: u32) {
    let shift = (byte_offset & 3u) * 8u;
    let word_index = byte_offset >> 2u;
    output_words[word_index] = output_words[word_index] | ((value & 255u) << shift);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>) {
    let lane = lanes[group.x];
    var output_index = 0u;
    var literal_index = lane.literal_offset;
    let literal_end = lane.literal_offset + lane.literal_len;

    for (var token_index = 0u; token_index < lane.token_count; token_index = token_index + 1u) {
        let token = stream_words[(lane.stream_offset >> 2u) + token_index];
        let field = (token >> 24u) & 127u;
        if (token & 0x80000000u) == 0u {
            let length = field + 1u;
            if (token & 0x00ffffffu) != 0u
                || literal_index + length > literal_end
                || output_index + length > lane.raw_len {
                results[lane.result_index] = vec2<u32>(0u, output_index);
                return;
            }
            for (var i = 0u; i < length; i = i + 1u) {
                set_output_byte(lane.output_offset + output_index, get_byte(literal_index));
                literal_index = literal_index + 1u;
                output_index = output_index + 1u;
            }
        } else {
            let length = field + 3u;
            let distance = ((token >> 8u) & 65535u) + 1u;
            if ((token & 255u) != 0u
                || distance > output_index
                || output_index + length > lane.raw_len) {
                results[lane.result_index] = vec2<u32>(0u, output_index);
                return;
            }
            for (var i = 0u; i < length; i = i + 1u) {
                let value = get_output_byte(lane.output_offset + output_index - distance);
                set_output_byte(lane.output_offset + output_index, value);
                output_index = output_index + 1u;
            }
        }
    }
    results[lane.result_index] = vec2<u32>(
        select(0u, 1u, literal_index == literal_end && output_index == lane.raw_len),
        output_index
    );
}
