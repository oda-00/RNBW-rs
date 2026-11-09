struct Lane {
    input_offset: u32,
    input_len: u32,
    token_offset: u32,
    literal_offset: u32,
    heads_offset: u32,
    previous_offset: u32,
    result_index: u32,
    reserved: u32,
}

@group(0) @binding(0) var<storage, read> input_words: array<u32>;
@group(0) @binding(1) var<storage, read> lanes: array<Lane>;
@group(0) @binding(2) var<storage, read_write> token_words: array<u32>;
@group(0) @binding(3) var<storage, read_write> literal_words: array<u32>;
@group(0) @binding(4) var<storage, read_write> heads: array<u32>;
@group(0) @binding(5) var<storage, read_write> previous: array<u32>;
@group(0) @binding(6) var<storage, read_write> results: array<vec2<u32>>;

const NONE: u32 = 0xffffffffu;
const WINDOW: u32 = 65536u;
const WINDOW_MASK: u32 = 65535u;
const HASH_SIZE: u32 = 65536u;
const MAX_CHAIN: u32 = 32u;
const MAX_MATCH: u32 = 130u;

fn get_byte(input_offset: u32, index: u32) -> u32 {
    let byte_index = input_offset + index;
    return (input_words[byte_index >> 2u] >> ((byte_index & 3u) * 8u)) & 255u;
}

fn hash_at(lane: Lane, index: u32) -> u32 {
    let value = get_byte(lane.input_offset, index)
        | (get_byte(lane.input_offset, index + 1u) << 8u)
        | (get_byte(lane.input_offset, index + 2u) << 16u);
    return (value * 0x1e35a7bdu) >> 16u;
}

fn insert_position(lane: Lane, index: u32) {
    if index + 2u >= lane.input_len {
        return;
    }
    let hash = hash_at(lane, index);
    let slot = index & WINDOW_MASK;
    previous[lane.previous_offset + slot] = heads[lane.heads_offset + hash];
    heads[lane.heads_offset + hash] = index;
}

fn write_literal(lane: Lane, byte_index: u32, value: u32) {
    let absolute = lane.literal_offset + byte_index;
    let shift = (absolute & 3u) * 8u;
    let word_index = absolute >> 2u;
    literal_words[word_index] = literal_words[word_index] | ((value & 255u) << shift);
}

fn emit_literals(lane: Lane, start: u32, length: u32, token_count: ptr<function, u32>, literal_count: ptr<function, u32>) {
    var offset = 0u;
    while offset < length {
        let count = min(128u, length - offset);
        token_words[lane.token_offset + *token_count] = (count - 1u) << 24u;
        *token_count = *token_count + 1u;
        for (var i = 0u; i < count; i = i + 1u) {
            write_literal(lane, *literal_count, get_byte(lane.input_offset, start + offset + i));
            *literal_count = *literal_count + 1u;
        }
        offset = offset + count;
    }
}

fn emit_matches(lane: Lane, length: u32, distance: u32, token_count: ptr<function, u32>) {
    var remaining = length;
    while remaining > 0u {
        var part = min(remaining, MAX_MATCH);
        if remaining > MAX_MATCH && remaining - part < 3u {
            part = remaining - 3u;
        }
        token_words[lane.token_offset + *token_count] = 0x80000000u
            | ((part - 3u) << 24u)
            | ((distance - 1u) << 8u);
        *token_count = *token_count + 1u;
        remaining = remaining - part;
    }
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>) {
    let lane = lanes[group.x];
    for (var i = 0u; i < HASH_SIZE; i = i + 1u) {
        heads[lane.heads_offset + i] = NONE;
        previous[lane.previous_offset + i] = NONE;
    }

    var token_count = 0u;
    var literal_count = 0u;
    var index = 0u;
    var literal_start = 0u;
    while index + 2u < lane.input_len {
        let hash = hash_at(lane, index);
        var candidate = heads[lane.heads_offset + hash];
        var best_len = 0u;
        var best_distance = 0u;
        var probes = 0u;
        while candidate != NONE && probes < MAX_CHAIN {
            let distance = index - candidate;
            if distance > WINDOW {
                break;
            }
            if get_byte(lane.input_offset, candidate) == get_byte(lane.input_offset, index)
                && get_byte(lane.input_offset, candidate + 1u) == get_byte(lane.input_offset, index + 1u)
                && get_byte(lane.input_offset, candidate + 2u) == get_byte(lane.input_offset, index + 2u) {
                var length = 3u;
                while index + length < lane.input_len
                    && get_byte(lane.input_offset, candidate + length) == get_byte(lane.input_offset, index + length) {
                    length = length + 1u;
                }
                if length > best_len {
                    best_len = length;
                    best_distance = distance;
                    if index + length == lane.input_len {
                        break;
                    }
                }
            }
            candidate = previous[lane.previous_offset + (candidate & WINDOW_MASK)];
            probes = probes + 1u;
        }

        if best_len >= 3u {
            emit_literals(lane, literal_start, index - literal_start, &token_count, &literal_count);
            emit_matches(lane, best_len, best_distance, &token_count);
            let end = index + best_len;
            while index < end {
                insert_position(lane, index);
                index = index + 1u;
            }
            literal_start = index;
        } else {
            insert_position(lane, index);
            index = index + 1u;
        }
    }
    emit_literals(lane, literal_start, lane.input_len - literal_start, &token_count, &literal_count);
    results[lane.result_index] = vec2<u32>(token_count, literal_count);
}
