use deckcraft_hevc::Decoder;

fn split_access_units(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i..i + 3] == [0, 0, 1] {
            let start = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            starts.push((start, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }

    let mut units = Vec::new();
    let mut unit_start = 0;
    let mut seen_vcl = false;
    for &(start, payload) in &starts {
        if payload + 2 >= data.len() {
            continue;
        }
        let nal_type = (data[payload] >> 1) & 0x3f;
        let vcl = nal_type < 32;
        let first_slice = vcl && data[payload + 2] & 0x80 != 0;
        if seen_vcl && (matches!(nal_type, 32..=35 | 39) || first_slice) {
            units.push(&data[unit_start..start]);
            unit_start = start;
            seen_vcl = false;
        }
        seen_vcl |= vcl;
    }
    if unit_start < data.len() {
        units.push(&data[unit_start..]);
    }
    units
}

fn damaged_fixture() -> Vec<u8> {
    let mut data = include_bytes!("data/determinism-bframes.hevc").to_vec();
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if i + 4 <= data.len() && data[i..i + 4] == [0, 0, 0, 1] {
            starts.push(i + 4);
            i += 4;
        } else if data[i..i + 3] == [0, 0, 1] {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut vcl = 0;
    for (n, &start) in starts.iter().enumerate() {
        if start + 2 >= data.len() || (data[start] >> 1) & 0x3f > 31 {
            continue;
        }
        if vcl == 8 {
            let end = starts.get(n + 1).copied().unwrap_or(data.len());
            let at = start + 3 + (end.saturating_sub(start + 4)) / 2;
            data[at] ^= 0x20;
            return data;
        }
        vcl += 1;
    }
    panic!("no suitable VCL NAL in deterministic HEVC fixture");
}

fn decode_hash(data: &[u8], threads: usize) -> (u64, Option<String>) {
    let mut decoder = Decoder::with_threads(threads);
    let mut hash = 0xcbf29ce484222325u64;
    let mut add = |bytes: &[u8]| {
        for &byte in bytes {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    };
    for (i, access_unit) in split_access_units(data).into_iter().enumerate() {
        match decoder.decode(access_unit, i as i64) {
            Ok(pictures) => {
                for picture in pictures {
                    add(&picture.width.to_le_bytes());
                    add(&picture.height.to_le_bytes());
                    add(&picture.pts.to_le_bytes());
                    add(&picture.y.to_le_bytes());
                    add(&picture.u.to_le_bytes());
                    add(&picture.v.to_le_bytes());
                }
            }
            Err(error) => add(error.to_string().as_bytes()),
        }
    }
    for picture in decoder.flush() {
        add(&picture.width.to_le_bytes());
        add(&picture.height.to_le_bytes());
        add(&picture.pts.to_le_bytes());
        add(&picture.y.to_le_bytes());
        add(&picture.u.to_le_bytes());
        add(&picture.v.to_le_bytes());
    }
    let worker_error = decoder.take_error().map(|error| error.to_string());
    drop(add);
    (hash, worker_error)
}

#[test]
fn damaged_stream_error_and_output_are_stable_across_thread_counts() {
    let data = damaged_fixture();
    assert!(decode_hash(&data, 1).1.is_some(), "mutation must trigger a worker decode error");
    for threads in [1, 2, 4] {
        let expected = decode_hash(&data, threads);
        for _ in 0..3 {
            assert_eq!(decode_hash(&data, threads), expected, "threads={threads}");
        }
    }
    let expected = decode_hash(&data, 1);
    for threads in [2, 4] {
        assert_eq!(decode_hash(&data, threads), expected, "thread count {threads}");
    }
}
