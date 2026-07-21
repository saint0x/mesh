use crate::errors::{AgentError, Result};
use std::slice;

pub(crate) fn copy_into_f32_slice(dst: &mut [f32], src: &[u8]) {
    let expected_bytes = dst.len().saturating_mul(std::mem::size_of::<f32>());
    assert_eq!(
        src.len(),
        expected_bytes,
        "wire payload byte length {} did not match destination byte length {}",
        src.len(),
        expected_bytes
    );
    #[cfg(target_endian = "little")]
    unsafe {
        let dst_bytes = slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, expected_bytes);
        dst_bytes.copy_from_slice(src);
    }
    #[cfg(target_endian = "big")]
    for (slot, chunk) in dst
        .iter_mut()
        .zip(src.chunks_exact(std::mem::size_of::<f32>()))
    {
        *slot = f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
}

pub(crate) fn accumulate_into_f32_slice(dst: &mut [f32], src: &[u8]) {
    let expected_bytes = dst.len().saturating_mul(std::mem::size_of::<f32>());
    assert_eq!(
        src.len(),
        expected_bytes,
        "wire payload byte length {} did not match destination byte length {}",
        src.len(),
        expected_bytes
    );
    #[cfg(target_endian = "little")]
    unsafe {
        let (prefix, values, suffix) = src.align_to::<f32>();
        if prefix.is_empty() && suffix.is_empty() {
            for (slot, value) in dst.iter_mut().zip(values.iter()) {
                *slot += *value;
            }
            return;
        }
    }
    for (slot, chunk) in dst
        .iter_mut()
        .zip(src.chunks_exact(std::mem::size_of::<f32>()))
    {
        *slot += f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
}

pub(crate) fn decode_into_f32_scratch<'a>(
    expected_len: usize,
    payload_bytes: &[u8],
    scratch: &'a mut Vec<f32>,
) -> Result<&'a [f32]> {
    let expected_bytes = expected_len.saturating_mul(std::mem::size_of::<f32>());
    if payload_bytes.len() != expected_bytes {
        return Err(AgentError::Execution(format!(
            "Wire payload byte length {} did not match expected byte length {}",
            payload_bytes.len(),
            expected_bytes
        )));
    }

    scratch.clear();
    scratch.resize(expected_len, 0.0);
    copy_into_f32_slice(scratch.as_mut_slice(), payload_bytes);
    Ok(scratch.as_slice())
}

pub(crate) fn decode_to_f32_vec(payload_bytes: &[u8]) -> Result<Vec<f32>> {
    let expected_len = payload_bytes
        .len()
        .checked_div(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            AgentError::Execution("wire payload length overflow while decoding f32 vec".to_string())
        })?;
    if payload_bytes.len() != expected_len.saturating_mul(std::mem::size_of::<f32>()) {
        return Err(AgentError::Execution(format!(
            "Wire payload byte length {} did not align to f32 element width",
            payload_bytes.len()
        )));
    }
    let mut out = vec![0.0f32; expected_len];
    copy_into_f32_slice(out.as_mut_slice(), payload_bytes);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        accumulate_into_f32_slice, copy_into_f32_slice, decode_into_f32_scratch, decode_to_f32_vec,
    };

    fn bytes_from_f32(values: &[f32]) -> &[u8] {
        let byte_len = std::mem::size_of_val(values);
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, byte_len) }
    }

    #[test]
    fn copy_into_f32_slice_round_trips_little_endian_bytes() {
        let src = [1.5f32, -2.25, 8.0, 0.125];
        let mut dst = [0.0f32; 4];
        copy_into_f32_slice(&mut dst, bytes_from_f32(&src));
        assert_eq!(dst, src);
    }

    #[test]
    fn accumulate_into_f32_slice_supports_aligned_payloads() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [10.0f32, 20.0, 30.0, 40.0];
        accumulate_into_f32_slice(&mut dst, bytes_from_f32(&src));
        assert_eq!(dst, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn accumulate_into_f32_slice_supports_unaligned_payloads() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let aligned_bytes = bytes_from_f32(&src);
        let mut misaligned = Vec::with_capacity(aligned_bytes.len() + 1);
        misaligned.push(0);
        misaligned.extend_from_slice(aligned_bytes);
        let payload = &misaligned[1..];

        let mut dst = [10.0f32, 20.0, 30.0, 40.0];
        accumulate_into_f32_slice(&mut dst, payload);
        assert_eq!(dst, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn decode_into_f32_scratch_reuses_buffer() {
        let src = [5.0f32, 6.0, 7.0];
        let mut scratch = vec![0.0f32; 8];
        let original_capacity = scratch.capacity();
        let decoded =
            decode_into_f32_scratch(src.len(), bytes_from_f32(&src), &mut scratch).unwrap();
        assert_eq!(decoded, src);
        assert_eq!(scratch.capacity(), original_capacity);
    }

    #[test]
    fn decode_to_f32_vec_round_trips_bytes() {
        let src = [9.0f32, -1.5, 2.25];
        let decoded = decode_to_f32_vec(bytes_from_f32(&src)).unwrap();
        assert_eq!(decoded, src);
    }
}
