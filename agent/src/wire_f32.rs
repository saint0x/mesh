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
