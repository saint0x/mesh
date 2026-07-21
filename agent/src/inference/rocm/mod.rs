#![cfg(all(target_os = "linux", feature = "rocm"))]

use crate::errors::{AgentError, Result};
use crate::inference::tensor_ops::{Tensor1D, Tensor2D};
use std::ffi::CStr;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int};
use std::ptr::NonNull;

extern "C" {
    fn meshnet_rocm_last_error() -> *const c_char;
    fn meshnet_rocm_device_count(count: *mut c_int) -> c_int;
    fn meshnet_rocm_malloc(ptr: *mut *mut f32, len: usize) -> c_int;
    fn meshnet_rocm_malloc_u32(ptr: *mut *mut u32, len: usize) -> c_int;
    fn meshnet_rocm_free(ptr: *mut std::ffi::c_void) -> c_int;
    fn meshnet_rocm_upload_f32(dst: *mut f32, src: *const f32, len: usize) -> c_int;
    fn meshnet_rocm_download_f32(dst: *mut f32, src: *const f32, len: usize) -> c_int;
    fn meshnet_rocm_upload_u32(dst: *mut u32, src: *const u32, len: usize) -> c_int;
    fn meshnet_rocm_fill(out: *mut f32, value: f32, len: usize) -> c_int;
    fn meshnet_rocm_add(lhs: *const f32, rhs: *const f32, out: *mut f32, len: usize) -> c_int;
    fn meshnet_rocm_mul(lhs: *const f32, rhs: *const f32, out: *mut f32, len: usize) -> c_int;
    fn meshnet_rocm_silu(input: *const f32, out: *mut f32, len: usize) -> c_int;
    fn meshnet_rocm_embedding(
        table: *const f32,
        tokens: *const u32,
        out: *mut f32,
        token_count: usize,
        hidden_dim: usize,
    ) -> c_int;
    fn meshnet_rocm_rms_norm(
        input: *const f32,
        gamma: *const f32,
        out: *mut f32,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> c_int;
    fn meshnet_rocm_rope(
        input: *const f32,
        positions: *const u32,
        out: *mut f32,
        rows: usize,
        cols: usize,
        head_dim: usize,
        base: f32,
    ) -> c_int;
    fn meshnet_rocm_matmul(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> c_int;
    fn meshnet_rocm_copy_columns(
        input: *const f32,
        out: *mut f32,
        rows: usize,
        input_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> c_int;
    fn meshnet_rocm_copy_rows(
        input: *const f32,
        out: *mut f32,
        rows: usize,
        cols: usize,
        dst_row_start: usize,
    ) -> c_int;
    fn meshnet_rocm_copy_rows_range(
        input: *const f32,
        out: *mut f32,
        row_count: usize,
        cols: usize,
        src_row_start: usize,
        dst_row_start: usize,
    ) -> c_int;
    fn meshnet_rocm_add_range(
        dst: *mut f32,
        update: *const f32,
        offset: usize,
        len: usize,
    ) -> c_int;
    fn meshnet_rocm_attention(
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        local_kv_indices: *const u32,
        out: *mut f32,
        q_rows: usize,
        q_cols: usize,
        kv_cols: usize,
        cache_prefix_len: usize,
        cache_seq_len: usize,
        head_dim: usize,
    ) -> c_int;
}

pub(crate) fn probe() -> (bool, Option<String>) {
    match device_count() {
        Ok(count) if count > 0 => (true, None),
        Ok(_) => (
            false,
            Some("rocm runtime reported zero HIP devices".to_string()),
        ),
        Err(err) => (false, Some(err.to_string())),
    }
}

pub(crate) mod forward;

fn device_count() -> Result<i32> {
    let mut count = 0;
    call(|| unsafe { meshnet_rocm_device_count(&mut count) })?;
    Ok(count)
}

fn call(call: impl FnOnce() -> c_int) -> Result<()> {
    if call() == 0 {
        return Ok(());
    }
    Err(AgentError::Execution(format!(
        "ROCm runtime error: {}",
        last_error()
    )))
}

fn last_error() -> String {
    unsafe {
        let ptr = meshnet_rocm_last_error();
        if ptr.is_null() {
            return "unknown error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

#[derive(Debug)]
pub(crate) struct RocmTensor {
    ptr: NonNull<f32>,
    rows: usize,
    cols: usize,
}

unsafe impl Send for RocmTensor {}
unsafe impl Sync for RocmTensor {}

impl RocmTensor {
    pub(crate) fn zeros(rows: usize, cols: usize) -> Result<Self> {
        Self::filled(rows, cols, 0.0)
    }

    pub(crate) fn filled(rows: usize, cols: usize, value: f32) -> Result<Self> {
        let tensor = Self::uninitialized(rows, cols)?;
        call(|| unsafe { meshnet_rocm_fill(tensor.ptr.as_ptr(), value, tensor.len()) })?;
        Ok(tensor)
    }

    pub(crate) fn from_2d(tensor: &Tensor2D) -> Result<Self> {
        let out = Self::uninitialized(tensor.rows, tensor.cols)?;
        call(|| unsafe {
            meshnet_rocm_upload_f32(out.ptr.as_ptr(), tensor.data.as_ptr(), tensor.len())
        })?;
        Ok(out)
    }

    pub(crate) fn from_1d(tensor: &Tensor1D) -> Result<Self> {
        let out = Self::uninitialized(1, tensor.len())?;
        call(|| unsafe {
            meshnet_rocm_upload_f32(out.ptr.as_ptr(), tensor.data.as_ptr(), tensor.len())
        })?;
        Ok(out)
    }

    pub(crate) fn uninitialized(rows: usize, cols: usize) -> Result<Self> {
        let len = rows.checked_mul(cols).ok_or_else(|| {
            AgentError::Execution(format!("ROCm tensor shape overflow: {rows}x{cols}"))
        })?;
        let mut ptr = std::ptr::null_mut();
        call(|| unsafe { meshnet_rocm_malloc(&mut ptr, len) })?;
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            AgentError::Execution("ROCm allocation returned a null pointer".to_string())
        })?;
        Ok(Self { ptr, rows, cols })
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        self.len().saturating_mul(std::mem::size_of::<f32>())
    }

    pub(crate) fn to_vec(&self) -> Result<Vec<f32>> {
        let mut data = vec![0.0; self.len()];
        call(|| unsafe {
            meshnet_rocm_download_f32(data.as_mut_ptr(), self.ptr.as_ptr(), data.len())
        })?;
        Ok(data)
    }

    pub(crate) fn download_range(&self, range: std::ops::Range<usize>) -> Result<Vec<f32>> {
        if range.start > range.end || range.end > self.len() {
            return Err(AgentError::Execution(format!(
                "ROCm download range {}..{} out of {} elements",
                range.start,
                range.end,
                self.len()
            )));
        }
        let mut data = vec![0.0; range.len()];
        call(|| unsafe {
            meshnet_rocm_download_f32(
                data.as_mut_ptr(),
                self.ptr.as_ptr().add(range.start),
                data.len(),
            )
        })?;
        Ok(data)
    }

    pub(crate) fn upload_range(&self, range_start: usize, values: &[f32]) -> Result<()> {
        if range_start.saturating_add(values.len()) > self.len() {
            return Err(AgentError::Execution(format!(
                "ROCm upload range {}..{} out of {} elements",
                range_start,
                range_start.saturating_add(values.len()),
                self.len()
            )));
        }
        call(|| unsafe {
            meshnet_rocm_upload_f32(
                self.ptr.as_ptr().add(range_start),
                values.as_ptr(),
                values.len(),
            )
        })
    }

    #[cfg(test)]
    pub(crate) fn to_2d(&self) -> Result<Tensor2D> {
        Tensor2D::new(self.to_vec()?, self.rows, self.cols)
    }

    pub(crate) fn to_2d_prefix_rows(&self, rows: usize) -> Result<Tensor2D> {
        if rows > self.rows {
            return Err(AgentError::Execution(format!(
                "ROCm prefix rows {} exceed tensor rows {}",
                rows, self.rows
            )));
        }
        Tensor2D::new(
            self.download_range(0..rows.saturating_mul(self.cols))?,
            rows,
            self.cols,
        )
    }

    pub(crate) fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.cols != rhs.rows {
            return Err(AgentError::Execution(format!(
                "ROCm matmul shape mismatch: {}x{} @ {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            )));
        }
        let out = Self::uninitialized(self.rows, rhs.cols)?;
        call(|| unsafe {
            meshnet_rocm_matmul(
                self.ptr.as_ptr(),
                rhs.ptr.as_ptr(),
                out.ptr.as_ptr(),
                self.rows,
                self.cols,
                rhs.cols,
            )
        })?;
        Ok(out)
    }

    pub(crate) fn add(&self, rhs: &Self) -> Result<Self> {
        self.same_shape(rhs, "add")?;
        let out = Self::uninitialized(self.rows, self.cols)?;
        call(|| unsafe {
            meshnet_rocm_add(
                self.ptr.as_ptr(),
                rhs.ptr.as_ptr(),
                out.ptr.as_ptr(),
                self.len(),
            )
        })?;
        Ok(out)
    }

    pub(crate) fn mul(&self, rhs: &Self) -> Result<Self> {
        self.same_shape(rhs, "mul")?;
        let out = Self::uninitialized(self.rows, self.cols)?;
        call(|| unsafe {
            meshnet_rocm_mul(
                self.ptr.as_ptr(),
                rhs.ptr.as_ptr(),
                out.ptr.as_ptr(),
                self.len(),
            )
        })?;
        Ok(out)
    }

    pub(crate) fn silu(&self) -> Result<Self> {
        let out = Self::uninitialized(self.rows, self.cols)?;
        call(|| unsafe { meshnet_rocm_silu(self.ptr.as_ptr(), out.ptr.as_ptr(), self.len()) })?;
        Ok(out)
    }

    pub(crate) fn rms_norm(&self, gamma: &Self, eps: f32) -> Result<Self> {
        if gamma.len() != self.cols {
            return Err(AgentError::Execution(format!(
                "ROCm RMSNorm dimension mismatch: tensor cols {} vs gamma {}",
                self.cols,
                gamma.len()
            )));
        }
        let out = Self::uninitialized(self.rows, self.cols)?;
        call(|| unsafe {
            meshnet_rocm_rms_norm(
                self.ptr.as_ptr(),
                gamma.ptr.as_ptr(),
                out.ptr.as_ptr(),
                self.rows,
                self.cols,
                eps,
            )
        })?;
        Ok(out)
    }

    pub(crate) fn rope(&self, positions: &[u32], head_dim: usize, base: f32) -> Result<Self> {
        if self.rows != positions.len() {
            return Err(AgentError::Execution(format!(
                "ROCm RoPE position count {} does not match rows {}",
                positions.len(),
                self.rows
            )));
        }
        if head_dim == 0 || head_dim % 2 != 0 || self.cols % head_dim != 0 {
            return Err(AgentError::Execution(format!(
                "ROCm RoPE invalid geometry cols={} head_dim={}",
                self.cols, head_dim
            )));
        }
        let positions = RocmU32Buffer::from_slice(positions)?;
        let out = Self::uninitialized(self.rows, self.cols)?;
        call(|| unsafe {
            meshnet_rocm_rope(
                self.ptr.as_ptr(),
                positions.ptr.as_ptr(),
                out.ptr.as_ptr(),
                self.rows,
                self.cols,
                head_dim,
                base,
            )
        })?;
        Ok(out)
    }

    pub(crate) fn embedding(table: &Self, tokens: &[u32]) -> Result<Self> {
        let token_buffer = RocmU32Buffer::from_slice(tokens)?;
        let out = Self::uninitialized(tokens.len(), table.cols)?;
        call(|| unsafe {
            meshnet_rocm_embedding(
                table.ptr.as_ptr(),
                token_buffer.ptr.as_ptr(),
                out.ptr.as_ptr(),
                tokens.len(),
                table.cols,
            )
        })?;
        Ok(out)
    }

    pub(crate) fn copy_columns(&self, col_start: usize, col_count: usize) -> Result<Self> {
        if col_count == 0 || col_start.saturating_add(col_count) > self.cols {
            return Err(AgentError::Execution(format!(
                "ROCm column slice {}..{} out of {} cols",
                col_start,
                col_start.saturating_add(col_count),
                self.cols
            )));
        }
        let out = Self::uninitialized(self.rows, col_count)?;
        call(|| unsafe {
            meshnet_rocm_copy_columns(
                self.ptr.as_ptr(),
                out.ptr.as_ptr(),
                self.rows,
                self.cols,
                col_start,
                col_count,
            )
        })?;
        Ok(out)
    }

    pub(crate) fn copy_rows_to(&self, dst: &Self, dst_row_start: usize) -> Result<()> {
        if self.cols != dst.cols || dst_row_start.saturating_add(self.rows) > dst.rows {
            return Err(AgentError::Execution(format!(
                "ROCm row copy shape mismatch: src {}x{}, dst {}x{}, dst_row_start {}",
                self.rows, self.cols, dst.rows, dst.cols, dst_row_start
            )));
        }
        call(|| unsafe {
            meshnet_rocm_copy_rows(
                self.ptr.as_ptr(),
                dst.ptr.as_ptr(),
                self.rows,
                self.cols,
                dst_row_start,
            )
        })
    }

    pub(crate) fn copy_rows_range_to(
        &self,
        dst: &Self,
        src_row_start: usize,
        row_count: usize,
        dst_row_start: usize,
    ) -> Result<()> {
        if self.cols != dst.cols
            || src_row_start.saturating_add(row_count) > self.rows
            || dst_row_start.saturating_add(row_count) > dst.rows
        {
            return Err(AgentError::Execution(format!(
                "ROCm row range copy shape mismatch: src {}x{} rows {}..{}, dst {}x{} rows {}..{}",
                self.rows,
                self.cols,
                src_row_start,
                src_row_start.saturating_add(row_count),
                dst.rows,
                dst.cols,
                dst_row_start,
                dst_row_start.saturating_add(row_count)
            )));
        }
        call(|| unsafe {
            meshnet_rocm_copy_rows_range(
                self.ptr.as_ptr(),
                dst.ptr.as_ptr(),
                row_count,
                self.cols,
                src_row_start,
                dst_row_start,
            )
        })
    }

    pub(crate) fn copy_rows(&self, row_start: usize, row_count: usize) -> Result<Self> {
        if row_start.saturating_add(row_count) > self.rows {
            return Err(AgentError::Execution(format!(
                "ROCm row slice {}..{} out of {} rows",
                row_start,
                row_start.saturating_add(row_count),
                self.rows
            )));
        }
        let out = Self::uninitialized(row_count, self.cols)?;
        self.copy_rows_range_to(&out, row_start, row_count, 0)?;
        Ok(out)
    }

    pub(crate) fn add_range_from(&self, range_start: usize, update: &Self) -> Result<()> {
        if update.rows != 1 {
            return Err(AgentError::Execution(format!(
                "ROCm range update expects a flat 1-row tensor, got {}x{}",
                update.rows, update.cols
            )));
        }
        if range_start.saturating_add(update.len()) > self.len() {
            return Err(AgentError::Execution(format!(
                "ROCm add range {}..{} out of {} elements",
                range_start,
                range_start.saturating_add(update.len()),
                self.len()
            )));
        }
        call(|| unsafe {
            meshnet_rocm_add_range(
                self.ptr.as_ptr(),
                update.ptr.as_ptr(),
                range_start,
                update.len(),
            )
        })
    }

    pub(crate) fn attention(
        q: &Self,
        k_cache: &Self,
        v_cache: &Self,
        local_kv_indices: &RocmU32Buffer,
        cache_prefix_len: usize,
        cache_seq_len: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if k_cache.cols != v_cache.cols {
            return Err(AgentError::Execution(
                "ROCm attention K/V cache width mismatch".to_string(),
            ));
        }
        let out = Self::uninitialized(q.rows, q.cols)?;
        call(|| unsafe {
            meshnet_rocm_attention(
                q.ptr.as_ptr(),
                k_cache.ptr.as_ptr(),
                v_cache.ptr.as_ptr(),
                local_kv_indices.ptr.as_ptr(),
                out.ptr.as_ptr(),
                q.rows,
                q.cols,
                k_cache.cols,
                cache_prefix_len,
                cache_seq_len,
                head_dim,
            )
        })?;
        Ok(out)
    }

    fn same_shape(&self, rhs: &Self, op: &str) -> Result<()> {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            return Ok(());
        }
        Err(AgentError::Execution(format!(
            "ROCm {op} shape mismatch: {}x{} vs {}x{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        )))
    }
}

impl Drop for RocmTensor {
    fn drop(&mut self) {
        let _ = unsafe { meshnet_rocm_free(self.ptr.as_ptr().cast()) };
    }
}

#[derive(Debug)]
pub(crate) struct RocmU32Buffer {
    ptr: NonNull<u32>,
    _marker: PhantomData<u32>,
}

unsafe impl Send for RocmU32Buffer {}
unsafe impl Sync for RocmU32Buffer {}

impl RocmU32Buffer {
    pub(crate) fn from_slice(values: &[u32]) -> Result<Self> {
        let out = Self::uninitialized(values.len())?;
        call(|| unsafe {
            meshnet_rocm_upload_u32(out.ptr.as_ptr(), values.as_ptr(), values.len())
        })?;
        Ok(out)
    }

    pub(crate) fn uninitialized(len: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        call(|| unsafe { meshnet_rocm_malloc_u32(&mut ptr, len) })?;
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            AgentError::Execution("ROCm u32 allocation returned a null pointer".to_string())
        })?;
        Ok(Self {
            ptr,
            _marker: PhantomData,
        })
    }
}

impl Drop for RocmU32Buffer {
    fn drop(&mut self) {
        let _ = unsafe { meshnet_rocm_free(self.ptr.as_ptr().cast()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_probe_reports_device() {
        let (available, reason) = probe();
        assert!(available, "ROCm probe failed: {:?}", reason);
    }

    #[test]
    fn rocm_hipblas_matmul_matches_reference() {
        let a = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let b = Tensor2D::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2).unwrap();
        let got = RocmTensor::from_2d(&a)
            .unwrap()
            .matmul(&RocmTensor::from_2d(&b).unwrap())
            .unwrap()
            .to_2d()
            .unwrap();

        assert_eq!(got.rows, 2);
        assert_eq!(got.cols, 2);
        assert_eq!(got.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn rocm_attention_matches_reference() {
        let q = Tensor2D::new(
            vec![
                1.0, 0.0, 0.5, 0.5, //
                0.0, 1.0, 1.0, -0.5,
            ],
            2,
            4,
        )
        .unwrap();
        let k = Tensor2D::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
        let v = Tensor2D::new(vec![2.0, 0.0, 0.0, 4.0, 1.0, 3.0], 3, 2).unwrap();
        let kv_indices = RocmU32Buffer::from_slice(&[0, 0]).unwrap();

        let got = RocmTensor::attention(
            &RocmTensor::from_2d(&q).unwrap(),
            &RocmTensor::from_2d(&k).unwrap(),
            &RocmTensor::from_2d(&v).unwrap(),
            &kv_indices,
            1,
            3,
            2,
        )
        .unwrap()
        .to_2d()
        .unwrap();

        let expected = reference_attention(&q, &k, &v, &[0, 0], 1, 3, 2);
        for (actual, expected) in got.data.iter().zip(expected.data.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "attention mismatch: actual={actual} expected={expected}"
            );
        }
    }

    fn reference_attention(
        q: &Tensor2D,
        k: &Tensor2D,
        v: &Tensor2D,
        local_kv_indices: &[u32],
        cache_prefix_len: usize,
        cache_seq_len: usize,
        head_dim: usize,
    ) -> Tensor2D {
        let local_q_heads = q.cols / head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = Tensor2D::zeros(q.rows, q.cols);
        for row in 0..q.rows {
            let visible = cache_prefix_len
                .saturating_add(row)
                .saturating_add(1)
                .min(cache_seq_len);
            for q_head in 0..local_q_heads {
                let kv_head = local_kv_indices[q_head] as usize;
                let mut scores = Vec::with_capacity(visible);
                for seq in 0..visible {
                    let mut score = 0.0;
                    for dim in 0..head_dim {
                        score += q.data[row * q.cols + q_head * head_dim + dim]
                            * k.data[seq * k.cols + kv_head * head_dim + dim];
                    }
                    scores.push(score * scale);
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denom = scores
                    .iter()
                    .map(|score| (*score - max_score).exp())
                    .sum::<f32>();
                for dim in 0..head_dim {
                    let mut value = 0.0;
                    for (seq, score) in scores.iter().enumerate() {
                        let weight = (*score - max_score).exp();
                        value += weight * v.data[seq * v.cols + kv_head * head_dim + dim];
                    }
                    out.data[row * q.cols + q_head * head_dim + dim] = value / denom;
                }
            }
        }
        out
    }
}
