// =============================================================================
// NPU Kernels for Multi-Head Attention
// =============================================================================
//
// Two kernels used in the MHA pipeline:
//   1. scale_f16: element-wise multiply by a scalar (1/sqrt(d_k))
//   2. softmax_rows_f16: row-wise softmax over a matrix stored in row-major order

#![feature(no_core)]
#![no_std]
#![no_core]

/// Scale kernel: output[i] = input[i] * scale_factor
///
/// Parameters:
///   - input: pointer to f16 input data (as u16)
///   - output: pointer to f16 output data (as u16)
///   - n: number of elements (single-element buffer)
///   - scale: scale factor as f32 (single-element buffer)
#[tile_std::tile_kernel]
pub fn scale_f16(input: *const u16, output: *mut u16, n: *const u32, scale: *const f32) {
    unsafe {
        let count = *n;
        let scale_val = *scale;

        let buf_in = tile_std::__tile_buf_alloc(count);
        let buf_out = tile_std::__tile_buf_alloc(count);

        tile_std::__tile_buf_load_f16(buf_in, input, count);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f16(buf_out, buf_in, scale_val, count);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(output, buf_out, count);
    }
}

/// Row-wise softmax kernel for f16 data.
///
/// Processes `num_rows` rows of `row_len` elements each.
/// For each row: max → subtract max → exp → sum → divide by sum.
///
/// Parameters:
///   - input: pointer to f16 input matrix (row-major, as u16)
///   - output: pointer to f16 output matrix (as u16)
///   - row_len: number of columns per row (single-element buffer)
///   - num_rows: number of rows (single-element buffer)
#[tile_std::tile_kernel]
pub fn softmax_rows_f16(
    input: *const u16,
    output: *mut u16,
    row_len: *const u32,
    num_rows: *const u32,
) {
    unsafe {
        let cols = *row_len;
        let rows = *num_rows;

        let buf_in = tile_std::__tile_buf_alloc(cols);
        let buf_out = tile_std::__tile_buf_alloc(cols);
        let buf_work = tile_std::__tile_buf_alloc(cols);
        let buf_rwork = tile_std::__tile_buf_alloc(cols);

        let mut row = 0u32;
        loop {
            if row >= rows {
                break;
            }

            let row_offset = row * cols;
            let in_ptr = input.wrapping_add(row_offset as usize);
            let out_ptr = output.wrapping_add(row_offset as usize);

            // Load one row
            tile_std::__tile_buf_load_f16(buf_in, in_ptr, cols);
            tile_std::__tile_pipe_barrier();

            // ReduceMax → max_val
            let max_val = tile_std::__tile_reduce_max_f16(buf_work, buf_in, buf_rwork, cols);

            // Subtract max: out = in - max
            let neg_max = 0.0f32 - max_val;
            tile_std::__tile_adds_f16(buf_out, buf_in, neg_max, cols);
            tile_std::__tile_pipe_barrier();

            // Exp
            tile_std::__tile_v_exp_f16(buf_out, buf_out, cols);
            tile_std::__tile_pipe_barrier();

            // ReduceSum → sum_val
            let sum_val = tile_std::__tile_reduce_sum_f16(buf_work, buf_out, buf_rwork, cols);

            // Divide by sum: out = out * (1/sum)
            let inv_sum = 1.0f32 / sum_val;
            tile_std::__tile_muls_f16(buf_out, buf_out, inv_sum, cols);

            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f16(out_ptr, buf_out, cols);

            row = row + 1;
        }
    }
}
