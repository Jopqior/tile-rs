// build-pass

// RNN/sequence model building blocks.
// Maps to MultiKernelBench/reference/arch/ RNN category
// (vanilla_rnn, lstm, gru, mamba variants).

#![feature(no_core)]

#![no_std]
#![no_core]

/// Vanilla RNN cell: h_new = tanh(W_h * h + W_x * x + b)
/// Simplified: tanh(x + h * scale + bias)
#[tile_std::tile_kernel]
pub fn vanilla_rnn(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        // bh is dead after add, so output into bh
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// LSTM forget gate: f = sigmoid(W_f * [h, x] + b_f)
/// Simplified: sigmoid(x + h * scale + bias)
#[tile_std::tile_kernel]
pub fn lstm_forget_gate(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// LSTM input gate: i = sigmoid(W_i * [h, x] + b_i)
#[tile_std::tile_kernel]
pub fn lstm_input_gate(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// LSTM cell candidate: c_hat = tanh(W_c * [h, x] + b_c)
#[tile_std::tile_kernel]
pub fn lstm_cell_candidate(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// LSTM cell update: c_new = f * c_old + i * c_hat
#[tile_std::tile_kernel]
pub fn lstm_cell_update(c_old: *const f32, f_gate: *const f32, i_gate: *const f32, c_hat: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bc = tile_std::__tile_buf_alloc(n);
        let bf = tile_std::__tile_buf_alloc(n);
        let bi = tile_std::__tile_buf_alloc(n);
        let bch = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bc, c_old, n);
        tile_std::__tile_buf_load_f32(bf, f_gate, n);
        tile_std::__tile_buf_load_f32(bi, i_gate, n);
        tile_std::__tile_buf_load_f32(bch, c_hat, n);
        tile_std::__tile_pipe_barrier();
        // f * c_old → store in bf (bc and bf both needed, bf dead after)
        tile_std::__tile_v_mul_f32(bf, bc, bf, n);
        tile_std::__tile_pipe_barrier();
        // i * c_hat → store in bch (bi and bch both needed, bch dead after)
        tile_std::__tile_v_mul_f32(bch, bi, bch, n);
        tile_std::__tile_pipe_barrier();
        // c_new = f*c_old + i*c_hat
        tile_std::__tile_v_add_f32(bc, bf, bch, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bc, n);
    }
}

/// LSTM output gate + hidden: h = o * tanh(c)
#[tile_std::tile_kernel]
pub fn lstm_output(cell: *const f32, o_gate: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bc = tile_std::__tile_buf_alloc(n);
        let bo = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bc, cell, n);
        tile_std::__tile_buf_load_f32(bo, o_gate, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(bc, bc, n);
        tile_std::__tile_pipe_barrier();
        // bo is dead after, use as output
        tile_std::__tile_v_mul_f32(bo, bc, bo, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bo, n);
    }
}

/// GRU reset gate: r = sigmoid(W_r * [h, x] + b_r)
#[tile_std::tile_kernel]
pub fn gru_reset_gate(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// GRU update gate: z = sigmoid(W_z * [h, x] + b_z)
#[tile_std::tile_kernel]
pub fn gru_update_gate(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// GRU candidate: h_hat = tanh(W * [r*h, x] + b)
#[tile_std::tile_kernel]
pub fn gru_candidate(x: *const f32, h: *const f32, r_gate: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        let br = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_buf_load_f32(br, r_gate, n);
        tile_std::__tile_pipe_barrier();
        // r * h → store in br (dead after)
        tile_std::__tile_v_mul_f32(br, bh, br, n);
        tile_std::__tile_pipe_barrier();
        // x + r*h → store in br (bx dead after, br has r*h)
        tile_std::__tile_v_add_f32(bh, bx, br, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// GRU hidden update: h_new = (1-z)*h + z*h_hat
#[tile_std::tile_kernel]
pub fn gru_hidden_update(h: *const f32, z_gate: *const f32, h_hat: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bh = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);
        let bhh = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_buf_load_f32(bz, z_gate, n);
        tile_std::__tile_buf_load_f32(bhh, h_hat, n);
        tile_std::__tile_pipe_barrier();
        // (1-z)*h: negate z, add 1, multiply by h
        tile_std::__tile_muls_f32(tmp, bz, -1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(tmp, tmp, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // (1-z)*h → store in bh (dead after)
        tile_std::__tile_v_mul_f32(bh, tmp, bh, n);
        tile_std::__tile_pipe_barrier();
        // z*h_hat → store in bhh (dead after)
        tile_std::__tile_v_mul_f32(bhh, bz, bhh, n);
        tile_std::__tile_pipe_barrier();
        // sum
        tile_std::__tile_v_add_f32(tmp, bh, bhh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, tmp, n);
    }
}

// === Split variants for 1:1 MKB kernel mapping ===

/// vanilla_rnn_hidden - same as vanilla_rnn
#[tile_std::tile_kernel]
pub fn vanilla_rnn_hidden(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// lstm - same as lstm_forget_gate
#[tile_std::tile_kernel]
pub fn lstm(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// lstm_bidirectional - same as lstm_forget_gate
#[tile_std::tile_kernel]
pub fn lstm_bidirectional(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// lstm_cn - same as lstm_cell_candidate
#[tile_std::tile_kernel]
pub fn lstm_cn(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// gru - same as gru_reset_gate
#[tile_std::tile_kernel]
pub fn gru(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// gru_birectional - same as gru_reset_gate
#[tile_std::tile_kernel]
pub fn gru_birectional(x: *const f32, h: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bias = *config.wrapping_add(1);
        let bx = tile_std::__tile_buf_alloc(n);
        let bh = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bh, bh, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bh, bx, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bh, bh, bias, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bh, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bh, n);
    }
}

/// gru_bidirectional_hidden - same as gru_hidden_update
#[tile_std::tile_kernel]
pub fn gru_bidirectional_hidden(h: *const f32, z_gate: *const f32, h_hat: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bh = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);
        let bhh = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_buf_load_f32(bz, z_gate, n);
        tile_std::__tile_buf_load_f32(bhh, h_hat, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(tmp, bz, -1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(tmp, tmp, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bh, tmp, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bhh, bz, bhh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(tmp, bh, bhh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, tmp, n);
    }
}

/// gru_hidden - same as gru_hidden_update
#[tile_std::tile_kernel]
pub fn gru_hidden(h: *const f32, z_gate: *const f32, h_hat: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bh = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);
        let bhh = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bh, h, n);
        tile_std::__tile_buf_load_f32(bz, z_gate, n);
        tile_std::__tile_buf_load_f32(bhh, h_hat, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(tmp, bz, -1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(tmp, tmp, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bh, tmp, bh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bhh, bz, bhh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(tmp, bh, bhh, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, tmp, n);
    }
}
