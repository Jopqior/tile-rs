use kernel_correctness::conv;

const TOL: f32 = 1e-5;

fn assert_approx(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(err < TOL, "{ctx} elem {i}: got {g}, want {w}, err {err}");
    }
}

// ====================================================================
// conv_standard_1d
// ====================================================================

#[test]
fn conv1d_identity_kernel() {
    // kernel=[1] acts as identity
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0];
    let mut output = vec![0.0; 5];
    conv::conv_standard_1d(&input, &weight, &mut output, 1, 1, 5, 1, 1);
    assert_approx(&output, &[1.0, 2.0, 3.0, 4.0, 5.0], "conv1d identity");
}

#[test]
fn conv1d_sum_kernel() {
    // kernel=[1,1,1] sums sliding window of 3
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0];
    let mut output = vec![0.0; 3];
    conv::conv_standard_1d(&input, &weight, &mut output, 1, 1, 5, 3, 1);
    assert_approx(&output, &[6.0, 9.0, 12.0], "conv1d sum3");
}

#[test]
fn conv1d_stride2() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight = vec![1.0, 1.0];
    let mut output = vec![0.0; 3];
    conv::conv_standard_1d(&input, &weight, &mut output, 1, 1, 6, 2, 2);
    assert_approx(&output, &[3.0, 7.0, 11.0], "conv1d stride2");
}

#[test]
fn conv1d_multi_channel() {
    // 2 input channels, 1 output channel, length=3, kernel=2
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // ic0=[1,2,3], ic1=[4,5,6]
    let weight = vec![1.0, 0.0, 0.0, 1.0]; // oc0: ic0=[1,0], ic1=[0,1]
    let mut output = vec![0.0; 2];
    conv::conv_standard_1d(&input, &weight, &mut output, 2, 1, 3, 2, 1);
    // out[0] = 1*1+2*0 + 4*0+5*1 = 1+5 = 6
    // out[1] = 2*1+3*0 + 5*0+6*1 = 2+6 = 8
    assert_approx(&output, &[6.0, 8.0], "conv1d multi_ch");
}

#[test]
fn conv1d_multi_output_channel() {
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, -1.0, -1.0]; // oc0=[1,1], oc1=[-1,-1]
    let mut output = vec![0.0; 4];
    conv::conv_standard_1d(&input, &weight, &mut output, 1, 2, 3, 2, 1);
    // oc0: [3.0, 5.0], oc1: [-3.0, -5.0]
    assert_approx(&output, &[3.0, 5.0, -3.0, -5.0], "conv1d multi_oc");
}

// ====================================================================
// conv_standard_1d_dilated_strided
// ====================================================================

#[test]
fn conv1d_dilated() {
    // dilation=2: kernel taps at positions 0, 2
    let input = vec![1.0, 0.0, 2.0, 0.0, 3.0];
    let weight = vec![1.0, 1.0];
    let mut output = vec![0.0; 3];
    conv::conv_standard_1d_dilated_strided(&input, &weight, &mut output, 1, 1, 5, 2, 1, 2);
    // eff_k = (2-1)*2+1 = 3, out_len = (5-3)/1+1 = 3
    // out[0] = in[0]+in[2] = 3, out[1] = in[1]+in[3] = 0, out[2] = in[2]+in[4] = 5
    assert_approx(&output, &[3.0, 0.0, 5.0], "conv1d dilated");
}

// ====================================================================
// conv_standard_2d
// ====================================================================

#[test]
fn conv2d_identity() {
    // 1x1 kernel = identity
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1ch, 2x2
    let weight = vec![1.0]; // 1x1x1x1
    let mut output = vec![0.0; 4];
    conv::conv_standard_2d(&input, &weight, &mut output, 1, 1, 2, 2, 1, 1, 1);
    assert_approx(&output, &[1.0, 2.0, 3.0, 4.0], "conv2d identity");
}

#[test]
fn conv2d_sum_3x3() {
    // 3x3 all-ones kernel sums 3x3 window
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect(); // 1ch, 4x4
    let weight = vec![1.0; 9]; // 1x1x3x3
    let mut output = vec![0.0; 4]; // oh=ow=2
    conv::conv_standard_2d(&input, &weight, &mut output, 1, 1, 4, 4, 3, 3, 1);
    // top-left 3x3: 1+2+3+5+6+7+9+10+11 = 54
    // top-right 3x3: 2+3+4+6+7+8+10+11+12 = 63
    assert_approx(&output, &[54.0, 63.0, 90.0, 99.0], "conv2d sum3x3");
}

#[test]
fn conv2d_stride2() {
    let input: Vec<f32> = (1..=25).map(|x| x as f32).collect(); // 1ch, 5x5
    let weight = vec![1.0; 4]; // 1x1x2x2
    let mut output = vec![0.0; 4]; // oh=ow=2
    conv::conv_standard_2d(&input, &weight, &mut output, 1, 1, 5, 5, 2, 2, 2);
    // out[0,0] = 1+2+6+7 = 16
    // out[0,1] = 3+4+8+9 = 24
    // out[1,0] = 11+12+16+17 = 56
    // out[1,1] = 13+14+18+19 = 64
    assert_approx(&output, &[16.0, 24.0, 56.0, 64.0], "conv2d stride2");
}

#[test]
fn conv2d_asymmetric_kernel() {
    // 1x1 input, but test asymmetric kernel
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 1ch, 2x3
    let weight = vec![1.0, -1.0]; // 1x1x1x2
    let mut output = vec![0.0; 4]; // oh=2, ow=2
    conv::conv_standard_2d(&input, &weight, &mut output, 1, 1, 2, 3, 1, 2, 1);
    // out[0,0] = 1-2 = -1, out[0,1] = 2-3 = -1
    // out[1,0] = 4-5 = -1, out[1,1] = 5-6 = -1
    assert_approx(&output, &[-1.0, -1.0, -1.0, -1.0], "conv2d asym_k");
}

// ====================================================================
// conv_standard_2d_dilated_padded
// ====================================================================

#[test]
fn conv2d_padded() {
    // padding=1, kernel=3x3, stride=1, dilation=1: same-size output
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1ch, 2x2
    let weight = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // center=1 (identity)
    let mut output = vec![0.0; 4];
    conv::conv_standard_2d_dilated_padded(&input, &weight, &mut output, 1, 1, 2, 2, 3, 3, 1, 1, 1);
    assert_approx(&output, &[1.0, 2.0, 3.0, 4.0], "conv2d padded identity");
}

// ====================================================================
// conv_standard_3d
// ====================================================================

#[test]
fn conv3d_identity() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 1ch, 2x2x2
    let weight = vec![1.0]; // 1x1x1x1x1
    let mut output = vec![0.0; 8];
    conv::conv_standard_3d(&input, &weight, &mut output, 1, 1, 2, 2, 2, 1, 1, 1, 1);
    assert_approx(&output, &input, "conv3d identity");
}

#[test]
fn conv3d_sum() {
    let input: Vec<f32> = (1..=27).map(|x| x as f32).collect(); // 1ch, 3x3x3
    let weight = vec![1.0; 8]; // 1x1x2x2x2
    let mut output = vec![0.0; 8]; // od=oh=ow=2
    conv::conv_standard_3d(&input, &weight, &mut output, 1, 1, 3, 3, 3, 2, 2, 2, 1);
    // out[0,0,0] = sum of 2x2x2 block at (0,0,0) = 1+2+4+5+10+11+13+14 = 60
    assert!((output[0] - 60.0).abs() < TOL, "conv3d sum corner");
}

// ====================================================================
// Depthwise convolutions
// ====================================================================

#[test]
fn depthwise_2d_identity() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // 2ch, 2x2
                                                                  // For kh=kw=1: weight is [w_ch0, w_ch1] = [1.0, 1.0]
    let weight = vec![1.0, 1.0]; // each channel has 1x1 kernel = 1.0
    let mut output = vec![0.0; 8];
    conv::conv_depthwise_2d(&input, &weight, &mut output, 2, 2, 2, 1, 1, 1);
    assert_approx(
        &output,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "dw identity",
    );
}

#[test]
fn depthwise_2d_sum() {
    // 1ch, 3x3 input, 2x2 kernel=[1,1,1,1]
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let mut output = vec![0.0; 4];
    conv::conv_depthwise_2d(&input, &weight, &mut output, 1, 3, 3, 2, 2, 1);
    // out[0,0] = 1+2+4+5 = 12
    // out[0,1] = 2+3+5+6 = 16
    // out[1,0] = 4+5+7+8 = 24
    // out[1,1] = 5+6+8+9 = 28
    assert_approx(&output, &[12.0, 16.0, 24.0, 28.0], "dw sum");
}

// ====================================================================
// Pointwise convolution
// ====================================================================

#[test]
fn pointwise_2d() {
    // 2 input channels -> 1 output channel, 2x2 spatial
    let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // ic0=[1,2,3,4], ic1=[10,20,30,40]
    let weight = vec![1.0, 0.5]; // oc0: 1.0*ic0 + 0.5*ic1
    let mut output = vec![0.0; 4];
    conv::conv_pointwise_2d(&input, &weight, &mut output, 2, 1, 2, 2);
    assert_approx(&output, &[6.0, 12.0, 18.0, 24.0], "pointwise");
}

// ====================================================================
// Transposed convolutions
// ====================================================================

#[test]
fn conv_transposed_1d_basic() {
    // in_ch=1, out_ch=1, in_len=3, k=2, stride=1
    // output_len = (3-1)*1 + 2 = 4
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0];
    let mut output = vec![0.0; 4];
    conv::conv_transposed_1d(&input, &weight, &mut output, 1, 1, 3, 2, 1);
    // scatter-add: pos0 += 1*1=1, pos1 += 1*1+2*1=3, pos2 += 2*1+3*1=5, pos3 += 3*1=3
    assert_approx(&output, &[1.0, 3.0, 5.0, 3.0], "transposed_1d");
}

#[test]
fn conv_transposed_1d_stride2() {
    // stride=2: output_len = (3-1)*2 + 2 = 6
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0];
    let mut output = vec![0.0; 6];
    conv::conv_transposed_1d(&input, &weight, &mut output, 1, 1, 3, 2, 2);
    // p=0: pos0+=1, pos1+=1; p=1: pos2+=2, pos3+=2; p=2: pos4+=3, pos5+=3
    assert_approx(&output, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], "transposed_1d_s2");
}

#[test]
fn conv_transposed_2d_basic() {
    // 1x1 input -> expand to kh x kw
    let input = vec![2.0]; // 1ch, 1x1
    let weight = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let mut output = vec![0.0; 4]; // oh=ow=2
    conv::conv_transposed_2d(&input, &weight, &mut output, 1, 1, 1, 1, 2, 2, 1);
    assert_approx(&output, &[2.0, 4.0, 6.0, 8.0], "transposed_2d");
}

#[test]
fn conv_transposed_2d_padded_basic() {
    // padding=1 with 3x3 kernel, 1x1 input, stride=1
    // oh = (1-1)*1 + 3 - 2*1 = 1
    let input = vec![5.0]; // 1ch, 1x1
    let weight = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // center=1
    let mut output = vec![0.0; 1];
    conv::conv_transposed_2d_padded(&input, &weight, &mut output, 1, 1, 1, 1, 3, 3, 1, 1);
    assert_approx(&output, &[5.0], "transposed_2d_padded");
}

#[test]
fn conv_transposed_3d_basic() {
    let input = vec![3.0]; // 1ch, 1x1x1
    let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 1x1x2x2x2
    let mut output = vec![0.0; 8]; // od=oh=ow=2
    conv::conv_transposed_3d(&input, &weight, &mut output, 1, 1, 1, 1, 1, 2, 2, 2, 1);
    let expected: Vec<f32> = (1..=8).map(|x| x as f32 * 3.0).collect();
    assert_approx(&output, &expected, "transposed_3d");
}
