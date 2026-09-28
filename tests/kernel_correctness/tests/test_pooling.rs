use kernel_correctness::pooling;

const TOL: f32 = 1e-5;

fn assert_approx(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(err < TOL, "{ctx} elem {i}: got {g}, want {w}, err {err}");
    }
}

// ====================================================================
// Max pooling
// ====================================================================

#[test]
fn max_pool_1d_basic() {
    let input = [1.0, 3.0, 2.0, 5.0, 4.0];
    let mut output = [0.0; 3];
    pooling::max_pooling_1d(&input, &mut output, 5, 3, 1);
    assert_approx(&output, &[3.0, 5.0, 5.0], "maxpool1d");
}

#[test]
fn max_pool_1d_stride2() {
    let input = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let mut output = [0.0; 2];
    pooling::max_pooling_1d(&input, &mut output, 6, 3, 2);
    assert_approx(&output, &[3.0, 5.0], "maxpool1d_s2");
}

#[test]
fn max_pool_1d_k_equals_input() {
    // Global max pool
    let input = [3.0, 1.0, 4.0, 1.0, 5.0];
    let mut output = [0.0; 1];
    pooling::max_pooling_1d(&input, &mut output, 5, 5, 1);
    assert_approx(&output, &[5.0], "maxpool1d_global");
}

#[test]
fn max_pool_2d_basic() {
    // 1ch, 4x4, kernel 2x2, stride 2
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let mut output = [0.0; 4];
    pooling::max_pooling_2d(&input, &mut output, 1, 4, 4, 2, 2, 2);
    // [0,0]: max(1,2,5,6)=6, [0,1]: max(3,4,7,8)=8
    // [1,0]: max(9,10,13,14)=14, [1,1]: max(11,12,15,16)=16
    assert_approx(&output, &[6.0, 8.0, 14.0, 16.0], "maxpool2d");
}

#[test]
fn max_pool_2d_multichannel() {
    let input = [
        1.0, 5.0, 3.0, 7.0, // ch0: 2x2
        10.0, 50.0, 30.0, 70.0, // ch1: 2x2
    ];
    let mut output = [0.0; 2]; // 2ch, 1x1
    pooling::max_pooling_2d(&input, &mut output, 2, 2, 2, 2, 2, 1);
    assert_approx(&output, &[7.0, 70.0], "maxpool2d_mch");
}

#[test]
fn max_pool_3d_basic() {
    let input: Vec<f32> = (1..=27).map(|x| x as f32).collect(); // 1ch, 3x3x3
    let mut output = [0.0; 1]; // global pool
    pooling::max_pooling_3d(&input, &mut output, 1, 3, 3, 3, 3, 3, 3, 1);
    assert_approx(&output, &[27.0], "maxpool3d_global");
}

// ====================================================================
// Average pooling
// ====================================================================

#[test]
fn avg_pool_1d_basic() {
    let input = [2.0, 4.0, 6.0, 8.0, 10.0];
    let mut output = [0.0; 3];
    pooling::average_pooling_1d(&input, &mut output, 5, 3, 1);
    assert_approx(&output, &[4.0, 6.0, 8.0], "avgpool1d");
}

#[test]
fn avg_pool_1d_k1() {
    // k_size=1: identity
    let input = [1.0, 2.0, 3.0];
    let mut output = [0.0; 3];
    pooling::average_pooling_1d(&input, &mut output, 3, 1, 1);
    assert_approx(&output, &[1.0, 2.0, 3.0], "avgpool1d_k1");
}

#[test]
fn avg_pool_2d_basic() {
    // 1ch, 4x4, kernel 2x2, stride 2
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let mut output = [0.0; 4];
    pooling::average_pooling_2d(&input, &mut output, 1, 4, 4, 2, 2, 2);
    // [0,0]: mean(1,2,5,6)=3.5, [0,1]: mean(3,4,7,8)=5.5
    // [1,0]: mean(9,10,13,14)=11.5, [1,1]: mean(11,12,15,16)=13.5
    assert_approx(&output, &[3.5, 5.5, 11.5, 13.5], "avgpool2d");
}

#[test]
fn avg_pool_3d_basic() {
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect(); // 1ch, 2x2x2
    let mut output = [0.0; 1];
    pooling::average_pooling_3d(&input, &mut output, 1, 2, 2, 2, 2, 2, 2, 1);
    // mean(1..=8) = 4.5
    assert_approx(&output, &[4.5], "avgpool3d_global");
}

#[test]
fn avg_pool_2d_non_overlapping() {
    // k_size = stride (non-overlapping windows)
    let input = [1.0, 2.0, 3.0, 4.0]; // 1ch, 2x2
    let mut output = [0.0; 1];
    pooling::average_pooling_2d(&input, &mut output, 1, 2, 2, 2, 2, 2);
    assert_approx(&output, &[2.5], "avgpool2d_nonoverlap");
}
