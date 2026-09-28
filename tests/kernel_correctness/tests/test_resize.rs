use kernel_correctness::resize;

const TOL: f32 = 1e-5;

fn assert_approx(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(err < TOL, "{ctx} elem {i}: got {g}, want {w}, err {err}");
    }
}

// ====================================================================
// Bilinear upsample 2D
// ====================================================================

#[test]
fn bilinear_identity() {
    // Same size -> identity
    let input = [1.0, 2.0, 3.0, 4.0]; // 1ch, 2x2
    let mut output = [0.0; 4];
    resize::bilinear_upsample_2d(&input, &mut output, 1, 2, 2, 2, 2);
    assert_approx(&output, &[1.0, 2.0, 3.0, 4.0], "bilinear_id");
}

#[test]
fn bilinear_upsample_2x() {
    // 1ch, 2x2 -> 3x3 (align_corners)
    let input = [0.0, 1.0, 2.0, 3.0]; // 2x2
    let mut output = [0.0; 9];
    resize::bilinear_upsample_2d(&input, &mut output, 1, 2, 2, 3, 3);
    // Corners should be exact: [0,0]=0, [0,2]=1, [2,0]=2, [2,2]=3
    assert!((output[0] - 0.0).abs() < TOL, "corner 0,0");
    assert!((output[2] - 1.0).abs() < TOL, "corner 0,2");
    assert!((output[6] - 2.0).abs() < TOL, "corner 2,0");
    assert!((output[8] - 3.0).abs() < TOL, "corner 2,2");
    // Center should be average of all 4: (0+1+2+3)/4 = 1.5
    assert!((output[4] - 1.5).abs() < TOL, "center");
}

#[test]
fn bilinear_1x1_input() {
    // 1x1 -> NxN should replicate the single value
    let input = [42.0];
    let mut output = [0.0; 4]; // 2x2
    resize::bilinear_upsample_2d(&input, &mut output, 1, 1, 1, 2, 2);
    assert_approx(&output, &[42.0, 42.0, 42.0, 42.0], "bilinear_1x1");
}

// ====================================================================
// Nearest upsample 2D
// ====================================================================

#[test]
fn nearest_upsample_2x() {
    // 1ch, 2x2 -> 4x4
    let input = [1.0, 2.0, 3.0, 4.0]; // 2x2
    let mut output = [0.0; 16];
    resize::nearest_upsample_2d(&input, &mut output, 1, 2, 2, 4, 4);
    let expected = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_approx(&output, &expected, "nearest_2x");
}

#[test]
fn nearest_identity() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];
    resize::nearest_upsample_2d(&input, &mut output, 1, 2, 2, 2, 2);
    assert_approx(&output, &[1.0, 2.0, 3.0, 4.0], "nearest_id");
}

// ====================================================================
// Trilinear upsample 3D
// ====================================================================

#[test]
fn trilinear_identity() {
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect(); // 1ch, 2x2x2
    let mut output = [0.0; 8];
    resize::trilinear_upsample_3d(&input, &mut output, 1, 2, 2, 2, 2, 2, 2);
    assert_approx(&output, &input, "trilinear_id");
}

#[test]
fn trilinear_corners() {
    // 1ch, 2x2x2 -> 3x3x3
    let input = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // 2x2x2
    let mut output = [0.0; 27]; // 3x3x3
    resize::trilinear_upsample_3d(&input, &mut output, 1, 2, 2, 2, 3, 3, 3);
    // Corners should match exactly
    assert!((output[0] - 0.0).abs() < TOL, "corner 0,0,0"); // input[0,0,0]
    assert!((output[2] - 1.0).abs() < TOL, "corner 0,0,2"); // input[0,0,1]
    assert!((output[6] - 2.0).abs() < TOL, "corner 0,2,0"); // input[0,1,0]
    assert!((output[18] - 4.0).abs() < TOL, "corner 2,0,0"); // input[1,0,0]
    assert!((output[26] - 7.0).abs() < TOL, "corner 2,2,2"); // input[1,1,1]
}

// ====================================================================
// Downsample bilinear 2D
// ====================================================================

#[test]
fn downsample_bilinear() {
    // 1ch, 3x3 -> 2x2
    let input = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 3x3
    let mut output = [0.0; 4];
    resize::downsample_bilinear_2d(&input, &mut output, 1, 3, 3, 2, 2);
    // Corners: [0,0]=0, [0,1]=2, [1,0]=6, [1,1]=8
    assert_approx(&output, &[0.0, 2.0, 6.0, 8.0], "downsample");
}

// ====================================================================
// Multi-channel
// ====================================================================

#[test]
fn bilinear_multichannel() {
    let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // 2ch, 2x2
    let mut output = [0.0; 8]; // 2ch, 2x2 (identity)
    resize::bilinear_upsample_2d(&input, &mut output, 2, 2, 2, 2, 2);
    assert_approx(&output, &input, "bilinear_mch");
}
