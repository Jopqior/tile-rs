use kernel_correctness::index;

#[test]
fn test_argmax() {
    assert_eq!(index::argmax(&[1.0, 5.0, 3.0, 2.0]), 1);
    assert_eq!(index::argmax(&[5.0, 1.0, 3.0, 2.0]), 0);
    assert_eq!(index::argmax(&[1.0, 2.0, 3.0, 5.0]), 3);
    assert_eq!(index::argmax(&[42.0]), 0);
}

#[test]
fn test_argmin() {
    assert_eq!(index::argmin(&[3.0, 1.0, 5.0, 2.0]), 1);
    assert_eq!(index::argmin(&[1.0, 5.0, 3.0, 2.0]), 0);
    assert_eq!(index::argmin(&[-5.0, 2.0, 3.0, 1.0]), 0);
    assert_eq!(index::argmin(&[42.0]), 0);
}

#[test]
fn test_gather() {
    let input = [10.0, 20.0, 30.0, 40.0, 50.0];
    let index = [4, 2, 0, 3, 1];
    let mut output = [0.0; 5];
    index::gather(&input, &index, &mut output);
    assert_eq!(output, [50.0, 30.0, 10.0, 40.0, 20.0]);
}

#[test]
fn test_scatter() {
    let src = [100.0, 200.0, 300.0];
    let index = [2, 0, 4];
    let mut output = [0.0; 5];
    index::scatter(&src, &index, &mut output);
    assert_eq!(output, [200.0, 0.0, 100.0, 0.0, 300.0]);
}

#[test]
fn test_scatter_add() {
    let src = [1.0, 2.0, 3.0];
    let index = [0, 0, 0]; // all accumulate to position 0
    let mut output = [10.0, 0.0, 0.0];
    index::scatter_add(&src, &index, &mut output);
    assert_eq!(output, [16.0, 0.0, 0.0]); // 10 + 1 + 2 + 3 = 16
}

#[test]
fn test_scatter_add_distinct() {
    let src = [1.0, 2.0, 3.0];
    let index = [0, 1, 2];
    let mut output = [0.0; 3];
    index::scatter_add(&src, &index, &mut output);
    assert_eq!(output, [1.0, 2.0, 3.0]);
}

#[test]
fn test_index_select() {
    // 5 rows of length 3, select rows [2, 0]
    let input: Vec<f32> = (0..15).map(|x| x as f32).collect();
    let index = [2, 0];
    let mut output = [0.0; 6];
    index::index_select(&input, &index, &mut output, 3);
    assert_eq!(output, [6.0, 7.0, 8.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_index_copy() {
    let src = [100.0, 101.0, 200.0, 201.0]; // 2 rows of length 2
    let index = [3, 1]; // copy row 0 to position 3, row 1 to position 1
    let mut output = [0.0; 8]; // 4 rows of length 2
    index::index_copy(&src, &index, &mut output, 2);
    assert_eq!(output, [0.0, 0.0, 200.0, 201.0, 0.0, 0.0, 100.0, 101.0]);
}

#[test]
fn test_index_add() {
    let src = [1.0, 2.0, 3.0, 4.0]; // 2 rows of length 2
    let index = [0, 0]; // both add to row 0
    let mut output = [10.0, 20.0, 30.0, 40.0]; // 2 rows of length 2
    index::index_add(&src, &index, &mut output, 2);
    assert_eq!(output, [14.0, 26.0, 30.0, 40.0]); // row 0: 10+1+3=14, 20+2+4=26
}

#[test]
fn test_embedding() {
    // Weight table: 3 embeddings of dim 2
    let weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = [2, 0, 1];
    let mut output = [0.0; 6];
    index::embedding(&weight, &indices, &mut output, 2);
    assert_eq!(output, [5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_masked_fill() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mask = [true, false, true, false, true];
    let mut output = [0.0; 5];
    index::masked_fill(&input, &mask, &mut output, -1.0);
    assert_eq!(output, [-1.0, 2.0, -1.0, 4.0, -1.0]);
}

#[test]
fn test_masked_fill_all_false() {
    let input = [1.0, 2.0, 3.0];
    let mask = [false, false, false];
    let mut output = [0.0; 3];
    index::masked_fill(&input, &mask, &mut output, 99.0);
    assert_eq!(output, [1.0, 2.0, 3.0]);
}

#[test]
fn test_inplace_update() {
    let values = [100.0, 200.0];
    let index = [3, 1];
    let mut output = [0.0; 5];
    index::inplace_update(&values, &index, &mut output);
    assert_eq!(output, [0.0, 200.0, 0.0, 100.0, 0.0]);
}

#[test]
fn test_take_along_dim() {
    // 2 rows of 3 columns (inner=3)
    let input = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let index = [2, 0, 1, 1, 2, 0]; // 6 elements
    let mut output = [0.0; 6];
    index::take_along_dim(&input, &index, &mut output, 3);
    // row 0: [input[0+2], input[0+0], input[0+1]] = [30, 10, 20]
    // row 1: [input[3+1], input[3+2], input[3+0]] = [50, 60, 40]
    assert_eq!(output, [30.0, 10.0, 20.0, 50.0, 60.0, 40.0]);
}
