// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Labeled break from nested loop
fn test_labeled_break() {
    let mut result = 0i32;
    'outer: loop {
        let mut j = 0i32;
        loop {
            if j == 3 {
                result = 42;
                break 'outer;
            }
            j = j + 1;
        }
    }
    assert_eq!(result, 42);
}

// Labeled continue
fn test_labeled_continue() {
    let mut count = 0i32;
    let mut i = 0i32;
    'outer: loop {
        if i >= 3 { break; }
        let mut j = 0i32;
        loop {
            if j >= 3 { i = i + 1; continue 'outer; }
            count = count + 1;
            j = j + 1;
        }
    }
    assert_eq!(count, 9); // 3 * 3
}

// Loop with value (break with value)
fn test_loop_value() {
    let mut i = 0i32;
    let result = loop {
        if i == 10 {
            break i * 2;
        }
        i = i + 1;
    };
    assert_eq!(result, 20);
}

// Nested conditionals
fn classify(x: i32) -> i32 {
    if x < -10 {
        -2
    } else if x < 0 {
        -1
    } else if x == 0 {
        0
    } else if x < 10 {
        1
    } else {
        2
    }
}

fn test_nested_if() {
    assert_eq!(classify(-100), -2);
    assert_eq!(classify(-5), -1);
    assert_eq!(classify(0), 0);
    assert_eq!(classify(5), 1);
    assert_eq!(classify(100), 2);
}

// Match with complex patterns
fn test_match_range() {
    let x = 5i32;
    let result = match x {
        0 => 0,
        1..=5 => 1,
        6..=10 => 2,
        _ => 3,
    };
    assert_eq!(result, 1);
}

fn test_match_tuple() {
    let point = (1i32, -1i32);
    let quadrant = match point {
        (x, y) if x > 0 && y > 0 => 1,
        (x, y) if x < 0 && y > 0 => 2,
        (x, y) if x < 0 && y < 0 => 3,
        (x, y) if x > 0 && y < 0 => 4,
        _ => 0, // on an axis
    };
    assert_eq!(quadrant, 4);
}

// Nested loops with counters
fn test_nested_loops() {
    let mut total = 0i32;
    let mut i = 0i32;
    loop {
        if i >= 4 { break; }
        let mut j = 0i32;
        loop {
            if j >= i { break; }
            total = total + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    assert_eq!(total, 6); // 0 + 1 + 2 + 3
}

// While-like pattern with early return
fn find_first_ge(arr: &[i32; 5], target: i32) -> i32 {
    let mut i = 0usize;
    loop {
        if i >= 5 { return -1; }
        if arr[i] >= target { return i as i32; }
        i = i + 1;
    }
}

fn test_early_return() {
    let arr = [2i32, 5, 8, 11, 14];
    assert_eq!(find_first_ge(&arr, 8), 2);
    assert_eq!(find_first_ge(&arr, 1), 0);
    assert_eq!(find_first_ge(&arr, 100), -1);
}

// Conditional assignment
fn test_conditional_assign() {
    let x = 7i32;
    let y = if x % 2 == 0 { x / 2 } else { x * 3 + 1 };
    assert_eq!(y, 22);
}

// Match with binding
fn test_match_binding() {
    let opt: Option<i32> = Some(42);
    let result = match opt {
        Some(x @ 1..=50) => x * 2,
        Some(x) => x,
        None => 0,
    };
    assert_eq!(result, 84);
}
