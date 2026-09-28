// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Basic function call
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn test_basic_call() {
    assert_eq!(add(3, 4), 7);
}

// Recursive function
fn factorial(n: i32) -> i32 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn test_recursion() {
    assert_eq!(factorial(1), 1);
    assert_eq!(factorial(5), 120);
    assert_eq!(factorial(6), 720);
}

// Function with multiple return paths
fn classify(x: i32) -> i32 {
    if x < 0 {
        -1
    } else if x == 0 {
        0
    } else {
        1
    }
}

fn test_multiple_returns() {
    assert_eq!(classify(-5), -1);
    assert_eq!(classify(0), 0);
    assert_eq!(classify(42), 1);
}

// Generic function
fn max_of<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

fn test_generic_function() {
    assert_eq!(max_of(3i32, 7i32), 7);
    assert_eq!(max_of(10i32, 2i32), 10);
}

// Nested function calls
fn square(x: i32) -> i32 {
    x * x
}

fn sum_of_squares(a: i32, b: i32) -> i32 {
    add(square(a), square(b))
}

fn test_nested_calls() {
    assert_eq!(sum_of_squares(3, 4), 25);
}

// Function with loop
fn sum_range(start: i32, end: i32) -> i32 {
    let mut sum = 0i32;
    let mut i = start;
    loop {
        if i >= end { break; }
        sum = sum + i;
        i = i + 1;
    }
    sum
}

fn test_loop_function() {
    assert_eq!(sum_range(1, 6), 15); // 1+2+3+4+5
    assert_eq!(sum_range(0, 0), 0);
}

// Closure usage
fn test_closure_basic() {
    let double = |x: i32| x * 2;
    assert_eq!(double(5), 10);
    assert_eq!(double(0), 0);
}

fn test_closure_capture() {
    let offset = 10i32;
    let add_offset = |x: i32| x + offset;
    assert_eq!(add_offset(5), 15);
    assert_eq!(add_offset(0), 10);
}

// Higher-order function
fn apply(f: fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}

fn test_higher_order() {
    assert_eq!(apply(square, 5), 25);
    assert_eq!(apply(|x| x + 1, 10), 11);
}

// Function pointer array
fn test_fn_ptr_array() {
    let ops: [fn(i32, i32) -> i32; 2] = [add, |a, b| a * b];
    assert_eq!(ops[0](3, 4), 7);
    assert_eq!(ops[1](3, 4), 12);
}
