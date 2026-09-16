// build-pass
// ignore-test: borrow checker ICE with IntoIterator for arrays in no_core

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;
use tile_std::iter::*;

fn test_fold() {
    let arr = [1i32, 2, 3, 4, 5];
    let _sum = IntoIterator::into_iter(arr).take(3).fold(0i32, |acc, x| acc + x);
}

fn test_map_type() {
    let arr = [1i32, 2, 3];
    let _mapped = IntoIterator::into_iter(arr).map(|x| x * 2);
}

fn test_filter_type() {
    let arr = [1i32, 2, 3, 4];
    let _filtered = IntoIterator::into_iter(arr).filter(|x| *x > 2);
}

fn test_zip_type() {
    let a = [1i32, 2, 3];
    let b = [4i32, 5, 6];
    let _zipped = IntoIterator::into_iter(a).zip(IntoIterator::into_iter(b));
}

fn test_enumerate_type() {
    let arr = [10i32, 20, 30];
    let _enumerated = IntoIterator::into_iter(arr).enumerate();
}

fn test_chain_type() {
    let a = [1i32, 2];
    let b = [3i32, 4];
    let _chained = IntoIterator::into_iter(a).chain(IntoIterator::into_iter(b));
}

fn test_sum() {
    let arr = [1i32, 2, 3];
    let _s: i32 = IntoIterator::into_iter(arr).take(2).sum();
}

fn test_product() {
    let arr = [2i32, 3, 4];
    let _p: i32 = IntoIterator::into_iter(arr).take(2).product();
}

fn test_filter_map_type() {
    let arr = [1i32, 2, 3, 4];
    let _fm = IntoIterator::into_iter(arr).filter_map(|x| {
        if x > 2 { Some(x * 10) } else { None }
    });
}

fn test_combinator_chaining() {
    let arr = [1i32, 2, 3, 4, 5];
    let _result = IntoIterator::into_iter(arr)
        .filter(|x| *x > 1)
        .map(|x| x * 2)
        .take(3);
}

fn test_take() {
    let arr = [1i32, 2, 3, 4, 5];
    let _first_three = IntoIterator::into_iter(arr).take(3);
}
