// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

fn test_option_basic() {
    let some_val: Option<i32> = Some(42);
    let none_val: Option<i32> = None;

    assert!(some_val.is_some());
    assert!(!some_val.is_none());
    assert!(none_val.is_none());
    assert!(!none_val.is_some());
}

fn test_option_unwrap() {
    let val = Some(10i32);
    assert_eq!(val.unwrap(), 10);
}

fn test_option_map_or() {
    let some_val = Some(5i32);
    let none_val: Option<i32> = None;

    let a = some_val.map_or(0, |x| x * 2);
    assert_eq!(a, 10);

    let b = none_val.map_or(0, |x| x * 2);
    assert_eq!(b, 0);
}

fn test_option_and_then() {
    let some_val = Some(10i32);
    let result = some_val.and_then(|x| {
        if x > 5 { Some(x * 2) } else { None }
    });
    assert_eq!(result, Some(20));

    let none_result = some_val.and_then(|x| {
        if x > 100 { Some(x) } else { None }
    });
    assert!(none_result.is_none());
}

fn test_option_map() {
    let some_val = Some(3i32);
    let mapped = some_val.map(|x| x + 1);
    assert_eq!(mapped, Some(4));

    let none_val: Option<i32> = None;
    let mapped_none = none_val.map(|x| x + 1);
    assert!(mapped_none.is_none());
}

fn test_option_ok_or() {
    let some_val = Some(42i32);
    let ok: Result<i32, &str> = some_val.ok_or("error");
    assert!(ok.is_ok());

    let none_val: Option<i32> = None;
    let err: Result<i32, &str> = none_val.ok_or("error");
    assert!(err.is_err());
}

fn test_result_basic() {
    let ok_val: Result<i32, &str> = Ok(42);
    let err_val: Result<i32, &str> = Err("oops");

    assert!(ok_val.is_ok());
    assert!(!ok_val.is_err());
    assert!(err_val.is_err());
    assert!(!err_val.is_ok());
}

fn test_pattern_matching() {
    let val: Option<i32> = Some(10);

    let result = match val {
        Some(x) if x > 5 => x * 2,
        Some(x) => x,
        None => 0,
    };
    assert_eq!(result, 20);
}

fn test_if_let() {
    let val = Some(42i32);
    let mut found = false;

    if let Some(x) = val {
        assert_eq!(x, 42);
        found = true;
    }
    assert!(found);
}

fn test_while_let() {
    let mut opt = Some(3i32);
    let mut count = 0;

    while let Some(x) = opt {
        count = count + x;
        opt = if x > 1 { Some(x - 1) } else { None };
    }
    assert_eq!(count, 6); // 3 + 2 + 1
}
