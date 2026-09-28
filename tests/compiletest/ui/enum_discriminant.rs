// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// C-like enum with explicit discriminants
#[repr(i32)]
enum Color {
    Red = 1,
    Green = 2,
    Blue = 3,
}

// Enum with default discriminants (0, 1, 2, 3)
enum Direction {
    North,
    South,
    East,
    West,
}

fn test_c_like_enum_match() {
    let c = Color::Red;
    let val = match c {
        Color::Red => 10,
        Color::Green => 20,
        Color::Blue => 30,
    };
    assert_eq!(val, 10);
}

fn test_c_like_enum_all_variants() {
    let colors = [Color::Red, Color::Green, Color::Blue];
    let mut sum = 0i32;
    let mut i = 0usize;
    loop {
        if i >= 3 { break; }
        sum = sum + match colors[i] {
            Color::Red => 1,
            Color::Green => 2,
            Color::Blue => 3,
        };
        i = i + 1;
    }
    assert_eq!(sum, 6); // 1 + 2 + 3
}

fn test_four_variant_enum() {
    let dir = Direction::East;
    let is_horizontal = match dir {
        Direction::North | Direction::South => false,
        Direction::East | Direction::West => true,
    };
    assert!(is_horizontal);
}

fn test_direction_all_variants() {
    let dirs = [Direction::North, Direction::South, Direction::East, Direction::West];
    let mut ns_count = 0i32;
    let mut ew_count = 0i32;
    let mut i = 0usize;
    loop {
        if i >= 4 { break; }
        match dirs[i] {
            Direction::North | Direction::South => ns_count = ns_count + 1,
            Direction::East | Direction::West => ew_count = ew_count + 1,
        }
        i = i + 1;
    }
    assert_eq!(ns_count, 2);
    assert_eq!(ew_count, 2);
}

// Enum with data in variants
enum Shape {
    Circle(i32),       // radius
    Rectangle(i32, i32), // width, height
    Point,
}

fn test_data_enum_match() {
    let s = Shape::Circle(5);
    let area = match s {
        Shape::Circle(r) => r * r, // approximate
        Shape::Rectangle(w, h) => w * h,
        Shape::Point => 0,
    };
    assert_eq!(area, 25);
}

fn test_data_enum_rectangle() {
    let s = Shape::Rectangle(3, 4);
    let area = match s {
        Shape::Circle(r) => r * r,
        Shape::Rectangle(w, h) => w * h,
        Shape::Point => 0,
    };
    assert_eq!(area, 12);
}

fn test_data_enum_point() {
    let s = Shape::Point;
    let area = match s {
        Shape::Circle(r) => r * r,
        Shape::Rectangle(w, h) => w * h,
        Shape::Point => 0,
    };
    assert_eq!(area, 0);
}

fn test_nested_enum_match() {
    let val: Option<Option<i32>> = Some(Some(42));
    let result = match val {
        Some(Some(x)) => x,
        Some(None) => -1,
        None => -2,
    };
    assert_eq!(result, 42);

    let val2: Option<Option<i32>> = Some(None);
    let result2 = match val2 {
        Some(Some(x)) => x,
        Some(None) => -1,
        None => -2,
    };
    assert_eq!(result2, -1);

    let val3: Option<Option<i32>> = None;
    let result3 = match val3 {
        Some(Some(x)) => x,
        Some(None) => -1,
        None => -2,
    };
    assert_eq!(result3, -2);
}
