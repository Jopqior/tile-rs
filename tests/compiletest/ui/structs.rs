// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Basic struct with fields
struct Point {
    x: i32,
    y: i32,
}

fn test_struct_creation() {
    let p = Point { x: 10, y: 20 };
    assert_eq!(p.x, 10);
    assert_eq!(p.y, 20);
}

fn test_struct_modify() {
    let mut p = Point { x: 1, y: 2 };
    p.x = 100;
    p.y = 200;
    assert_eq!(p.x, 100);
    assert_eq!(p.y, 200);
}

// Struct with methods
struct Counter {
    count: i32,
}

impl Counter {
    fn new() -> Self {
        Counter { count: 0 }
    }

    fn increment(&mut self) {
        self.count = self.count + 1;
    }

    fn value(&self) -> i32 {
        self.count
    }
}

fn test_struct_methods() {
    let mut c = Counter::new();
    assert_eq!(c.value(), 0);
    c.increment();
    c.increment();
    c.increment();
    assert_eq!(c.value(), 3);
}

// Struct with generic types
struct Pair<T> {
    first: T,
    second: T,
}

impl<T: Copy> Pair<T> {
    fn swap(&self) -> Pair<T> {
        Pair {
            first: self.second,
            second: self.first,
        }
    }
}

fn test_generic_struct() {
    let p = Pair { first: 10i32, second: 20i32 };
    let swapped = p.swap();
    assert_eq!(swapped.first, 20);
    assert_eq!(swapped.second, 10);
}

// Nested structs
struct Rect {
    origin: Point,
    width: i32,
    height: i32,
}

fn test_nested_struct() {
    let r = Rect {
        origin: Point { x: 5, y: 10 },
        width: 100,
        height: 50,
    };
    assert_eq!(r.origin.x, 5);
    assert_eq!(r.origin.y, 10);
    assert_eq!(r.width, 100);
    assert_eq!(r.height, 50);
}

fn test_struct_area() {
    let r = Rect {
        origin: Point { x: 0, y: 0 },
        width: 7,
        height: 6,
    };
    let area = r.width * r.height;
    assert_eq!(area, 42);
}

// Function returning struct
fn make_point(x: i32, y: i32) -> Point {
    Point { x, y }
}

fn test_return_struct() {
    let p = make_point(3, 4);
    let dist_sq = p.x * p.x + p.y * p.y;
    assert_eq!(dist_sq, 25);
}

// Array of structs
fn test_array_of_structs() {
    let points = [
        Point { x: 1, y: 2 },
        Point { x: 3, y: 4 },
        Point { x: 5, y: 6 },
    ];
    let mut sum_x = 0i32;
    let mut i = 0usize;
    loop {
        if i >= 3 { break; }
        sum_x = sum_x + points[i].x;
        i = i + 1;
    }
    assert_eq!(sum_x, 9); // 1 + 3 + 5
}
