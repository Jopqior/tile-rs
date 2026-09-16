//! PyTorch CPU f32 reference. Same input values as the kernel; CPU device.

use std::io::Write;
use std::process::{Command, Stdio};

use crate::cases::Case;
use crate::compare::{assert_close, require_dtype, require_shape, CompareError};

pub struct TorchRef {
    pub version: String,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub enum RefError {
    Unverified(String),
    Fail(String),
}

impl std::fmt::Display for RefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefError::Unverified(s) | RefError::Fail(s) => write!(f, "{s}"),
        }
    }
}

fn json_array(xs: &[f32]) -> String {
    let parts: Vec<String> = xs.iter().map(|x| {
        if x.is_finite() {
            format!("{x}")
        } else {
            "null".into()
        }
    }).collect();
    format!("[{}]", parts.join(","))
}

pub fn torch_version() -> Result<String, RefError> {
    let out = Command::new("python3")
        .args(["-c", "import torch; print(torch.__version__, end='')"])
        .output()
        .map_err(|e| RefError::Unverified(format!("python3 not runnable: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(RefError::Unverified(format!(
            "PyTorch CPU import failed: {err}"
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn add_f32_cpu(case: &Case) -> Result<TorchRef, RefError> {
    let version = torch_version()?;
    let payload = format!(
        "{{\"a\":{},\"b\":{}}}",
        json_array(case.a),
        json_array(case.b)
    );
    let py = r#"
import json, sys, torch
req = json.load(sys.stdin)
a = torch.tensor(req["a"], dtype=torch.float32, device="cpu")
b = torch.tensor(req["b"], dtype=torch.float32, device="cpu")
c = a + b
if c.dtype != torch.float32:
    sys.stderr.write("dtype="+str(c.dtype)+"\n")
    sys.exit(2)
if tuple(c.shape) != tuple(a.shape):
    sys.stderr.write("shape="+str(tuple(c.shape))+"\n")
    sys.exit(2)
if not torch.isfinite(c).all():
    sys.stderr.write("nonfinite reference\n")
    sys.exit(2)
sys.stdout.write(",".join(repr(float(x)) for x in c.detach().reshape(-1)))
"#;
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(py)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| RefError::Unverified(format!("python3 spawn failed: {e}")))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| RefError::Fail("python stdin missing".into()))?
        .write_all(payload.as_bytes())
        .map_err(|e| RefError::Fail(format!("write torch payload: {e}")))?;
    let out = child
        .wait_with_output()
        .map_err(|e| RefError::Fail(format!("wait torch: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(RefError::Fail(format!("torch add failed: {err}")));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut values = Vec::new();
    for part in stdout.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let v: f32 = part.parse().map_err(|e| {
            RefError::Fail(format!("parse torch output {part:?}: {e}"))
        })?;
        values.push(v);
    }
    Ok(TorchRef { version, values })
}

/// Check the PyTorch result against the independent hand anchors.
pub fn check_hand_anchor(case: &Case, got: &[f32]) -> Result<(), CompareError> {
    require_dtype("f32", case.dtype)?;
    require_shape(&[got.len()], case.shape)?;
    assert_close(got, case.hand_expected)
}
