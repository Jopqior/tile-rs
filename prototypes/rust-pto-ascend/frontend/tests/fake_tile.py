#!/usr/bin/env python3
"""Offline CLI double: copy independently authored per-case stage outputs."""

import hashlib
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
stage = args[args.index("-t") + 1]
source = Path(args[0])
case = source.stem
output = Path(args[args.index("-o") + 1])
fixtures = Path(os.environ["FAKE_FIXTURES"]) / case
trace = Path(os.environ["FAKE_TRACE"])
record = {
    "case": case,
    "stage": stage,
    "args": args,
    "rustflags": os.environ.get("RUSTFLAGS"),
    "tile_std_path": os.environ.get("TILE_STD_PATH"),
    "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}
with trace.open("a") as stream:
    stream.write(json.dumps(record) + "\n")
print(f"fake {stage} {case}")
if os.environ.get("FAKE_FAIL_CASE") == case and os.environ.get("FAKE_FAIL_STAGE") == stage:
    sys.exit(9)
if stage == "msl":
    kernel = Path(args[args.index("--keep-dir") + 1])
    kernel.mkdir(parents=True)
    if not (os.environ.get("FAKE_OMIT_CASE") == case and
            os.environ.get("FAKE_OMIT_STAGE") == stage):
        (kernel / f"{case}.2.mlir.mlir").write_bytes((fixtures / "compiler.mlir").read_bytes())
    output.write_text("fake metal\n")
elif stage == "pto":
    output.write_bytes((fixtures / "emitted.pto.mlir").read_bytes())
else:
    sys.exit(f"unexpected stage: {stage}")
