"""Synthetic fixture helpers; these are not real DSL or semantic PTO kernels."""

import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as frontend

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_TILE = Path(__file__).resolve().parent / "fake_tile.py"


def fixture_names():
    return [case.name for case in frontend.discover_cases(FIXTURES)]


def fixture(name):
    path = FIXTURES / name
    return (path / "compiler.mlir").read_text(), (path / "emitted.pto.mlir").read_text(), frontend.expectations(path / "expectations.json")


def copy_case(cases, name):
    source = FIXTURES / name
    dest = cases / name
    dest.mkdir()
    for filename in (f"{name}.rs", "expectations.json"):
        shutil.copyfile(source / filename, dest / filename)
    return dest


def trace(path):
    return [json.loads(line) for line in path.read_text().splitlines()]
