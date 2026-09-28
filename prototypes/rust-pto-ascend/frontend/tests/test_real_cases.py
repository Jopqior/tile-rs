"""Discover actual cases and validate only their config and source completeness.

Backend compilation and compiler output validation belong to run.py in CI, not
these offline contract tests. No real-kernel expectations are duplicated here.
"""

import tempfile
import unittest
from pathlib import Path

from support import frontend


class RealCaseContractTests(unittest.TestCase):
    def test_every_discovered_case_has_valid_config_and_nonempty_source(self):
        shared_cases = frontend.ROOT / "prototypes/rust-pto-ascend/cases"
        self.assertEqual(frontend.CASES, shared_cases)
        cases = frontend.discover_cases(shared_cases)
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case.name):
                frontend.expectations(case / "expectations.json")
                frontend.require_nonempty_file(case / f"{case.name}.rs")

    def test_empty_real_suite_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ValueError, "empty frontend suite"):
                frontend.discover_cases(Path(empty))

    def test_ci_checkout_matches_runner_pin(self):
        workflow = (frontend.ROOT / ".github/workflows/prototype-rust-pto-frontend.yml").read_text()
        self.assertIn(f"ref: {frontend.SDK_COMMIT}", workflow)


if __name__ == "__main__":
    unittest.main()
