"""Structural checker tests against independently authored synthetic outputs."""

import re
import unittest

from support import fixture, fixture_names, frontend


class CheckerTests(unittest.TestCase):
    def test_distinct_fixtures_have_distinct_contracts_and_pass(self):
        names = fixture_names()
        self.assertGreaterEqual(len(names), 2)
        configs = [fixture(name)[2] for name in names]
        self.assertEqual(len({config["function"] for config in configs}), len(names))
        self.assertGreater(len({config["pointer_type"] for config in configs}), 1)
        self.assertGreater(len({config["pointer_count"] for config in configs}), 1)
        self.assertGreater(len({tuple(config["operations"].items()) for config in configs}), 1)
        for name in names:
            with self.subTest(case=name):
                mlir, pto, expected = fixture(name)
                self.assertNotEqual(name, expected["function"])
                frontend.check_outputs(mlir, pto, expected)

    def test_mutations_detect_intrinsic_function_pointer_view_ssa_and_operations(self):
        for name in fixture_names():
            mlir, pto, expected = fixture(name)
            operation = next(op for op, count in expected["operations"].items() if count)
            mutations = [
                (mlir.replace(expected["mlir_intrinsic"], "__absent", 1), pto, "missing MLIR intrinsic"),
                (mlir, pto.replace("@" + expected["function"], "@absent", 1), "PTO function"),
                (mlir, pto.replace("!pto.ptr<" + expected["pointer_type"] + ">",
                                   "!pto.ptr<wrong>", 1), "ordered"),
                (mlir, pto.replace("%arg0: !pto.ptr<" + expected["pointer_type"] + ">, ", "", 1),
                 "ordered"),
                (mlir, pto.replace("pto.make_tensor_view %arg0", "pto.make_tensor_view %arg1", 1),
                 "tensor view from input pointer"),
                (mlir, re.sub(r"(pto\.tstore )%[A-Za-z_]\w*", r"\1%undefined_operand", pto, count=1),
                 "undefined PTO SSA operands"),
                (mlir, pto.replace(operation, "pto.absent_op", 1), re.escape(operation) + ": expected"),
            ]
            for bad_mlir, bad_pto, error in mutations:
                with self.subTest(case=name, error=error), self.assertRaisesRegex(ValueError, error):
                    frontend.check_outputs(bad_mlir, bad_pto, expected)

    def test_zero_expected_operations_reject_new_active_operation(self):
        zero_checks = 0
        for name in fixture_names():
            mlir, pto, expected = fixture(name)
            for operation, count in expected["operations"].items():
                if count == 0:
                    zero_checks += 1
                    with self.subTest(case=name, operation=operation), self.assertRaisesRegex(
                        ValueError, re.escape(operation) + ": expected 0, got 1"
                    ):
                        frontend.check_outputs(mlir, pto.replace("\n}", f"\n  {operation}\n}}"), expected)
        self.assertGreater(zero_checks, 0)


if __name__ == "__main__":
    unittest.main()
