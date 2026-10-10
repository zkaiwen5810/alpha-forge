import unittest

from alpha_forge_langchain.tools import calculator


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculator.invoke({"expression": "-(2 + 3) * 4 + 11 // 2"}), "-15")
        self.assertEqual(calculator.invoke({"expression": "2 ** 3 % 3"}), "2")

    def test_rejects_code_and_unbounded_computation(self):
        for expression in (
            "", "__import__('os').getcwd()", "True + 1", "x", "[1, 2]",
            "2 ** 101", "10 ** 100 * 10", "1e309", "(-1) ** 0.5",
            "1 / 0", "1 + " * 100 + "1", "0" * 2001,
        ):
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError, ArithmeticError, SyntaxError)):
                    calculator.invoke({"expression": expression})


if __name__ == "__main__":
    unittest.main()
