"""Answers that used to hang the scorer (real MATH rollouts) must be handled in bounded time and not change normal results."""

import time
import unittest

from task3_test_time import equivalence as E


class TestPathologicalAnswers(unittest.TestCase):
    def test_huge_power_pair_returns_quickly(self):
        # real pair from the supplied MATH rollouts: sympy.expand of the first one never terminates
        t0 = time.time()
        self.assertFalse(E.are_answers_equivalent(r"(2 + \sqrt{2 + \sqrt{2}})^{2525}", r"2^{100}", dataset="math"))
        self.assertLess(time.time() - t0, 15)

    def test_small_exponents_still_use_the_symbolic_check(self):
        self.assertTrue(E._sympy_symbolic_equiv("x^2 - 1", "(x-1)*(x+1)"))
        self.assertTrue(E.are_answers_equivalent(r"\frac{1}{2}", "0.5", dataset="math"))
        self.assertFalse(E.are_answers_equivalent("3", "4", dataset="math"))

    def test_exponent_guard(self):
        self.assertTrue(E._small_exponents("2**10 + x**32"))
        self.assertFalse(E._small_exponents("x**33"))
        self.assertFalse(E._small_exponents("(2+sqrt(2))**2525"))
        self.assertFalse(E._small_exponents("3**(100)"))


if __name__ == "__main__":
    unittest.main()
