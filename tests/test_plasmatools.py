#!/usr/bin/env python

"""Tests for `plasmatools` package."""


import unittest
import numpy as np
from plasmatools import vibstates as vs

class TestPlasmatools(unittest.TestCase):
    """Tests for `plasmatools` package."""

    def test_normalized_net_vibrational_excitation_rates(self):
        f_v = [0.5, 0.25, 0.25]
        rate_constants = [1.0, 0.5, 0.25]
        level = 3
        rates = vs.normalized_net_vibrational_excitation_rates(f_v, rate_constants, level)
        rates_solution = np.array([-0.875, 0.125, 0.25, 0.5])
        np.testing.assert_array_equal(rates, rates_solution)

    def test_normalized_net_vibrational_quench_rates(self):
        f_v = [0.5, 0.25, 0.125, 0.125]
        rate_constants = [0.25, 0.5, 1.0]
        level = 3
        delta_vt = 1.0
        rates = vs.normalized_net_vibrational_quench_rates(f_v, rate_constants, delta_vt, level)
        rates_solution = np.array([0.0625,  0.27729,  2.43111, -2.7709])
        np.testing.assert_array_almost_equal(rates, rates_solution, decimal=5)

if __name__ == '__main__':
    unittest.main()
