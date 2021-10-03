import numpy as np

class VTSystem:
    """
    Calculate VV transfer rate coefficient
    """
    def __init__(self, w_ei, x_ei, c, A, B, C):
        """
        param w_ei:
            The energy spacing between the vobrational energy levels for species i.
            Correspond to index v. [K]
        param w_ej:
            The energy spacing between the vobrational energy levels for species j.
            Correspond to index w. [K]
        param x_ei:
            The anharmonicity of species i. Correspond to index v. [K]
        param x_ej:
            The anharmonicity of species j. Correspond to index w. [K]
        param c:
            An adustable parameter for the short range interaction. [1/K]
        param S:
            An adustable parameter for the short range interaction.
        param sigma:
            The square root of collision cross section. [m]
        """
        self.w_ei = w_ei
        self.x_ei = x_ei
        self.c = c
        self.A = A
        self.B = B
        self.C = C
        self.kb = 1.381e-23

    def dE(self, v):
        return self.w_ei * (1.0 - 2.0 * self.x_ei * v)

    def lam(self, v, T):
        """
        λ = 2^(-1.5) * (c/T) ^ 0.5 * |ΔE_v|
        """
        return 2.0**(-1.5) * (self.c/T)**0.5 * np.abs(self.dE(v))

    def F(self, x):
        """
        F(λ) = [3-exp(-2λ/3)] * exp(-2λ/3)
        """
        return (3.0 - np.exp(-2.0 * x / 3.0)) * np.exp(-2.0 * x / 3.0)

    def tau_P(self, T):
        return np.exp(self.A + self.B * T**(-1./3.) + self.C * T**(-2./3.)) * 101325 * 1e-6

    def k(self, v, T):
        return (self.kb * T / self.tau_P(T) /
               self.F(self.lam(1.0, T)) /
               (1. - np.exp(-self.w_ei / T)) *
               v / (1.0 - self.x_ei * v) *
               self.F(self.lam(v, T)))
