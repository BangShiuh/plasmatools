import numpy as np

class VVSystem:
    """
    Calculate VV transfer rate coefficient
    """
    def __init__(self, w_ei, x_ei, w_ej, x_ej, c, S, sigma, rm):
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
        self.w_ej = w_ej
        self.x_ej = x_ej
        self.c = c #[K-1]
        self.S = S
        self.sigma = sigma
        self.kb = 1.381e-23
        self.rm = rm

    def dE(self, v, w):
        return self.w_ei * (1.0 - 2.0 * self.x_ei * v) - self.w_ej * (1.0 - 2.0 * self.x_ej * w)

    def lam(self, v, w, T):
        """
        λ = 2^(-1.5) * (c/T) ^ 0.5 * |ΔE_v|
        """
        return 2.0**(-1.5) * (self.c/T)**0.5 * np.abs(self.dE(v,w))

    def F(self, x):
        """
        F(λ) = [3-exp(-2λ/3)] * exp(-2λ/3)
        """
        return (3.0 - np.exp(-2.0 * x / 3.0)) * np.exp(-2.0 * x / 3.0)

    def S_vw(self, v, w, T):
        '''
        S(v->v-1, w-1->w) = 1/2 * S * T * v / (1-xe * v) * w / [1-xe * w] * F(λ),
        '''
        S_vw = 0.5 * self.S * T * v / (1 - self.x_ei * v) * w / (1 - self.x_ej * w) * self.F(self.lam(v,w,T))
        return S_vw

    def Z(self, T):
        return 4.0 * self.sigma * self.sigma * np.sqrt(np.pi * self.kb * T / 2.0 / self.rm)

    def k(self, v, w, T):
        return self.Z(T) * self.S_vw(v, w, T) * np.exp(-self.dE(v,w) / 2.0 / T)