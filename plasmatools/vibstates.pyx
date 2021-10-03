import numpy as np
from libc.math cimport exp
from libc.math cimport abs

def normalized_net_multipath_vibrational_excitation_rates(f_v, rate_constants, level):
    """
    Get normalized net vibrational excitation rates.

    :param f_v:
        The population fraction of the vibrational states (not include the highest vibration level)
    :param rate_constants:
        The rate constants of the electron collision process. (in vector form)
    :param level:
        The vibrational level.

    For example,

    N2(v) + e => N2(v+1) + e, k0
    N2(v) + e => N2(v+2) + e, k1
    N2(v) + e => N2(v+3) + e, k2

    The rate constant vector rate_constants = [k0, k1, k2]
    The population fraction vector is f_v = [f0, f1, f2]

    Define rate constant array as

            k0  0   0 
    k_arr = k1  k0  0
            k2  k1  k0

    The production rates equal to k_arr dot* f_v
    The destruction rates equal to the column summation of k_arr multiplies
    f_v element-wise.

    (k0 + k1 + k2) * f0  (destruction rate of f0)
    (0  + k0 + k1) * f1  (destruction rate of f1)
    (0  +  0 + k0) * f2  (destruction rate of f2)
    """
    k_arr = np.zeros([level, level])
    for v in range(level):
        diag_vec = np.ones(level-v) * rate_constants[v]
        k_arr += np.diag(diag_vec, -v)

    production_rates = np.insert(k_arr.dot(f_v[:level]), 0, 0)
    destruction_rates = np.insert(k_arr.sum(axis=0) * f_v[:level], level, 0)
    return production_rates - destruction_rates

def normalized_vibrational_excitation_rates(f_v, rate_constants, level):
    """
    Get normalized net vibrational excitation rates.

    :param f_v:
        The population fraction of the vibrational states (not include the highest vibration level)
    :param rate_constants:
        The rate constants of the electron collision process. (in vector form)
    :param level:
        The vibrational level.

    For example,

    N2(v) + e => N2(v+1) + e, k0
    N2(v) + e => N2(v+2) + e, k1
    N2(v) + e => N2(v+3) + e, k2

    The rate constant vector rate_constants = [k0, k1, k2]
    The population fraction vector is f_v = [f0, f1, f2]

    Define rate constant array as

            k0  0   0 
    k_arr = k1  k0  0
            k2  k1  k0

    The production rates equal to k_arr dot* f_v
    The destruction rates equal to the column summation of k_arr multiplies
    f_v element-wise.

    (k0 + k1 + k2) * f0  (destruction rate of f0)
    (0  + k0 + k1) * f1  (destruction rate of f1)
    (0  +  0 + k0) * f2  (destruction rate of f2)
    """
    k_arr = np.zeros([level, level])
    for v in range(level):
        diag_vec = np.ones(level-v) * rate_constants[v]
        k_arr += np.diag(diag_vec, -v)

    production_rates = np.insert(k_arr.dot(f_v[:level]), 0, 0)
    destruction_rates = np.insert(k_arr.sum(axis=0) * f_v[:level], level, 0)
    return production_rates - destruction_rates

def normalized_net_vibrational_quench_rates(f_v, rate_constants, delta_vt, level):
    """
    Get normalized net vibrational exchanging rates.

    k_v->v-1 = v * k_1->0 * exp[delta_vt(v-1)]

    :param f_v:
        The population fraction of the vibrational states (not include the highest vibration level)
    :param rate_constants:
        The rate constants of the electron collision process. (in vector form)
    :param delta_vt
        The radii of the VT relaxation.
    :param level
        The vibrational level.

    For example,

    N2(v2) + O => N2(v1) + O,

    k_2->1 = 2 * k_1->0 * exp(delta_vt)
    """
    v_vec = np.linspace(1, level, level)
    rates = v_vec * rate_constants * np.exp(delta_vt * (v_vec - 1)) * f_v[1:level+1]
    production_rates = np.insert(rates, level, 0)
    destruction_rates = np.insert(rates, 0, 0)
    return production_rates - destruction_rates

def normalized_net_vibrational_exchange_rates(double[:] f_v,
                                              double k01,
                                              double delta_vv,
                                              int level):

    cdef Py_ssize_t v, w
    cdef double rate
    net_rates = np.zeros(level+1)
    cdef double[:] net_rates_view = net_rates

    for v in range(level):
        for w in range(level):
            rate = ((v+1) * (w+1) * k01 *
                   exp(delta_vv * abs(v-w)) *
                   (1.5 - 0.5 * exp(delta_vv * abs(v-w))) *
                   f_v[v] * f_v[w+1]);
            net_rates_view[v+1] += rate;
            net_rates_view[w] += rate;
            net_rates_view[v] -= rate;
            net_rates_view[w+1] -= rate;
    return net_rates


# class VVTransferSystem:
#     """
#     Calculate VV transfer rate coefficient
#     """
#     def __init__(self, w_ei, x_ei, w_ej, x_ej, c, S, sigma, rm):
#         """
#         param w_ei:
#             The energy spacing between the vobrational energy levels for species i.
#             Correspond to index v. [K]
#         param w_ej:
#             The energy spacing between the vobrational energy levels for species j.
#             Correspond to index w. [K]
#         param x_ei:
#             The anharmonicity of species i. Correspond to index v. [K]
#         param x_ej:
#             The anharmonicity of species j. Correspond to index w. [K]
#         param c:
#             An adustable parameter for the short range interaction. [1/K]
#         param S:
#             An adustable parameter for the short range interaction.
#         param sigma:
#             The square root of collision cross section. [m]
#         """
#         self.w_ei = w_ei
#         self.x_ei = x_ei
#         self.w_ej = w_ej
#         self.x_ei = x_ei
#         self.c = c
#         self.S = S
#         self.sigma = sigma
#         self.kb = 1.381e-23
#         self.rm = rm

#     def dE(self):
#         return self.w_ei * (1.0 - 2.0 * self.x_ei) - self.w_ej * (1.0 - 2.0 * self.x_ej)

#     def lam(self, T):
#         """
#         λ = 2^(-1.5) * (c/T) ^ 0.5 * |ΔE_v|
#         """
#         return 2.0**(-1.5) * (self.c/T)**0.5 * abs(self.dE)

#     def F(self, x):
#         """
#         F(λ) = [3-exp(-2λ/3)] * exp(-2λ/3)
#         """
#         return (3.0 - exp(-2.0 * x / 3.0)) * exp(-2.0 * x / 3.0)

#     def S(self, v, w, T):
#         '''
#         S(v->v-1, w->w+1) = 1/2 * S * T * v / (1-xe * v) * (w+1) / [1-xe(w+1)] * F(λ),
#         '''
#         S_vw = self.S * T * v / (1 - self.x_ei * v) * w / (1 - self.x_ej * w) * self.F(self.lam(T))
#         return S_vw

#     def Z(self, T):
#         return 4.0 * self.sigma * self.sigma * np.sqrt(np.pi * self.kb * T / 2.0 / self.rm)

#     def k(self, v, w, T):
#         return self.Z(T) * self.S(v, w, T) * exp(-self.dE / 2.0 / T)

    # def nitrogen_VV_transfer_rate_constant(v, w, T):
        # c = 0.36
        # S = 1.5e-7
        # # Z = 3e-10 * (T/300.0)**0.5
        # sigma = 3.75e-10
        # kb = 1.381e-23
        # Z = 4.0 * sigma * sigma * np.sqrt(np.pi * kb * T / 4.65e-26)
        # xe = 6.0732e-3
        # omega_e = 3393.5
        # dE = omega_e * (1.0 - 2.0 * xe * v) - omega_e * (1.0 - 2.0 * xe * v)
        # lam = 2.0 ** (-1.5) * (c / T) ** 0.5 * abs(dE)
        # F = 0.5 * (3.0 - exp(-2.0 * lam / 3.0)) * exp(-2.0 * lam / 3.0)
        # S_vw = S * T * v / (1 - xe * v) * w / (1 - xe * w) * F
        # return Z * S_vw * exp(-dE / 2 / T)