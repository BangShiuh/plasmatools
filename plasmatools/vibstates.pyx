import numpy as np
from libc.math cimport exp
from libc.math cimport abs

def normalized_net_vibrational_excitation_rates(f_v, rate_constants, level):
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
    cdef double[:] net_rates = np.zeros(level+1)
    
    for v in range(level):
        for w in range(level):
            rate = ((v+1) * (w+1) * k01 *
                   exp(delta_vv * abs(v-w)) *
                   (1.5 - 0.5 * exp(delta_vv * abs(v-w))) *
                   f_v[v] * f_v[w+1]);
            net_rates[v+1] += rate;
            net_rates[w] += rate;
            net_rates[v] -= rate;
            net_rates[w+1] -= rate;
    return net_rates
