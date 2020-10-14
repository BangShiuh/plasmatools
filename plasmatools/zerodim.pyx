import numpy as np
import cantera as ct
from libc.math cimport exp
from libc.math cimport abs

# NRPs Perfect stir reactor
class ReactorOde:
    def __init__(self
                 , gas
                 , gas_vib
                 , EN_method
                 , discharge_radius
                 , k_interp
                 , kv_N2_interp
                 , N_PLASMA_RXN
                 , N_VIB_RXN
                 , t_res
                 , Te_interp
                 , EN_interp):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.P0 = gas.P
        self.kE = gas.species_index('E')
        self.kO2p = gas.species_index('O2^+')
        self.kO2 = gas.species_index('O2')
        self.kO = gas.species_index('O')
        self.kN2 = gas.species_index('N2')
        self.kN = gas.species_index('N')
        # reduced electric field
        self.EN = 0.0
        # laplacian electric field
        self.EL = 0.0
        # coflow constants
        self.T0 = gas.T
        self.Y0 = gas.Y
        self.h0 = gas.partial_molar_enthalpies / gas.molecular_weights
        self.rho0 = gas.density
        self.k0 = gas.thermal_conductivity #thermal conductivity
        # get species transport
        self.D0 = self.gas.mix_diff_coeffs
        self.D0_N2 = self.gas.mix_diff_coeffs_mole[self.kN2]
        for k in range(self.gas.n_species):
            if self.gas.species(k).charge != 0:
                self.D0[k] = (self.gas.mix_diff_coeffs_mole[self.kO2p]
                       * (1 + self.gas.electron_temperature / self.gas.T))
        self.gas_vib = gas_vib
        self.N2v_indices = [self.gas_vib.species_index('N2'),
                            self.gas_vib.species_index('N2(v1)'),
                            self.gas_vib.species_index('N2(v2)'),
                            self.gas_vib.species_index('N2(v3)'),
                            self.gas_vib.species_index('N2(v4)'),
                            self.gas_vib.species_index('N2(v5)'),
                            self.gas_vib.species_index('N2(v6)'),
                            self.gas_vib.species_index('N2(v7)'),
                            self.gas_vib.species_index('N2(v8)')]
        self.h0_mole = gas_vib.partial_molar_enthalpies
        self.f_N2v_0 = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.electric_model = "current-voltage"
        self.pulse_switch = 0.0
        self.r_d = discharge_radius
        self.cross_section_area = discharge_radius * discharge_radius * np.pi
        self.EN_method = EN_method
        self.k_interp = k_interp
        self.kv_N2_interp = kv_N2_interp
        self.N_PLASMA_RXN = N_PLASMA_RXN
        self.N_VIB_RXN = N_VIB_RXN
        self.t_res = t_res
        self.Te_interp = Te_interp
        self.EN_interp = EN_interp
    
    def __call__(self, t, y):

        """the ODE function, y' = f(t,y) """
        # pressure
        t_acoust = self.r_d / (331 * np.sqrt(self.gas.T / 273))
        P_ = self.P0 + self.rho0 * ct.gas_constant * (self.gas.T - self.T0) * np.exp(-(t / t_acoust)**2)

        # State vector is [T, Y_1, Y_2, ... Y_K]
        try:
            N_VIB_SPECIES = self.N_VIB_RXN + 1
            f_N2v = y[-N_VIB_SPECIES:]
            self.gas.set_unnormalized_mass_fractions(y[1:-9])
            self.gas.TP = y[0], P_
            rho = self.gas.density
            h = self.gas.partial_molar_enthalpies / self.gas.molecular_weights
            ND_ = self.gas.P / (ct.boltzmann * self.gas.T)
            Ne_ = self.gas.concentrations[self.kE] * ct.avogadro
            dfdt_N2v = np.zeros(N_VIB_SPECIES)
        except:
            print(y[0], P_)

        # set vibrational states
        self.gas_vib.TP = y[0], P_
        conc_array = np.zeros(self.gas_vib.n_species)
        conc_array[self.gas_vib.species_index('E')] = self.gas.concentrations[self.kE]
        conc_array[self.gas_vib.species_index('N')] = self.gas.concentrations[self.kN]
        conc_array[self.gas_vib.species_index('O')] = self.gas.concentrations[self.kO]
        for i, k in enumerate(self.N2v_indices):
            conc_array[k] = self.gas.concentrations[self.kN2] * f_N2v[i]

        self.gas_vib.concentrations = conc_array

        # get species transport
        D = self.gas.mix_diff_coeffs

        for k in range(self.gas.n_species):
            if self.gas.species(k).charge != 0:
                D[k] = (self.gas.mix_diff_coeffs_mole[self.kO2p]
                       * (1 + self.gas.electron_temperature / self.gas.T))

        Q_electron = 0.0
        vib_reaction_index = 0

        if self.pulse_switch == 0.0:
            self.gas.electron_temperature = self.gas.T
            for r in range(self.N_PLASMA_RXN):
                self.gas.set_multiplier(0.0, r)
            for i in range(self.N_VIB_RXN):
                for v in range(self.N_VIB_RXN-i):
                    self.gas_vib.set_multiplier(0.0, vib_reaction_index)
                    vib_reaction_index += 1
            self.EN = 0.0
        else:
            if self.electric_model == "direct-EN":
                self.EN = np.abs(self.EL / ND_)
            elif self.electric_model == "current-voltage":
                # use interpolation function to obtain E/N
                j_e_ne_ = np.abs(self.current / self.cross_section_area / (ct.electron_charge * Ne_))
                try:
                    self.EN = min(1000 * 1e-21, np.abs(self.EL / ND_), self.EN_interp(j_e_ne_))
                except:
                    self.EN = min(1000 * 1e-21, np.abs(self.EL / ND_))

            self.gas.electron_temperature = max(self.gas.T, self.Te_interp(self.EN))
            for r in range(self.N_PLASMA_RXN):
                self.gas.set_multiplier(self.k_interp[r](self.EN)*1e6, r)
                Q_electron += self.gas.heat_production_rates[r]

            # Eval vibrational excitatation of nitrogen
            for i in range(8):
                for v in range(8-i):
                    self.gas_vib.set_multiplier(self.kv_N2_interp[i](self.EN)*1e6, vib_reaction_index)
                    Q_electron += self.gas_vib.heat_production_rates[vib_reaction_index]
                    vib_reaction_index += 1

        wdot = self.gas.net_production_rates

        # The base equation of an adiabatic perfect-stir reactor
        dTdt = (-np.dot(self.gas.partial_molar_enthalpies, wdot) / rho
               + 1.0 / self.t_res * (np.dot(self.Y0, self.h0) - np.dot(self.Y0, h))) / self.gas.cp
        dYdt = wdot * self.gas.molecular_weights / rho + 1.0 / self.t_res * (self.Y0 - self.gas.Y)

        # electron collision heating term for plasma
        dTdt -= Q_electron / (rho * self.gas.cp)

        # Mass diffusion
        mass_diffusion = ((self.rho0 + rho) / 2.0 * (self.D0 + D) / 2.0 *
                          (self.Y0 - self.gas.Y) / self.r_d / self.r_d
                         )
        dYdt += mass_diffusion / rho

        # Thermal conduction
        dTdt += ((self.gas.thermal_conductivity + self.k0) / 2.0 *
                 (self.T0 - self.gas.T) / self.r_d / self.r_d
                ) / (rho * self.gas.cp)

        # Energy loss due to mass diffusion
        for k in range(self.gas.n_species):
            if k != self.kN2:
                dTdt += (self.h0[k] * max(mass_diffusion[k], 0.0) +
                         h[k] * min(mass_diffusion[k], 0.0)) / (rho * self.gas.cp)

        # Eval V-T relaxation of nitrogen by collision to oxygen atoms
        kvt_10 = self.gas.T * np.exp(-34.03 - 33.11 * self.gas.T**(-1./3.)) # Starikovskiy 2017
        delta_vt = 2.87 * (self.gas.T)**(-1.0/3.0) # Lanier 2015
        for v in range(self.N_VIB_RXN):
            kvt = (v+1) * kvt_10 * np.exp(delta_vt*v)
            self.gas_vib.set_multiplier(kvt, vib_reaction_index)
            vib_reaction_index += 1

        # Eval V-T relaxation of nitrogen by collision to nitrogen atoms
        for v in range(self.N_VIB_RXN):
            kvt = (v+1) * kvt_10 * np.exp(delta_vt*v)
            self.gas_vib.set_multiplier(kvt, vib_reaction_index)
            vib_reaction_index += 1

        # Eval V-T relaxation of nitrogen by collision to nitrogen molecules
        kvt_10 = self.gas.T * np.exp(-22.86 - 328.9 * self.gas.T**(-1./3.) +
                                     993.3 * self.gas.T**(-2./3.)) # Starikovskiy 2017
        for v in range(self.N_VIB_RXN):
            kvt = (v+1) * kvt_10 * np.exp(delta_vt*v)
            self.gas_vib.set_multiplier(kvt, vib_reaction_index)
            vib_reaction_index += 1

        # Eval V-V transfer of nitrogen to nitrogen
#         delta_vv = 6.8 / (self.gas.T)**0.5 # Gordiets 1995
#         kvv_0 = 1.5e-14 * (self.gas.T/300)**1.5 # Gordiets 1995
#         for w in range(1,8):
#             for v in range(w):
#                 kvv = (v+1) * (w+1) * kvv_0 * np.exp(delta_vv*(w-v))*(1.5-0.5*np.exp(delta_vv*(w-v)))
#                 self.gas_vib.set_multiplier(kvv, vib_reaction_index)
#                 vib_reaction_index += 1

        vib_wdot = self.gas_vib.net_production_rates
        dfdt_N2v += vib_wdot[-N_VIB_SPECIES:] / self.gas.concentrations[self.kN2]
        
        D_N2 = self.gas.mix_diff_coeffs_mole[self.kN2]

        # Mass diffusion
        mole_diffusion = ((self.D0_N2 + D_N2) / 2.0 *
                          (self.f_N2v_0 - f_N2v) / self.r_d / self.r_d
                         ) * self.gas.concentrations[self.kN2]
        dfdt_N2v += mole_diffusion / self.gas.concentrations[self.kN2]

        # Energy loss due to mass diffusion
        for i, k in enumerate(self.N2v_indices):
            dTdt += (max(mole_diffusion[i], 0.0) * self.h0_mole[k] + min(mole_diffusion[i], 0.0) * self.gas_vib.partial_molar_enthalpies[k]) / (rho * self.gas.cp)

        dTdt += -np.dot(self.gas_vib.partial_molar_enthalpies, vib_wdot) / (rho * self.gas.cp)
        print("test")
        return np.hstack((dTdt, dYdt, dfdt_N2v))