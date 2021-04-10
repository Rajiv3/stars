import numpy as np
from numpy import pi
import scipy.constants as spc

G = spc.G
c = spc.c
hbar = spc.hbar
k = spc.k
mp = spc.proton_mass
me = spc.electron_mass
sigma_sb = spc.sigma
a = 4*sigma_sb/c

M_sun = 1.98847e30 
L_sun = 3.828e26 
R_sun = 696340000 

gamma = 5 / 3

rho_index, T_index, M_index, L_index, tau_index = 0,1,2,3,4

X_default = 0.734
Z_default = 0.016
Y_default = 0.025
Lambda_default = 0



class Star:
    def __init__(self, X=X_default, Y=Y_default, Z=Z_default, Lambda=Lambda_default):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Lambda = Lambda

    
    def __str__(self):
        return 'Star (X=',str(self.X),'Y=',str(self.Y),'Z=',str(self.Z),'lambda',str(self.Lambda/ R_sun)

    def get_initial_conditions(self, rho_c, T_c, r_0=1):
        
        M_c = (4 / 3) * pi * r_0 ** 3 * rho_c
        L_c = M_c * self.epsilon(rho_c, T_c)
        kappa_c = self.kappa(rho_c, T_c)
        tau_c = kappa_c * rho_c * r_0
        return np.array([rho_c, T_c, M_c, L_c, tau_c])

    def get_state_derivative(self, r, state, return_kappa=False):
        """
        Calculates the derivative of each element of state
        state: rho, T, M,L at the given radius
        return_kappa: If True, then the opacity will be returned as the second item of a tuple.
        
        Returns the derivative of the state vector, and the optical depth if True
        """
        rho, T, M, L, _ = state
        kappa_value = self.kappa(rho, T)

        T_prime_value = self.dTdr (r, rho, T, M, L, kappa_value=kappa_value)
        rho_prime_value = self.rpdr (r, rho, T, M, L, T_prime_value=T_prime_value)
        M_prime_value = self.dMdr (r, rho)
        L_prime_value = self.dLdr (r, rho, T, M_prime_value=M_prime_value)
        tau_prime_value = self.dtaudr (rho, T, kappa_value=kappa_value)

        state_derivative = np.array([rho_prime_value, T_prime_value, M_prime_value, L_prime_value, tau_prime_value])
        return (state_derivative, kappa_value) if return_kappa else (state_derivative)

    def dpdr (self, r, rho, T, M, L):
        return -(G*M*rho/r**2 * (1 + self.Lambda/ r)  +  self.dPdT(rho, T) * T_prime_value ) / self.dPdp(rho, T)

    def dTdr (self, r, rho, T, M, L, kappa_value=None):
        if kappa_value is None:
            kappa_value = self.kappa(rho, T)
        
        rad = self.radiative(r, rho, T, L, kappa_value=kappa_value)
        con = self.convective(r, rho, T, M)
        
        return np.maximum(rad, con)

    def radiative(self, r, rho, T, L, kappa_value=None):
        """
        kappa_value could be given, if not, kappa will be calculated with (rho T)
        """
        
        if kappa_value is None:
            kappa_value = self.kappa(rho, T)
        return -3 * kappa_value * rho * L / (16 * pi * a * c * T ** 3 * r ** 2)

    def convective(self, r, rho, T, M):
        return -(1 - 1 / gamma) * T * G * M * rho / (self.P(rho, T) * r ** 2) * (1 + self.Lambda / r) 

    def is_convective(self, r, rho, T, M, L, kappa_value=None):
        """
        returns True if the star is convection
        """
        
        if kappa_value is None:
            kappa_value = self.kappa(rho, T)
        return self.convective(r, rho, T, M) > self.radiative(r, rho, T, L, kappa_value=kappa_value)

    def dMdr (r, rho):
        return 4 * pi * r ** 2 * rho

    def dLdr (self, r, rho, T, M_prime_value=None):
        if M_prime_value is None:
            M_prime_value = self.M_prime(r, rho)
        return M_prime_value * self.epsilon(rho, T)

    def L_pp_prime(self, r, rho, T, M_prime_value=None):
        if M_prime_value is None:
            M_prime_value = self.M_prime(r, rho)
        return M_prime_value * self.epsilon_pp(rho, T)

    def L_CNO_prime(self, r, rho, T, M_prime_value=None):
        if M_prime_value is None:
            M_prime_value = self.M_prime(r, rho)
        return M_prime_value * self.epsilon_CNO(rho, T)

    def dtaudr (self, rho, T, kappa_value=None):
        if kappa_value is None:
            kappa_value = self.kappa(rho, T)
        return kappa_value * rho

    def P(self, rho, T):
        return self.P_degeneracy(rho) + self.P_gas(rho, T) + self.P_photon(T)

    def P_degeneracy(rho):
        return (3 * pi ** 2) ** (2 / 3) * h_bar ** 2 / (5 * m_e) * (rho / m_p) ** (5 / 3)

    def P_gas(self, rho, T):
        return rho * k_b * T / (self.mu() * m_p)

    def P_photon(T):
        return (1 / 3) * a * T ** 4

    def dPdp (self, rho, T):
        return (3 * pi ** 2) ** (2 / 3) * h_bar ** 2 / (3 * m_e * m_p) * (rho / m_p) ** (2 / 3) + \
               k_b * T / (self.mu() * m_p)

    def dPdT (self, rho, T):
        return rho * k / (self.mu() * m_p) + (4 / 3) * a * T ** 3

    def mu(self):
        return (2 * self.X + 0.75 * self.Y + 0.5 * self.Z) ** -1

    def kappa(self, rho, T):
        return (1 / self.KH(rho, T) + 1 / max (self.Kes(), self.Kff(rho,T))) ** -1

    def Kes(self):
        return 0.02 * (self.X + 1)

    def Kff(self, rho, T):
        return 1e24 * (self.Z + 0.0001) * (rho / 10 ** 3) ** 0.7 * T ** -3.5

    def KH(self, rho, T):
        return 2.5e32 * (self.Z / 0.02) * (rho / 10 ** 3) ** 0.5 * T ** 9

    def epsilon(self, rho, T):
        return self.epsilon_pp(rho, T) + self.epsilon_CNO(rho, T)

    def epsilon_pp(self, rho, T):
        return 1.07e-7  * self.X ** 2 * (rho / 10 ** 5) * (T / 10 ** 6) ** 4

    def epsilon_CNO(self, rho, T, X_CNO=None):
        if X_CNO is None:
            X_CNO = 0.03 * self.X
        return 8.24e-26 * self.X * X_CNO * (rho / 10 ** 5) * (T / 10 ** 6) ** 19.9


    def has_gravity_modifications(self):
        return self.Lambda != Lambda_default
    
    def describe_gravity_modifications(self):
        description = ''
        if self.Lambda!= Lambda_default:
            description += r'$\lambda$=' + ('%.2f' % (self.Lambda/ R_sun)) + r'$R_{\odot}$'
        return description

###################################################################################
    
