import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc

'''
Definfining constants here. Taken from scipy.constants
'''

G = spc.G
c = spc.c
hbar = spc.hbar
k = spc.k
mp = spc.proton_mass

mu = 2*X + 0.75*Y + 0.5*Z
'''
To Do: Need to figure out what X, Y and Z are.
'''

'''
A note in general -  I have not defined some constants here.
'''

class star(object):
    def __init__(self, r0, rho_c, T_c):

        '''
        Given r0, rho_c, T_c calculate M0 and L0
        M0 - Initial Mass
        L0 - Initial Luminosity
        '''
        M0 = (4.0/3.0)*np.pi*r0**3*rho_c

        L0 = M0*self.epsilon(rho_c, T_c)


    def dpdr(self, r, rho, T, M, L):

        '''
        Hydrostatic equilibrium in rho
        '''

        return -(G*M*rho/r**2 + self.dPdT(rho, T)*self.dTdr(r, rho, T, M, L))/self.dPdp(rho, T)

    '''
    Below are DE required to solve
    '''

    def dTdr(self, r, rho, T, M, L):

        '''
        Equation for energy transport. Minimum of energy transport
        due to convection and radiation
        '''

        dTdr_rad = 3*self.kappa(rho, T)*L/(16*np.pi*a*c*T**3*r**2)
        dTdr_conv = (1 - 1/gamma)*T*G*M*rho/(self.P(rho, T)*r**2)

        '''
        To Do: Define gamma
        To Do: check if c is the speed of light.
        '''

        return -min(dTdr_rad , dTdr_conv)


    def dMdr(self, r, rho):

        '''
        definition of enclosed mass
        '''

        return 4*np.pi*r**2*rho

    def dLdr(self, r, rho, T):

        '''
        Energy generation equation
        '''

        return 4*np.pi*r**2*rho*self.epsilon(rho, T)

    def dtaudr(self, rho, T):

        '''
        Optical Depth
        '''

        return self.kappa(rho, T)*rho


    '''
    Functions to assist in solving above DEs
    '''

    def epsilon(self, rho, T):
        '''
        Sum up energy generation due to proton-proton chain and CNO cycle
        '''

        rho5 = rho/1e5
        T6 = T/1e6
        Xcno = 0.03*X

        epp = 1.07e-7*rho5*X**2*T6**4
        ecno = 8.24e-26*rho4*X*Xcno*T6**(19.9)



    def dPdT(self, rho, T):

        '''
        Differentiating Pressure equation(eq - 5) wrt T and
        considering rho a constant
        '''

        '''
        To Do: Define a
        '''

        dPdT_ig = rho*k/(mu*mp)
        dPdT_pg = (4.0/3.0)*a*T**3

        return  dPdT_ig + dPdT_pg



    def kappa(self, rho, T):

        '''
        Rosseland Mean Opacities.
        '''
        rho3 = rho/1e3

        Kes = 0.02*(1+X)
        Kff = 1e24*(Z+0.0001)*rho3**(0.7)*T**(-3.5)
        KH = 2.5e-32*(Z/0.02)*rho3**(0.5)*T**9

        '''
        Check: below I return eq 14, but the comments below the equation 14 in the
        project description say "near and below the net opacity is the minimum of that due to free-free/scattering
        and that due to H". I am not entirely sure if I need do put a conditionality here.
        '''

        return ((1/KH) + (1/max(Kes, Kff)))**(-1)

    def P(self, rho, T):

        '''
        Pressure. Sum of non relativistic degenerate gas,
        ideal gas and photon gas
        '''

        P_nr = (((3*np.pi**2)**(2.0/3.0))*(hbar**(2))*((rho/mp)**(5/3)))/(5*me)
        P_ig = rho*k*T/(mu*mp)
        P_pg = (1.0/3.0)*a*T**4

        return P_nr + P_ig + P_pg

    def dPdp(self, rho, T):

        '''
        Differentiating Pressure equation(eq - 5) wrt rho(p) and
        considering T a constant
        '''

        dPdp_nr = (((3*np.pi**2)**(2.0/3.0))*(hbar**2)*((rho/mp)**(2.0/3.0)))/(3.0*mp*me)
        dPdp_ig = k*T/(mu*mp)

        return dPdp_nr + dPdp_ig

    '''
    Test radius limit
    '''

    def deltau(self, r, rho, T, M, L):

        '''
        Opacity limit - to ensure that the radius is far enough
        '''

        '''
        Check: Would this need a conditionality statement here
        '''

        return kappa*rho**2/abs(self.dpdr())
