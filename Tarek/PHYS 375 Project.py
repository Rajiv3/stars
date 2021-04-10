import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc

######################################### Constants taken from scipy.constants ###################################

# all in SI units
G = spc.G
c = spc.c
hbar = spc.hbar
k = spc.k
mp = spc.proton_mass
me = spc.electron_mass
sigma_sb = spc.sigma

########################################### Definfing constants required #########################################

Msun = 1.989e30
a = 4*sigma_sb/c
gamma = 5.0/3.0 #The adiabatic index has a value γ = 5/3 for simple atomic gases and fully ionized gases.
'''
check: is gamma value correct?
'''
X, Y, Z = 0.734, 0.250, 0.016 #Values taken from Foundations of Astrophysics.
'''
check X, Y, Z values
'''
mu = 2*X + 0.75*Y + 0.5*Z


class star(object):
    def __init__(self, r0, rho_c, T_c, step_size):

        '''
        Given r0, rho_c, T_c calculate M0 and L0
        M0 - Initial Mass
        L0 - Initial Luminosity
        tau0 - Initial optical depth
        '''

        self.M0 = (4.0/3.0)*np.pi*r0**3*rho_c
        self.L0 = self.M0*self.epsilon(rho_c, T_c) 
        self.tau0 = self.kappa(rho_c,T_c)*rho_c
        self.r0 = r0
        self.rho_c = rho_c
        self.T_c = T_c
        self.step_size = step_size
        
        
################################################ Below are DEs required to solve ###################################

    def dpdr(self, r, rho, T, M, L):

        '''
        Hydrostatic equilibrium in rho
        '''

        return -(G*M*rho/r**2 + self.dPdT(rho, T)*self.dTdr(r, rho, T, M, L))/self.dPdp(rho, T)


    def dTdr(self, r, rho, T, M, L):

        '''
        Equation for energy transport. Minimum of energy transport
        due to convection and radiation
        '''

        dTdr_rad = 3*self.kappa(rho, T)*rho*L/(16*np.pi*a*c*T**3*r**2)
        dTdr_conv = (1 - 1/gamma)*T*G*M*rho/(self.P(rho, T)*r**2)

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


############################################# Functions to assist in solving DEs ###################################

    def epsilon(self, rho, T):
        '''
        Sum up energy generation due to proton-proton chain and CNO cycle
        '''

        rho5 = rho/1e5
        T6 = T/1e6
        Xcno = 0.03*X

        epp = 1.07e-7*rho5*X**2*T6**4
        ecno = 8.24e-26*rho5*X*Xcno*T6**(19.9)
        
        return epp + ecno



    def dPdT(self, rho, T):

        '''
        Differentiating Pressure equation(eq - 5) wrt T and
        considering rho a constant
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

################################################### Testing Radius Limit ######################################

    def deltau(self, r, rho, T, M, L):

        '''
        Opacity limit - to ensure that the radius is far enough
        '''

        return self.kappa(rho, T)*rho**2/abs(self.dpdr(r, rho, T, M, L))
    
    def tau_lim(self, r, rho, T, M, L):
        
        '''
        The runge kutta method stops when tau is satisfied
        '''
        
        deltau = self.deltau(r, rho, T, M, L)
        
        if deltau < 1e-5:
            return True
        
        '''
        introducing mass limit, since some fully radiative trial 
        solutions can erroneously extend to very large radii
        '''
        
        if M > 1e3*Msun:
            return True
        
        else:
            print('deltau', deltau)
            print('M', M)
            return False
    
    
################################################### Solving DEs ###################################################
    
    def runge_kutta(self, y0, r0, h, f):
    
        '''
        r0 = initial r value
        y0 = y(r0) = initial y value
        h = 'step size'
        f = 'function'
        '''
        
        #print(y0, r0)
    
        k1 = f(y0,r0)
        k2 = f(y0 + 0.5*h*k1, r0 + 0.5*h)
        k3 = f(y0 + 0.5*h*k2, r0 + 0.5*h)
        k4 = f(y0 + h*k3, r0 + h)

        return y0 + (k1 + 2*k2 + 2*k3 + k4)*h/6, r0+h
    
    def f(self, y0, r0):
        
        '''
        This function takes the y0, r0, values given in the runge kutta method
        and calculates the dpde, dTdr, dMdr, dLdr, dtaudr values.
        '''
        
        rho = y0[0]
        T = y0[1]
        M = y0[2]
        L = y0[3]
        tau = y0[4]
        
        rho_fun = self.dpdr(r0, rho, T, M, L)
        T_fun = self.dTdr(r0, rho, T, M, L)
        M_fun = self.dMdr(r0, rho)
        L_fun = self.dLdr(r0, rho, T)
        tau_fun = self.dtaudr(rho, T)
        
        #print('function array', np.array([rho_fun, T_fun, M_fun, L_fun, tau_fun]))
        
        return np.array([rho_fun, T_fun, M_fun, L_fun, tau_fun])
        
    def solve_DE(self):
        
        rho_vals, T_vals, M_vals, L_vals, tau_vals, r_vals = [], [], [], [], [], []
        
        r0 = self.r0
        init_vals = np.array([self.rho_c, self.T_c, self.M0, self.L0, self.tau0])
        
        func_vals, r = self.runge_kutta(init_vals, r0, self.step_size, self.f)
        
        #updating r0 and init values
        r0 = r
        init_vals = func_vals
        
        r_vals.append(r)
        rho_vals.append(func_vals[0])
        T_vals.append(func_vals[1])
        M_vals.append(func_vals[2])
        L_vals.append(func_vals[3])
        tau_vals.append(func_vals[4])
        
        '''
        Unless the tau limit criteria is met the loop runs
        '''
    
        while star_func.tau_lim(r, func_vals[0], func_vals[1], func_vals[2], func_vals[3]) == False: 
            
            func_vals, r = self.runge_kutta(init_vals, r0, self.step_size, self.f)
            
            r_vals.append(r)
            rho_vals.append(func_vals[0])
            T_vals.append(func_vals[1])
            M_vals.append(func_vals[2])
            L_vals.append(func_vals[3])
            tau_vals.append(func_vals[4])
    
            #updating r0 and init values
            init_vals = func_vals
            r0 = r
        
        # added r values list to output
        return rho_vals, T_vals, M_vals, L_vals, tau_vals, r_vals
    
star_func = star(1e-50, 1000, 1000, 1e4)
rho, T, M, L, tau, r  = star_func.solve_DE()

tau_Rstar = tau[len(tau)-1] - (2/3)

print("")

# The program looks for tau_Rstar or a value close to it. Since the stepsizes are equal,
# the program will favour one of the values and then take it to be Rstar. Thus, this algorithm is accurate to a stepsize.
# For the right stepsize, it can result in a negligible difference.
def search(lis, element):
    for i in range(len(lis)):
        if lis[i] == element or (lis[i] < element +(5e-2) and lis[i]> element -(5e-2)): # Bounds of acceptable discrepancy will depend on the inital conditions
#            print(i) # optional. It can be helpful in understanding how the search function works
            return i

search(tau, tau_Rstar) 

print(r[int(search(tau, tau_Rstar))])
print(L[int(search(tau, tau_Rstar))])
print(T[int(search(tau, tau_Rstar))])


