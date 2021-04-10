from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import numpy as np

#put rho,T,M,L,tau values into an array#

rho_index, T_index, M_index, L_index, tau_index = 0,1,2,3,4

def find_zeros_index(x, round_int=False, find_first=True):
    """
    Find the index of zeros in an array
    """
    return np.where(x == 0)[0]

def interpolate(x, index):
    """
    Linearly interpolates data at floating point indices within the given array or 2D matrix.
    :param x: The array or 2D matrix to interpolate from.
              If x is a matrix, then interpolation is done along the second axis.
    :param index: The (potentially fractional) index to interpolate at.
    :return: The interpolated value or (in the case where x is a matrix) array of values.
    """
    if len(x.shape) == 1:
        # if x is an array
        return np.interp(index, np.arange(len(x)), x)
    else:
        # if x is a 2D matrix
        return interp1d(np.arange(x.shape[1]), x)(index)
    
######################################2.2.2#########################################

def get_remaining_optical_depth(r, rho, T, M, L, kappa, rho_prime, Star=Star()):
    """
    Calculate the remaining optical depth from the current r to infinity.
    If small enough, we reached the centre of the star, integration is done 
    r,rho,T,M,L: current info 
    kappa_value: The current optical depth.
    rho_prime_value: The current derivative of density (with respect to radius).
    """
    return kappa * rho ** 2 / np.abs(rho_prime)


def truncate_star(r_values, state_values, return_star=False):
    """
    Calculates difference between the actual surface luminosity and the surface luminosity (part 2.2.2)
    The surface radius is interpolated such that the remaining optical depth from the surface is 2/3.
    Can optionally truncate the given stellar data at the surface of the star and add a final data point for the
    surface of the star, where the temperature is manually set to satisfy the boundary condition.
     r_values: array of radii
     state_values: array containing Rho, T, M, L conrresponding to r_values
    :param return_star:
    :return: The fractional surface luminosity error, and optionally the truncated r_values and state_values.
    """
    tau_infinity = state_values[tau_index, -1]
    surface_index = find_zeros_index(tau_infinity - state_values[tau_index, :] - 2 / 3)

    surface_r = interpolate(r_values, surface_index)
    surface_state = interpolate(state_values, surface_index)

    # calculate the fractional surface luminosity error
    expected_surface_L = 4 * pi * surface_r ** 2 * sigma_sb * surface_state[T_index] ** 4
    error = (surface_state[L_index] - expected_surface_L) / np.sqrt(surface_state[L_index] * expected_surface_L)
    if not return_star:
        return error

    # manually set surface temperature to satisfy boundary condition
    print('Old T:', surface_state[T_index])
    surface_state[T_index] = (surface_state[L_index] / (4 * pi * surface_r ** 2 * sigma)) ** (1 / 4)
    print('New T:', surface_state[T_index])

    # truncate the star at the surface, and append the manually corrected surface state
    surface_index = int(surface_index)
    r_values = np.append(r_values[:surface_index], surface_r)
    state_values = np.column_stack((state_values[:, :surface_index], surface_state))

    return error, r_values, state_values


def trial_solution(rho_c, T_c, r_0=100 , rtol=1e-9, atol=None,
                   return_star=False, optical_depth_threshold=1e-4, mass_threshold=1000 * M_sun,
                   Star=Star()):
    """
    Integrate the state of the star from r_0 until the estimated optical depth
    Return the array of radius values and the state matrix, with surface luminosity error
     rho_c: The central density.
     T_c: The central temperature.
     r_0: The starting value of the radius. Defaults to 100m. 0 does not work
     
     return_star: If True, then the radius values and state matrix will be returned alongside
                        the surface luminosity error
                        
     rtol: The required relative accuracy during integration.
     atol: The required absolute accuracy during integration. Defaults to rtol / 1000
    
    optical_depth_threshold: The value below which the estimated remaining optical depth
                             must drop for the integration to terminate
                             
    mass_threshold: If the mass of the star is beyond this value, then integration will be halted.
                           Defaults to 1000 solar masses.
                           
    :returns: The fractional surface luminosity error, and optionally the array of radius values and the state matrix
    """
    print(rho_c)
    if atol is None:
        atol = rtol / 1000

    def halt_integration(r, state):
        if state[M_index] > mass_threshold:
            return -1
        return get_remaining_optical_depth(r, state, config=config) - optical_depth_threshold

    halt_integration.terminal = True

    # Ending radius is infinity, integration will only be halted via the halt_integration event
    # Not sure what good values for atol and rtol are, but these seem to work well
    result = solve_ivp(Star.get_state_derivative, (r_0, np.inf),
                                 Star.get_initial_conditions(rho_c, T_c, r_0=r_0),
                                 events=halt_integration, atol=atol, rtol=rtol)
   
    r_values, state_values = result.t, result.y

    return truncate_star(r_values, state_values, return_star=return_star)
