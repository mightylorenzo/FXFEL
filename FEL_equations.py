# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 08:33:38 2016

@author: daniel bultrini
"""
import scipy.constants as const
import numpy as np
from scipy.special import j0, j1

alfven = 17000 #Amps

def undulator_parameter(magnetic_field, undulator_period): 
    """
    Returns the undulator paramter K,
    mag field in [T], undulator period in [m]
    """
    K_factor = const.e*magnetic_field*undulator_period/(2*const.pi*const.electron_mass*const.c)
    return K_factor


def coherence_length(wavelength, pierce):
    return float(wavelength)/(4*np.pi*np.sqrt(3)*pierce)

def optimal_slice_no(rms_length, coherence_length):
    return float(rms_length)/(2*np.pi*coherence_length)

def EM_charge_coupling(K_factor, m=1):
    """
    Returns the electromagnetic and charge coupling factor.
    J_0 is an even function and J_1 is odd, m is the harmonic in question
    """
    JJ = j0(m*K_factor*K_factor/(4+2*K_factor*K_factor))-j1(m*K_factor*K_factor/(4+2*K_factor*K_factor)) 
    return JJ
    
def resonant_electron_energy(
        beam_energy=False, undulator_period=False, wavelength=0, undulator_parameter=0):
    """
    if beam energy in SI and returns resonant gamma 
    if undulator period, seed wavelength and undulator parameter in SI and returns resonant gamma
    """
    if beam_energy:
        resonant_gamma = beam_energy/(const.m_e*const.c*const.c)
        return resonant_gamma
    elif undulator_period:
        resonant_gamma = np.sqrt((undulator_period/(2*wavelength))*
                                 (1+undulator_parameter*undulator_parameter/2))
        return resonant_gamma

def resonant_wavelength(undulator_period, K_factor, resonant_gamma_factor):
    """
    Returns the FEL's resonance [wavelength, wavenumber] in [m,m-1]
    """

    wavelength = (undulator_period/
        (2*resonant_gamma_factor*resonant_gamma_factor))*\
        (1+(K_factor*K_factor)/2)
    return [wavelength, 2*const.pi/wavelength]

def normalized_electron_energy(resonant_e_energy, electron_energy):
    if type(electron_energy) is (float or int):
        normalized = (electron_energy-resonant_e_energy)/resonant_e_energy
        return normalized
    else:
        normalized_list = []
        for i in electron_energy:
            normalized = (electron_energy-resonant_e_energy)/resonant_e_energy
            normalized_list.append(normalized)
        return normalized_list
        
#def gain_function(undulator_parameter,total_undulator_periods,undulator_period,
# electron_density,resonant_gamma,normalized_e_energy):
#    pass


def ming_xie_factor(nd, ne, ny):
    sp = np.array([0.45, 0.57, 0.55, 1.6, 3.0, 2.0,
          0.35, 2.9, 2.4, 51.0, 0.95, 3.0, 5.4,
          0.7, 1.9, 1140.0, 2.2, 2.9, 3.2])

    factor = (sp[0]*np.power(nd, sp[1])+ 
              sp[2]*np.power(ne, sp[3])+
              sp[4]*np.power(ny, sp[5])+
              sp[6]*np.power(ne, sp[7])*np.power(ny, sp[8])+
              sp[9]*np.power(nd, sp[10])*np.power(ny, sp[11])+
              sp[12]*np.power(nd, sp[13])*np.power(ne, sp[14])+
              sp[15]*np.power(nd, sp[16])*np.power(ne, sp[17])*np.power(ny, sp[18]))
    
    return factor

def scaled_e_spread(e_spread, gain_length, undulator_period):
    return np.array([4*const.pi*e_spread*gain_length/undulator_period])

def scaled_transverse_size(transverse_size, gain_length, wavelength):
    return np.array([gain_length*wavelength/(4*const.pi*transverse_size**2)])

def scaled_emittance(emittance, gain_length, wavelength,beta):
    return np.array([(4*const.pi*emittance*gain_length)/(beta*wavelength)])


def pierce(k_fact, gamma_res, undulator_period, current, std_x, std_y):
    '''Calculates Pierce Parameter for a slice,
    returns pierce and 1d gain_length'''

    K_JJ2 = (k_fact*EM_charge_coupling(k_fact))**2
    pierce_par = current/(alfven*gamma_res**3)
    pierce_par = pierce_par*(np.power(undulator_period, 2))/\
                (2*const.pi*std_x*std_y)
    pierce_par = (pierce_par*K_JJ2/(32*const.pi))**(1.0/3.0)
    gain_length = (undulator_period/(4*const.pi*np.sqrt(3.0)*pierce))

    return pierce, gain_length







