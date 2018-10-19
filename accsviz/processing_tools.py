# -*- coding: utf-8 -*-
"""
# Copyright (c) 2012-2017, University of Strathclyde
# Authors: Daniel Bultrini
# License: BSD-3-Clause
"""

import numpy as np
import pandas as pd
import sys
import tables
import warnings
import FEL_equations as feq


c = float(3.0e+8)                    # Speed of light
m = float(9.11e-31)                  # mass of electron
E_CH = float(1.602e-19)
P = np.pi

def fname_format(filename, addition):
    return filename+'_'+addition

def beta(std_pos, emittance):
    ''' Calculates a simple approximation of the Beta
    function for a slice given standard deviation of position and
    slice emittance'''
    #calc = (np.sqrt(4*np.log(2))*std_pos)
    #calc = np.power(calc , 2)
    calc = np.power(std_pos , 2)
    calc = np.divide(calc, emittance)

    return calc

#def talpha(xpx_cor, emittance):
def talpha(xpx_cor, beta):
    ''' Calculates a simple approximation of the alpha
    function for a slice given standard deviation of position and
    slice emittance'''

    #calc = - (4.*np.log(2.) * xpx_cor) / emittance
    calc = -1. * xpx_cor * beta

#    calc = (np.sqrt(4*np.log(2))*std_pos)
#    calc = np.power(calc , 2)
#    calc = np.divide(calc, emittance)

    return calc

def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return np.sqrt(variance)

class ParticleDistribution(object):
    '''Reads and provides processing to SU particle distributions

    initializes with a filename and the data,
    which then can be processed by in built methods'''

    def __init__(self, filename):
        self.filename = filename
        self.axis_labels = {'x':'X position ', 'px':'X momentum ', 'y':'Y position ',
                            'py':'Y momentum ', 'z':'Z position ', 'pz':'Z momentum', 'NE':'Weight',
                            'e_y': 'Y emittance ', 'e_x':'X emittance ', 'slice_z':'Z position ',
                            'mean_x' : 'mean x position ', 'mean_y' : 'mean y position '}

        with tables.open_file(filename, 'r') as F:
            self.SU_data = F.root.Particles.read()

        self.dict = {'x':self.SU_data[:, 0], 'px':self.SU_data[:, 1],
                     'y':self.SU_data[:, 2], 'py':self.SU_data[:, 3],
                     'z':self.SU_data[:, 4], 'pz':self.SU_data[:, 5],
                     'NE':self.SU_data[:, 6]}

        self.SI = False

    def su2si(self):
        '''Converts data to SI if needed'''
        if not self.SI:
            #print 'av px b4 = ', np.average(self.SU_data[:, 1])
            #print 'av py b4 = ', np.average(self.SU_data[:, 3])
            #print 'av pz b4 = ', np.average(self.SU_data[:, 5])
            self.SU_data[:, 1] = self.SU_data[:, 1]#*(m*c) #/E_CH
            self.SU_data[:, 3] = self.SU_data[:, 3]#*(m*c) #/E_CH
            self.SU_data[:, 5] = self.SU_data[:, 5]#*(m*c) #/E_CH
            self.SI = True
            #print 'av px after = ', np.average(self.SU_data[:, 1])
            #print 'av py after = ', np.average(self.SU_data[:, 3])
            #print 'av pz after = ', np.average(self.SU_data[:, 5])
        else:
            print('Already converted')
            pass
    def optimal_slice(self, undulator_period, k_fact):
        '''Attempts to give optimal slice number, but may be an incorrect implementation'''
        p_tot=np.sqrt((self.SU_data[:, 1]**2)+(self.SU_data[:, 3]**2)+(self.SU_data[:, 5]**2)) #* E_CH**2
        #gamma=(np.sqrt(1+(p_tot/(m*c))**2))
        gamma=(np.sqrt(1+(p_tot)**2))
        res_wavelength = feq.resonant_wavelength(undulator_period,k_fact,np.average(gamma))[0]
        #print 'lambda_r = ' + str(res_wavelength)
        #print 'gamma0 = ' + str(np.average(gamma))
        #print 'ave pz = ' + str(np.average(self.dict['pz']) / (m*c**2) )
        length_std = weighted_std(self.SU_data[:, 4],self.SU_data[:, 6])
        std_x = weighted_std(self.dict['x'],self.SU_data[:, 6])
        std_y = weighted_std(self.dict['y'],self.SU_data[:, 6])
        avg_current = (np.sum(self.SU_data[:, 6])*E_CH)*c/(length_std)
        #print 'avg_current = ' + str(avg_current)
        #print 'length_std = ' + str(length_std)
        avg_pierce = feq.pierce(k_fact,np.average(gamma),undulator_period,avg_current,std_x,std_y)[0]
        coherence_length = feq.coherence_length(res_wavelength,avg_pierce)
        num_slices = feq.optimal_slice_no(length_std,coherence_length)
        #num_slices = 100
        #print 'num slices = ' + str(num_slices)
        return np.int_(num_slices)

    def DistFrame(self):
        dist_dir = ('x', 'pz', 'y', 'x', 'py', 'px', 'z', 'NE')
        dist_dir = {k: self.dict[k] for k in dist_dir}
        return pd.DataFrame(dist_dir)

    def bin(self,reduction_factor):
        pass


class Statistics(ParticleDistribution):
    '''Class to calculate and contain statistical processing 
    of given particle distribution'''
    def __init__(self,filename):

        super(Statistics,self).__init__(filename)
        self.su2si()

    def slice(self, Num_Slices):
        '''set data slicing for use in certain routines if needed,
        returns 'self.dict['z_pos']' as array with z positions and self.dict['slice_keys']
        with boolean arrays for use in operations'''
# This makes slices of equal length, but if another method is required, the rest of the code requires that 
# slice keys be a boolean array of length equal to the particle distribution with true for the particles 
#wanted in a given slice
        self.dict['Num_Slices'] = Num_Slices
        self.dict['Step_Z'] = (np.max(self.SU_data[:, 4])-np.min(self.SU_data[:, 4]))/\
                                    self.dict['Num_Slices']
        self.dict['slice_keys'] = np.ones(
            (self.dict['Num_Slices'], self.SU_data.shape[0]), dtype=bool)
        self.dict['z_pos'] = np.zeros(Num_Slices)

        for slice_no in xrange(self.dict['Num_Slices']):
            z_low = np.min(self.SU_data[:, 4])+(slice_no*self.dict['Step_Z'])
            z_high = np.min(self.SU_data[:, 4])+((slice_no+1)*self.dict['Step_Z'])
            z_pos = (z_high+z_low)/2.0  #routine that calculates the z position at the center of the slice
            self.dict['z_pos'][slice_no] = z_pos
            self.dict['slice_keys'][slice_no] = np.array((self.SU_data[:, 4] >= z_low) &\
                                                     (self.SU_data[:, 4] < z_high), dtype=bool)

        self.dict.update({'slice_z': self.dict['z_pos']})

    def calc_emittance(self):
        '''Returns the emittance per slice as arrays self.dict['e_x'], self.dict['e_y']
        directories 'e_x' 'and e_y'''

        self.dict['e_x'], self.dict['e_y'] = \
            np.empty(self.dict['Num_Slices']), np.empty(self.dict['Num_Slices'])

        for i, comparison in enumerate(self.dict['slice_keys']):
            m_POSx = self.SU_data[:, 0][comparison]
            m_mOmx = self.SU_data[:, 1][comparison] / self.SU_data[:, 5][comparison]
            m_POSy = self.SU_data[:, 2][comparison]
            m_mOmy = self.SU_data[:, 3][comparison] / self.SU_data[:, 5][comparison]
            wts = self.SU_data[:, 6][comparison]
            
            ###########~~Code by Piotr Traczykowski~~###############################################
            x_2 = ((np.sum(wts*m_POSx*m_POSx))/np.sum(wts))-(np.average(m_POSx, weights=wts))**2.0                     #
            px_2 = ((np.sum(wts*m_mOmx*m_mOmx))/np.sum(wts))-(np.average(m_mOmx, weights=wts))**2.0                    #
            xpx = np.sum(wts*m_POSx*m_mOmx)/np.sum(wts)-np.sum(wts*m_POSx)*np.sum(wts*m_mOmx)/(np.sum(wts))**2 #
                                                                                                   #
            y_2 = ((np.sum(wts*m_POSy*m_POSy))/np.sum(wts))-(np.average(m_POSy, weights=wts))**2.0                     #
            py_2 = ((np.sum(wts*m_mOmy*m_mOmy))/np.sum(wts))-(np.average(m_mOmy, weights=wts))**2.0                    #
            ypy = np.sum(wts*m_POSy*m_mOmy)/np.sum(wts)-np.sum(wts*m_POSy)*np.sum(wts*m_mOmy)/(np.sum(wts))**2 #
                                                                                                   #
            #self.dict['e_x'][i] = (1.0/(m*c))*np.sqrt((x_2*px_2)-(xpx*xpx))                        #
            #self.dict['e_y'][i] = (1.0/(m*c))*np.sqrt((y_2*py_2)-(ypy*ypy))                        #
            self.dict['e_x'][i] = np.sqrt((x_2*px_2)-(xpx*xpx))                        #
            self.dict['e_y'][i] = np.sqrt((y_2*py_2)-(ypy*ypy))
            ########################################################################################

    def calc_emittanceG(self):
        '''Returns the global (integrated) emittance'''

        m_POSx = self.SU_data[:, 0]
        m_mOmx = self.SU_data[:, 1] / self.SU_data[:, 5]
        m_POSy = self.SU_data[:, 2]
        m_mOmy = self.SU_data[:, 3] / self.SU_data[:, 5]
        wts = self.SU_data[:, 6]


        x_2 = ((np.sum(wts*m_POSx*m_POSx))/np.sum(wts))-(np.average(m_POSx, weights=wts))**2.0                     #
        px_2 = ((np.sum(wts*m_mOmx*m_mOmx))/np.sum(wts))-(np.average(m_mOmx, weights=wts))**2.0                    #
        xpx = np.sum(wts*m_POSx*m_mOmx)/np.sum(wts)-np.sum(wts*m_POSx)*np.sum(wts*m_mOmx)/(np.sum(wts))**2 #
                                                                                                   #
        y_2 = ((np.sum(wts*m_POSy*m_POSy))/np.sum(wts))-(np.average(m_POSy, weights=wts))**2.0                     #
        py_2 = ((np.sum(wts*m_mOmy*m_mOmy))/np.sum(wts))-(np.average(m_mOmy, weights=wts))**2.0                    #
        ypy = np.sum(wts*m_POSy*m_mOmy)/np.sum(wts)-np.sum(wts*m_POSy)*np.sum(wts*m_mOmy)/(np.sum(wts))**2 #

        exf = np.sqrt((x_2*px_2)-(xpx*xpx))
        eyf = np.sqrt((y_2*py_2)-(ypy*ypy))

        return exf, eyf

    def calc_xpxypy(self):
        '''returns x-xp and y-yp correlations'''

        xpx, ypy = \
            np.empty(self.dict['Num_Slices']), np.empty(self.dict['Num_Slices'])

        for i, comparison in enumerate(self.dict['slice_keys']):
            m_POSx = self.SU_data[:, 0][comparison]
            m_mOmx = self.SU_data[:, 1][comparison] / self.SU_data[:, 5][comparison]
            m_POSy = self.SU_data[:, 2][comparison]
            m_mOmy = self.SU_data[:, 3][comparison] / self.SU_data[:, 5][comparison]
            wts = self.SU_data[:, 6][comparison]

            x_2 = ((np.sum(wts*m_POSx*m_POSx))/np.sum(wts))-(np.average(m_POSx, weights=wts))**2.0
            y_2 = ((np.sum(wts*m_POSy*m_POSy))/np.sum(wts))-(np.average(m_POSy, weights=wts))**2.0

            xpx[i] = np.sum(wts*m_POSx*m_mOmx)/np.sum(wts)-np.sum(wts*m_POSx)*np.sum(wts*m_mOmx)/(np.sum(wts))**2
            ypy[i] = np.sum(wts*m_POSy*m_mOmy)/np.sum(wts)-np.sum(wts*m_POSy)*np.sum(wts*m_mOmy)/(np.sum(wts))**2

            xpx[i] = xpx[i] / x_2
            ypy[i] = ypy[i] / y_2

        return xpx, ypy


    def calc_xpxypyG(self):
        '''returns the global x-xp and y-yp correlations'''

        m_POSx = self.SU_data[:, 0]
        m_mOmx = self.SU_data[:, 1] / self.SU_data[:, 5]
        m_POSy = self.SU_data[:, 2]
        m_mOmy = self.SU_data[:, 3] / self.SU_data[:, 5]
        wts = self.SU_data[:, 6]

        x_2 = ((np.sum(wts*m_POSx*m_POSx))/np.sum(wts))-(np.average(m_POSx, weights=wts))**2.0
        y_2 = ((np.sum(wts*m_POSy*m_POSy))/np.sum(wts))-(np.average(m_POSy, weights=wts))**2.0

        xpx = np.sum(wts*m_POSx*m_mOmx)/np.sum(wts)-np.sum(wts*m_POSx)*np.sum(wts*m_mOmx)/(np.sum(wts))**2
        ypy = np.sum(wts*m_POSy*m_mOmy)/np.sum(wts)-np.sum(wts*m_POSy)*np.sum(wts*m_mOmy)/(np.sum(wts))**2

        xpx = xpx / x_2
        ypy = ypy / y_2

        return xpx, ypy


    def calc_CoM(self):
        '''Returns the weighted average positions in
            x and y as self.mean_x, self.mean_y'''

        if not self.dict.__contains__('e_x'):
            warnings.warn('Need to run emittance first, calculating...')
            self.calc_emittance()


        # allocate arrays of appropiate size in memory
        self.dict['CoM_x'], self.dict['CoM_y'] = \
            np.empty(self.dict['Num_Slices']), np.empty(self.dict['Num_Slices'])
        self.dict['CoM_px'], self.dict['CoM_py'] = \
            np.empty(self.dict['Num_Slices']), np.empty(self.dict['Num_Slices'])
        self.dict['CoM_pz'], self.dict['std_pz'] = \
            np.empty(self.dict['Num_Slices']), np.empty(self.dict['Num_Slices'])
        self.dict['std_x'], self.dict['std_y'] = \
            np.empty(self.dict['Num_Slices']), np.empty(self.dict['Num_Slices'])
        self.dict['CoM_z'], self.dict['std_z'] = \
            np.empty(self.dict['Num_Slices']), np.empty(self.dict['Num_Slices'])
        self.dict['std_px'], self.dict['std_py'] = \
            np.empty(self.dict['Num_Slices']), np.empty(self.dict['Num_Slices'])


        #Loop through slices and apply weighed standard deviation and average (CoM)
        for i, comparison in enumerate(self.dict['slice_keys']):
            weight = self.SU_data[:, 6][comparison]

            self.dict['CoM_x'][i] = np.average(self.SU_data[:, 0][comparison], weights=weight)
            self.dict['CoM_y'][i] = np.average(self.SU_data[:, 2][comparison], weights=weight)
            self.dict['CoM_z'][i] = np.average(self.SU_data[:, 4][comparison], weights=weight)
            self.dict['CoM_px'][i] = np.average(self.SU_data[:, 1][comparison], weights=weight)
            self.dict['CoM_py'][i] = np.average(self.SU_data[:, 3][comparison], weights=weight)
            self.dict['CoM_pz'][i] = np.average(self.SU_data[:, 5][comparison], weights=weight)
            self.dict['std_pz'][i] = weighted_std(self.SU_data[:, 5][comparison], weight)
            self.dict['std_px'][i] = weighted_std(self.SU_data[:, 1][comparison], weight)
            self.dict['std_py'][i] = weighted_std(self.SU_data[:, 3][comparison], weight)
            self.dict['std_x'][i] = weighted_std(self.SU_data[:, 0][comparison], weight)
            self.dict['std_y'][i] = weighted_std(self.SU_data[:, 2][comparison], weight)
            self.dict['std_z'][i] = weighted_std(self.SU_data[:, 4][comparison], weight)

        self.dict['beta_x'] = beta(self.dict['std_x'], self.dict['e_x'])
        self.dict['beta_y'] = beta(self.dict['std_y'], self.dict['e_y'])

        xpx, ypy = self.calc_xpxypy()

        #self.dict['alpha_x'] = talpha(xpx, self.dict['e_x'])
        #self.dict['alpha_y'] = talpha(ypy, self.dict['e_y'])

        self.dict['alpha_x'] = talpha(xpx, self.dict['beta_x'])
        self.dict['alpha_y'] = talpha(ypy, self.dict['beta_y'])

        #    TEMP CALCULATE GLOBAL TWISS

        exG, eyG = self.calc_emittanceG()
        sdxG = weighted_std(self.SU_data[:, 0], self.SU_data[:, 6])
        sdyG = weighted_std(self.SU_data[:, 2], self.SU_data[:, 6])
        bxG = beta(sdxG, exG)
        byG = beta(sdyG, eyG)

        xpxG, ypyG = self.calc_xpxypyG()

        #self.dict['alpha_x'] = talpha(xpx, self.dict['e_x'])
        #self.dict['alpha_y'] = talpha(ypy, self.dict['e_y'])

        axG = talpha(xpxG, bxG)
        ayG = talpha(ypyG, byG)

        print 'Twiss in X are = ', bxG, axG
        print 'Twiss in Y are = ', byG, ayG


        self.axis_labels.update({'CoM_x': 'CoM X position',
                                 'CoM_y': 'CoM Y position',
                                 'CoM_px': 'CoM X momentum',
                                 'CoM_py': 'CoM Y momentum',
                                 'CoM_pz': 'CoM Z momentum',
                                 'std_pz': 'STD of Z momentum',
                                 'std_x': 'STD of x position',
                                 'std_y': 'STD of y position',
                                 'beta_x': 'beta x',
                                 'beta_y': 'beta y',
                                 'alpha_x': 'alpha x',
                                 'alpha_y': 'alpha y',
                                 'std_px': 'STD of x position',
                                 'std_py': 'STD of y position'})

    def calc_current(self):
        '''Calculates current per slice and returns array - uses approximation
        current = total charge per slice * speed of light '''

        if not self.SI:
            warnings.warn('might get strange results without SI conversion')
            n = str(raw_input('Convert? [y/n]:'))
            if n == 'y':
                self.su2si()
            else:
                print('Might not have selected right option, no conversion done.')

        self.dict['current'] = np.empty(self.dict['Num_Slices'])
        bin_length = self.dict['z_pos'][1]-self.dict['z_pos'][0]

        for i, comparison in enumerate(self.dict['slice_keys']):
            self.dict['current'][i] = (np.sum(self.SU_data[:, 6][comparison])*E_CH)*c/(bin_length)

        self.axis_labels.update({'current': 'slice current [A]'})

    def StatsFrame(self):
        stats_dir = ('beta_x', 'std_pz', 'CoM_y', 'CoM_x', 
                     'std_y', 'std_x', 'current', 'z_pos',
                     'CoM_py', 'beta_y', 'e_x','slice_z', 
                     'CoM_px','e_y','CoM_pz', 'CoM_z', 
                     'std_px', 'std_py', 'std_z')

        stats_dir = {k: self.dict[k] for k in stats_dir}

        return pd.DataFrame(stats_dir)


class FEL_Approximations(Statistics):
    '''Calculates and stores basic FEL parameters'''

    def __init__(self, filename):
        super(FEL_Approximations, self).__init__(filename)



    def undulator(self, undulator_period=0, magnetic_field=0, k_fact=1):
        '''Calculates basic undulator parameters - assumes planar arrangment'''

        if magnetic_field != 0:
            self.dict['K_fact'] = feq.undulator_parameter(magnetic_field, undulator_period)
        else:
            self.dict['K_fact'] = float(k_fact)

        self.dict['undulator_period'] = undulator_period
        self.dict['gamma_res'] = feq.resonant_electron_energy(np.average(self.dict['CoM_pz'])*m*c*c, 0)
        self.dict['wavelength_res'] = feq.resonant_wavelength(
            undulator_period, self.dict['K_fact'], self.dict['gamma_res'])

    def pierce(self, slice_no):
        '''Calculates Pierce Parameter for a slice, 
        returns pierce and 1d gain_length'''

        K_JJ2 = (self.dict['K_fact']*feq.EM_charge_coupling(self.dict['K_fact']))**2
        pierce = self.dict['current'][slice_no]/(feq.alfven*self.dict['gamma_res']**3)
        pierce = pierce*(self.dict['undulator_period']**2)/\
                 (2*np.pi*self.dict['std_x'][slice_no]*self.dict['std_y'][slice_no])
        pierce = (pierce*K_JJ2/(32*np.pi))**(1.0/3.0)
        gain_length = (self.dict['undulator_period']/(4*np.pi*np.sqrt(3.0)*pierce))

        return pierce, gain_length

    def gain_length(self):
        ''' Calculates the 1D and Ming Xie gain length per slice'''

        if not 'undulator_period' in self.dict:
            n = float(raw_input('Need to define undulator period [m]:'))
            T = float(raw_input(
                'Need to define peak field (optional, 0 to ignore, then you have to define K) [T]:')
                )
            if T != 0:
                K = float(raw_input('Need to define K:'))
            self.undulator(undulator_period=n, magnetic_field=T, K_fact=K)
        
        self.dict['MX_gain'] = np.empty(self.dict['Num_Slices'])
        self.dict['1D_gain'] = np.empty(self.dict['Num_Slices'])
        self.dict['pierce'] = np.empty(self.dict['Num_Slices'])

        for i, (std_pz, CoM_pz, std_x, e_x, beta_x) in\
         enumerate(zip(self.dict['std_pz'], self.dict['CoM_pz'],
                       self.dict['std_x'], self.dict['e_x'],
                       self.dict['beta_x'])):

            rho, gain = self.pierce(i)
            ny = float(feq.scaled_e_spread(
                std_pz/CoM_pz, gain, self.dict['undulator_period']))
            nd = float(feq.scaled_transverse_size(
                std_x, gain, self.dict['wavelength_res'][0]))
            ne = float(feq.scaled_emittance(
                e_x / CoM_pz, gain, self.dict['wavelength_res'][0], beta_x))

            self.dict['pierce'][i] = rho
            self.dict['1D_gain'][i] = gain
            self.dict['MX_gain'][i] = float(gain*(1+feq.ming_xie_factor(nd, ne, ny)))

        self.axis_labels.update({'MX_gain': 'Ming Xie Gain Length',
                                 '1D_gain': '1D Gain Length',
                                 'pierce': 'Pierce Parameter'})

    def FELFrame(self):
        FEL_dir = ('pierce','1D_gain','MX_gain','z_pos')
        FEL_dir = {k: self.dict[k] for k in FEL_dir}
        return pd.DataFrame(FEL_dir)

class ProcessedData(FEL_Approximations):
    '''Class that automatically prepares and stores all data for plotting'''

    def __init__(self, filename, undulator_period, peak_field=0, k_fact=1, num_slices=False):
        super(ProcessedData, self).__init__(filename)
        if not num_slices:
            warnings.warn('did not specify slice number, will slice by estimated coherence length')
            num_slices = self.optimal_slice(undulator_period,k_fact)
            #print 'num_slices = ' + str(num_slices)
            self.slice(num_slices)
        else:
            self.slice(num_slices)
        self.calc_emittance()
        self.calc_CoM()
        self.calc_current()
        self.undulator(undulator_period, peak_field, k_fact)
        self.gain_length()


class Panda_Plotting():
    '''Class to quickly create pandas/matplotlib plots from a Processed_Data object'''
    def __init__(self, processed_data):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        self.plt, self.tk = plt, ticker
        self.distframe = processed_data.DistFrame()
        self.statsframe = processed_data.StatsFrame()
        self.FELframe = processed_data.FELFrame()
        self.file_path = processed_data.filename[0:-3]



    def prep_plot(self, x_axis, y_ax_1, y_ax_2=False, title=None, kind='line',
                  log=False, ID=None, x_label=False, y_label=False, show=False):
        '''Plotting routine - can plot up to two variables on one graph, log turns on logarithmic scale 
        on y axis, other entries are self explanatory'''


        

        if x_axis in self.statsframe.keys(): 
            if (y_ax_1 in self.FELframe.keys()) or (y_ax_2 in self.FELframe.keys()):
                dataset = pd.concat([panda_FEL,panda_stats], axis=1, join_axes=[panda_FEL.index]) #joins the two
            else:
                dataset = self.statsframe
        elif x_axis in self.distframe.keys():
            dataset = self.distframe
        else:
            dataset = self.FELframe

        rng_x, rng_y = [9999,0], [99999,0]   #Routine to define plotting range, ugly, but necessary
        if dataset[x_axis].max() > rng_x[1]:
            rng_x[1] = dataset[x_axis].max()
        if dataset[x_axis].min() < rng_x[0]:
            rng_x[0] = dataset[x_axis].min()
        if dataset[y_ax_1].max() > rng_y[1]:
            rng_y[1] = dataset[y_ax_1].max()
        if dataset[y_ax_1].min() < rng_y[0]:
            rng_y[0] = dataset[y_ax_1].min()
        if y_ax_2:
            if dataset[y_ax_2].max() > rng_y[1]:
                rng_y[1] = dataset[y_ax_2].max()
            if dataset[y_ax_2].min() < rng_y[0]:
                rng_y[0] = dataset[y_ax_2].min()
        rng_x = (rng_x[0],rng_x[1])
        rng_y = (rng_y[0],rng_y[1])

        alpha = 1
        if kind == 'scatter':
            alpha = 0.05
        
        fig, ax = self.plt.subplots(1,1)
        
        if kind == 'scatter':
            if len(dataset[x_axis]) > 10:
                counts,xbins,ybins=np.histogram2d(dataset[x_axis],dataset[y_ax_1],bins=150,weights=dataset['NE'])
                asp = (xbins[-1] - xbins[0]) / (ybins[-1] - ybins[0])
                self.plt.imshow(counts.T, origin = 'lower', extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect=asp) # , interpolation = 'bicubic')
                self.plt.xlabel(x_axis)
                self.plt.ylabel(y_ax_1)
            else:
                dataset.plot(ax=ax, x=x_axis, y=y_ax_1, title=title, kind=kind, color='b',xlim=rng_x,ylim=rng_y, alpha=alpha, label=y_ax_1)
        else:
            dataset.plot(ax=ax, x=x_axis, y=y_ax_1, title=title, kind=kind, color='b',xlim=rng_x,ylim=rng_y, alpha=alpha, label=y_ax_1)


        if y_ax_2:
            dataset.plot(ax=ax, x=x_axis, y=y_ax_2, kind=kind, color='r', alpha=alpha, label=y_ax_2)
        else:
            pass

        if not show:
            fig.savefig(fname_format(self.file_path,ID), dpi = 150)
            self.plt.close(fig)
        else:
            self.plt.show()

    def plot_defaults(self):
        '''Plots the default selection of plots to png files'''
        self.prep_plot('z_pos','std_pz',
            title='Standard deviation of transverse coordinates per slice', ID='std-pz')        
        self.prep_plot('z_pos','std_x','std_y',
            'Standard deviation of transverse coordinates per slice', ID='CoM-pos')
        self.prep_plot('z_pos','std_px','std_py',
            'Standard deviation of transverse momenta per slice', ID='std-mom')
        self.prep_plot('z_pos','CoM_x','CoM_y',
            'Centre of mass of transverse coordinates per slice', ID='std-pos')
        self.prep_plot('z_pos','CoM_px','CoM_py',
            'Centre of mass of transverse momenta per slice', ID='CoM-mom')
        self.prep_plot('z_pos','current',
            title='Current per slice', ID='current')
        self.prep_plot('z_pos','e_x','e_y',
            'Transverse slice emittance', ID='em')
        self.prep_plot('z_pos','beta_x','beta_y',
            'Beta function per slice', ID='beta')

        self.prep_plot('x', 'y',kind='scatter', ID='xy',title='Transverse postitions')
        self.prep_plot('x', 'px',kind='scatter', ID='xpx',title='Horizontal phases pace')
        self.prep_plot('y', 'py',kind='scatter', ID='ypy',title='Vertical phasespace')
        self.prep_plot('z', 'pz',kind='scatter', ID='zpz',title='Longitudinal phasespace')
        self.prep_plot('px', 'py',kind='scatter', ID='pxpy',title='Screen divergence')
        self.prep_plot('z', 'px', kind='scatter', ID='zpx',title='Longitudinal horizontal phasespace correlations')
        self.prep_plot('z', 'py',kind='scatter', ID='zpy',title='Longitudinal vertical phasespace correlations')
        self.prep_plot('z', 'x', kind='scatter', ID='zx',title='Longitudinal and horizontal position correlations')
        self.prep_plot('z', 'y',kind='scatter', ID='zy',title='Longitudinal and vertical  position correlations')
        self.prep_plot('pz', 'x', kind='scatter', ID='pzx',title='Energy deviation - horizontal postion correlations')
        self.prep_plot('pz', 'y',kind='scatter', ID='pzy',title='Energy deviation - vertical postion correlations')
        self.prep_plot('pz', 'px', kind='scatter', ID='pzpx',title='Energy deviation - horizontal divergence correlations')
        self.prep_plot('pz', 'py',kind='scatter', ID='pzpy',title='Energy deviation - vertical divergence correlations')
        # to add more defaults, just add more prep plot functions to here with the required data
        
        


class Bokeh_Plotting():
    '''A class that stores plots in Bokeh and allows
        for the generation of a html with all plots'''

    def __init__(self, SU_distribution):
        from bokeh.plotting import figure #So not to be unusable if bokeh is not
        from bokeh.io import output_file, show, curdoc, save #on the system
        from bokeh.layouts import layout
        from bokeh.models import Tabs, Panel
        #from bokeh.models import Range1d
        self.SU_distribution = SU_distribution
        self.dict = SU_distribution.dict
        self.axis_labels = SU_distribution.axis_labels
        self.plots = {}
        self.figure, self.save, self.output_file, self.show, self.curdoc\
            = (figure, save, output_file, show, curdoc)
        self.layout, self.Tabs, self.Panel = (layout, Tabs, Panel)

    def custom_plot(self, x_axis, y_axis, key='', plotter='circle', color='green',
                    file_name=False, text_color='black', legend=False, title=True,
                    logscale = False, save = False):

        from bokeh.models import Range1d

        '''Takes two strings from dict and plots x,y with circles or line plot
        this is saved in a dictionary called 'plots' which contains key:plot_object '''

        axis_titles = {'x':'X position', 'px':'X momentum', 'y':'Y position',
                       'py':'Y momentum', 'z':'Z position', 'pz':'Z momentum',
                       'e_x':'emittance', 'e_y':'emittance', 'slice_z':'Z position',
                       'mean_x':'mean position', 'CoM_x':'CoM X position',
                       'CoM_y': 'CoM Y position', 'NE':'Weight', 'CoM_pz':'CoM Z momentum',
                       'CoM_px':'CoM X momentum', 'CoM_py':'CoM Y momentum', 'current':'Current',
                       'std_pz':'STD of Z momentum', 'std_x':'STD of x position',
                       'std_y':'STD of y position', 'beta_x':'beta x', 'beta_y':'beta y',
                       'MX_gain': 'Ming Xie Gain Length', '1D_gain': '1D Gain Length',
                       'std_px':'std_px', 'std_py':'std_py', 'pierce': 'Pierce Parameter',
                       'alpha_x':'alpha_x', 'alpha_y':'alpha_y'}

        x_data = self.dict[x_axis]
        y_data = self.dict[y_axis]

        if title:
            title = ''.join([axis_titles[y_axis], ' against ', axis_titles[x_axis]])


        if logscale:
            p = self.figure(title=title, y_axis_type="log",
                            x_axis_label=self.axis_labels[x_axis],
                            y_axis_label=self.axis_labels[y_axis])
        else:
            p = self.figure(title=title,
                            x_axis_label=self.axis_labels[x_axis],
                            y_axis_label=self.axis_labels[y_axis])


        p.yaxis.axis_label_text_color = text_color

        if not legend:
            if plotter == 'circle':
                # assuming particles in this case...
                if (len(x_data) > 10):
                    counts,xbins,ybins=np.histogram2d(x_data,y_data,bins=150,weights=self.dict['NE'])
                    p.image(image=[counts.transpose()], x=xbins[0], y=ybins[0], dw=xbins[-1]-xbins[0], dh=ybins[-1]-ybins[0], dilate=True, palette="Spectral11")
                    #p.image(image=[counts], x=xbins[0], y = ybins[0], dw=1, dh=1, palette="Spectral11")
                    p.x_range=Range1d(xbins[0], xbins[-1])
                    p.y_range=Range1d(ybins[0], ybins[-1])
                else:
                    p.circle(x_data, y_data, color=color)
            elif plotter == 'line':
                p.line(x_data, y_data, color=color)


        else:
            if plotter == 'circle':
                p.circle(x_data, y_data, color=color, legend=legend)
            elif plotter == 'line':
                p.line(x_data, y_data, color=color, legend=legend)            

        if key == '':
            key = x_axis+'_'+y_axis
        
        self.plots.update({key:p})
        if save:
            l = self.layout([[p]])
            if not file_name:
                self.save(l,key)
            else:
                self.save(l,file_name) 

        return p

    def prepare_defaults(self, file_name=False):
        '''
        Prepares the default plots that will then be plotted by plot_defaults 
        Modifying this does not automatically alter the layout, so this must be added in manually
        '''
        if not self.SU_distribution.SI:
            self.SU_distribution.su2si()
            n = int(raw_input('Enter number of slices (integer): '))
            self.SU_distribution.slice(n)
            self.SU_distribution.calc_emittance()
            self.SU_distribution.calc_CoM()
            self.SU_distribution.calc_current()

        if not file_name:
            self.output_file(self.SU_distribution.filename[:-3]+'.html')

        else:
            self.output_file(file_name+'.html')

        x_y = self.custom_plot('x', 'y')
        x_px = self.custom_plot('x', 'px')
        y_py = self.custom_plot('y', 'py')
        z_pz = self.custom_plot('z', 'pz')
        px_py = self.custom_plot('px', 'py')
        z_px = self.custom_plot('z', 'px')
        z_py = self.custom_plot('z', 'py')
        z_x = self.custom_plot('z', 'x')
        z_y = self.custom_plot('z', 'y')
        pz_x = self.custom_plot('pz', 'x')
        pz_y = self.custom_plot('pz', 'y')
        pz_px = self.custom_plot('pz', 'px')
        pz_py = self.custom_plot('pz', 'py')
        current = self.custom_plot('slice_z', 'current', key='current', plotter='line')
        std = self.custom_plot('slice_z', 'std_pz', key='std', plotter='line')
       

        e_y = self.custom_plot('slice_z', 'e_x', key='e_y', plotter='line', legend='E_x')
        e_y.line(self.dict['slice_z'], self.dict['e_y'], color='blue', legend="E_y")

        mean_pos = self.custom_plot('slice_z', 'CoM_x', key='mean_pos', 
                                    plotter='line', legend="CoM x")
        mean_pos.line(self.dict['slice_z'], self.dict['CoM_y'], 
                      color='blue', legend="CoM y")

        CoM_p = self.custom_plot('slice_z', 'CoM_px', key='CoM_p', 
                                 plotter='line',legend="CoM px")
        CoM_p.line(self.dict['slice_z'], self.dict['CoM_py'], 
                   color='blue', legend="CoM py")

        beta = self.custom_plot('slice_z', 'beta_x', key='beta', plotter='line', legend="B(x)")
        beta.line(self.dict['slice_z'], self.dict['beta_y'], color='blue', legend="B(y)")

        alphax = self.custom_plot('slice_z', 'alpha_x', key='alphax', plotter='line', legend="alpha x")
        alphax.line(self.dict['slice_z'], self.dict['alpha_y'], color='blue', legend="alpha y")
        
        CoM_pz = self.custom_plot('slice_z', 'CoM_pz', key='CoM_pz', plotter='line')

        if self.dict.__contains__('MX_gain'):
            FEL_gain = self.custom_plot('slice_z','MX_gain', key='gain', plotter='line', legend='Ming Xie gain')
            gain_length = self.custom_plot('slice_z', '1D_gain', key='gain_length', plotter='line', legend="1D gain length")
            FEL_gainL = self.custom_plot('slice_z','MX_gain', key='gain_log', plotter='line', legend='Ming Xie gain', logscale = True)
            gain_lengthL = self.custom_plot('slice_z', '1D_gain', key='gain_length_log', plotter='line', legend="1D gain length", logscale = True)

            

        #To add more simply add more lines in the format above, code must beedited below to display this
    def plot_defaults(self, show_html = False):
        '''creates bokeh plots and html file, shows automatically'''

        if hasattr(self, 'auto_plots'):
            pass
        else:
            self.prepare_defaults()

        for key, val in self.plots.iteritems():
            exec(key + '=val')
        #creates a temporary variable for each entry in the plot dictionary, simply for ease of use here
        l1 = self.layout([[x_y, px_py],
                          [x_px, y_py]], sizing_mode='fixed')
                    
        l2 = self.layout([[z_px, z_py],
                          [z_x, z_y], 
                          [z_pz]], sizing_mode='fixed')

        l3 = self.layout([[pz_x, pz_y],
                          [pz_px, pz_py]], sizing_mode='fixed')

        l4 = self.layout([[e_y, mean_pos],
                          [CoM_p,CoM_pz],
                          [current,std],
                          [beta, alphax]], sizing_mode='fixed')

        tab1 = self.Panel(child=l1, title="Transverse phase space")
        tab2 = self.Panel(child=l2, title="Longitudinal phase space 1")
        tab3 = self.Panel(child=l3, title="Longitudinal phase space 2")
        tab4 = self.Panel(child=l4, title="Slice properties")

        if self.dict.__contains__('MX_gain'):
            l5 = self.layout([[gain,gain_length], 
                              [gain_log, gain_length_log]], sizing_mode='fixed')
            tab5 = self.Panel(child=l5, title="FEL parameters")
            tabs = self.Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])
            #tabs = self.Tabs(tabs=[tab1, tab4, tab5])

        else:
            tabs = self.Tabs(tabs=[tab1, tab2, tab3, tab4])

        self.curdoc().add_root(tabs)
        self.save(tabs,self.SU_distribution.filename[:-3]+'.html')
        if show_html:
            self.show(tabs)

        #an arbitrary number of tabs and plots can be added, first by adding it to a layout, creating a panel and finally 
        #adding the panel to a tab and adding that tab to a page
