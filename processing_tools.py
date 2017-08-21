# -*- coding: utf-8 -*-

#import matplotlib.pyplot as plt
#
#Author: Daniel Bultrini, danielbultrini@gmail.com

import numpy as np
import sys
import tables
import warnings
from particle_distribution.FEL_equations import *


c = float(3.0e+8)                    # Speed of light
m = float(9.11e-31)                  # mass of electron
E_CH = float(1.602e-19)
P = np.pi


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return np.sqrt(variance)

class SU_particle_distribution(object):
    '''Reads and provides processing to SU particle distributions

    initializes with a filename and the data,
    which then can be processed by in built methods'''

    def __init__(self, filename):
        self.filename = filename
        self.axis_labels = {'x':'X position ', 'px':'X momentum ', 'y':'Y position ',
                            'py':'Y momentum ', 'z':'Z position ', 'pz':'Z momentum', 'NE':'Weight',
                            'e_y': 'Y emittance ', 'e_x':'X emittance ', 'slice_z':'Z position ',
                            'mean_x' : 'mean x position ', 'mean_y' : 'mean y position '}
        self.axis_units = {'x':'[SU]', 'px':'[SU]', 'y':'[SU]',
                           'py':'[SU]', 'z':'[SU]', 'pz':' [SU]', 'NE':'[SU]',
                           'e_y': '[SU]', 'e_x':'[SU]', 'slice_z':'[SU]',
                           'mean_x' :'[SU]', 'mean_y' : '[SU]'}

        with tables.open_file(filename, 'r') as f:
            self.SU_data = f.root.Particles.read()

        self.directory = {'x':self.SU_data[:, 0], 'px':self.SU_data[:, 1],
                          'y':self.SU_data[:, 2], 'py':self.SU_data[:, 3],
                          'z':self.SU_data[:, 4], 'pz':self.SU_data[:, 5],
                          'NE':self.SU_data[:, 6], 'SI' : False}

    def SU2SI(self):
        '''Converts data to SI if needed'''

        self.SU_data[:, 1] = self.SU_data[:, 1]*(m*c)
        self.SU_data[:, 3] = self.SU_data[:, 3]*(m*c)
        self.SU_data[:, 5] = self.SU_data[:, 5]*(m*c)
        self.directory['SI'] = True

    def Slice(self, Num_Slices):
        '''set data slicing for use in certain routines if needed,
        returns 'self.z_pos' as array with z positions and self.Slice_Keys
        with boolean arrays for use in operations'''

        self.Num_Slices = Num_Slices
        self.Step_Z = (np.max(self.SU_data[:, 4])-np.min(self.SU_data[:, 4]))/self.Num_Slices
        self.Slice_Keys = np.ones((self.Num_Slices, self.SU_data.shape[0]), dtype=bool)
        self.z_pos = np.zeros(Num_Slices)

        for slice_no in xrange(self.Num_Slices):
            z_low = np.min(self.SU_data[:, 4])+(slice_no*self.Step_Z)
            z_high = np.min(self.SU_data[:, 4])+((slice_no+1)*self.Step_Z)
            z_pos = (z_high+z_low)/2.0
            self.z_pos[slice_no] = z_pos
            self.Slice_Keys[slice_no] = np.array((self.SU_data[:, 4] >= z_low) &\
                (self.SU_data[:, 4] < z_high), dtype=bool)

        self.directory.update({'slice_z': self.z_pos})

    def Calculate_Emittance(self):
        '''Returns the emittance per slice as arrays self.eps_rms_x, self.eps_rms_y
        directories 'e_x' 'and e_y'''

        self.eps_rms_x, self.eps_rms_y = np.empty(self.Num_Slices), np.empty(self.Num_Slices)

        for i, comparison in enumerate(self.Slice_Keys):
            m_POSx = self.SU_data[:, 0][comparison]
            m_mOmx = self.SU_data[:, 1][comparison]
            m_POSy = self.SU_data[:, 2][comparison]
            m_mOmy = self.SU_data[:, 3][comparison]
            ###########~~Code by Piotr Traczkowski~~################################################
            x_2 = ((np.sum(m_POSx*m_POSx))/len(m_POSx))-(np.mean(m_POSx))**2.0                     #
            px_2 = ((np.sum(m_mOmx*m_mOmx))/len(m_mOmx))-(np.mean(m_mOmx))**2.0                    #
            xpx = np.sum(m_POSx*m_mOmx)/len(m_POSx)-np.sum(m_POSx)*np.sum(m_mOmx)/(len(m_POSx))**2 #
                                                                                                   #
            y_2 = ((np.sum(m_POSy*m_POSy))/len(m_POSy))-(np.mean(m_POSy))**2.0                     #
            py_2 = ((np.sum(m_mOmy*m_mOmy))/len(m_mOmy))-(np.mean(m_mOmy))**2.0                    #
            ypy = np.sum(m_POSy*m_mOmy)/len(m_POSy)-np.sum(m_POSy)*np.sum(m_mOmy)/(len(m_POSy))**2 #
                                                                                                   #
            self.eps_rms_x[i] = (1.0/(m*c))*np.sqrt((x_2*px_2)-(xpx*xpx))                          #
            self.eps_rms_y[i] = (1.0/(m*c))*np.sqrt((y_2*py_2)-(ypy*ypy))                          #
            ########################################################################################

        self.directory.update({'e_x': self.eps_rms_x,
                               'e_y': self.eps_rms_y})

    def Com(self):
        '''Returns the weighted average positions in
            x and y as self.mean_x, self.mean_y'''

        if not self.directory.__contains__('e_x'):
            warnings.warn('Need to run emittance first')
            self.Calculate_Emittance()


        # allocate arrays of appropiate size in memory
        self.Com_x, self.Com_y = np.empty(self.Num_Slices), np.empty(self.Num_Slices)
        self.Com_px, self.Com_py = np.empty(self.Num_Slices), np.empty(self.Num_Slices)
        self.Com_pz, self.std_pz = np.empty(self.Num_Slices), np.empty(self.Num_Slices)
        self.std_x, self.std_y = np.empty(self.Num_Slices), np.empty(self.Num_Slices)

        #Loop through slices and apply weighed standard deviation and average (Com)
        for i, comparison in enumerate(self.Slice_Keys):
            weight = self.SU_data[:, 6][comparison]

            self.Com_x[i] = np.average(self.SU_data[:, 0][comparison], weights=weight)
            self.Com_y[i] = np.average(self.SU_data[:, 2][comparison], weights=weight)
            self.Com_px[i] = np.average(self.SU_data[:, 1][comparison], weights=weight)
            self.Com_py[i] = np.average(self.SU_data[:, 3][comparison], weights=weight)
            self.Com_pz[i] = np.average(self.SU_data[:, 5][comparison], weights=weight)
            self.std_pz[i] = weighted_std(self.SU_data[:, 5][comparison], weight)
            self.std_x[i] = weighted_std(self.SU_data[:, 0][comparison], weight)
            self.std_y[i] = weighted_std(self.SU_data[:, 2][comparison], weight)


        self.beta_x, self.beta_y = ((np.sqrt(4*np.log(2))*self.std_x)**2)/self.eps_rms_x,\
                                    ((np.sqrt(4*np.log(2))*self.std_y)**2)/self.eps_rms_y

        self.directory.update({'Com_x': self.Com_x,
                               'Com_y': self.Com_y,
                               'Com_px': self.Com_px,
                               'Com_py': self.Com_py,
                               'Com_pz': self.Com_pz,
                               'std_pz': self.std_pz,
                               'std_y': self.std_y,
                               'std_x': self.std_x,
                               'beta_x': self.beta_x,
                               'beta_y': self.beta_y})

        self.axis_labels.update({'Com_x': 'Com X position',
                                 'Com_y': 'Com Y position',
                                 'Com_px': 'Com X momentum',
                                 'Com_py': 'Com Y momentum',
                                 'Com_pz': 'Com Z momentum',
                                 'std_pz': 'STD of Z momentum',
                                 'std_x': 'STD of x position',
                                 'std_y': 'STD of y position',
                                 'beta_x': 'beta x',
                                 'beta_y': 'beta y'})

    def get_current(self):
        '''Calculates current per slice and returns array - uses approximation
        current = total charge per slice * speed of light '''

        if not self.directory['SI']:
            warnings.warn('might get strange results without SI conversion')

        self.current = np.empty(self.Num_Slices)
        bin_length = self.z_pos[1]-self.z_pos[0]

        for i, comparison in enumerate(self.Slice_Keys):
            self.current[i] = (np.sum(self.SU_data[:, 6][comparison])*E_CH)*c/(bin_length)

        self.directory.update({'current': self.current})
        self.axis_labels.update({'current': 'slice current [A]'})


    def undulator(self, undulator_period=0, magnetic_field=0, K=1):

        if magnetic_field != 0:
            self.K = undulator_parameter(magnetic_field, undulator_period)
        else:
            self.K = float(K)

        self.undulator_period = undulator_period
        self.gama_res = resonant_electron_energy(np.average(
            self.SU_data[:, 5], weights=self.SU_data[:, 6])*c, 0)
        self.wavelen_res = resonant_wavelength(undulator_period, self.K, self.gama_res)

    def pierce(self, slice_no):
        K_JJ2 = (self.K*EM_charge_coupling(self.K))**2
        pierce = self.current[slice_no]/(alfven*self.gama_res**3)
        pierce = pierce*(self.undulator_period**2)/\
                 (2*const.pi*self.std_x[slice_no]*self.std_y[slice_no])
        pierce = (pierce*K_JJ2/(32*const.pi))**(1.0/3.0)
        gain_length = (self.undulator_period/(4*const.pi*np.sqrt(3.0)*pierce))
        return pierce, gain_length

    def gain_length(self):
        self.ming_xie_gain_length = np.empty(self.Num_Slices)
        self.gain_length_1D = np.empty(self.Num_Slices)
        self.pierce_param = np.empty(self.Num_Slices)

        for i in xrange(self.Num_Slices):

            rho, gain = self.pierce(i)
            ne = scaled_e_spread(self.std_pz[i]/self.Com_pz[i], gain, self.undulator_period)
            nd = scaled_transverse_size(self.std_x[i], gain, self.wavelen_res[0])
            ny = scaled_emittance(self.eps_rms_x[i], gain, self.wavelen_res[0], self.beta_x[i])
            self.ming_xie_gain_length[i] = gain*(1+ming_xie_factor(nd, ne, ny))
            self.pierce_param[i] = rho
            self.gain_length_1D[i] = gain
        self.directory.update({'mX_gain': self.ming_xie_gain_length})
        self.axis_labels.update({'mX_gain': 'ming Xie Gain Length'})




class SU_Bokeh_Plotting(SU_particle_distribution):
    '''A class that stores plots in Bokeh and allows
        for the generation of a html with all plots'''

    def __init__(self, SU_distribution):
        from bokeh.plotting import figure, save
        from bokeh.io import output_file, show, curdoc
        from bokeh.layouts import layout
        from bokeh.models import Tabs, Panel
        #from bokeh.models import Range1d
        self.SU_distribution = SU_distribution
        self.directory = SU_distribution.directory
        self.axis_labels = SU_distribution.axis_labels
        self.plots = {}
        self.figure, self.save, self.output_file, self.show, self.curdoc \
            = (figure, save, output_file, show, curdoc)
        self.layout, self.Tabs, self.Panel = (layout, Tabs, Panel)

    def custom_plot(self, x_axis, y_axis, key ='', plotter='circle', color='green',
                    filename=' ', text_color='black',
                    Legend=False):

        '''Takes two strings from directory and plots x,y with circles or line plot
        direct call = True creates '''

        axis_titles = {'x':'X position', 'px':'X momentum', 'y':'Y position',
                       'py':'Y momentum', 'z':'Z position', 'pz':'Z momentum',
                       'e_x':'emittance', 'e_y':'emittance', 'slice_z':'Z position',
                       'mean_x':'mean position', 'Com_x':'Com X position', 
                       'Com_y': 'Com Y position', 'NE':'Weight', 'Com_pz':'Com Z momentum',
                       'Com_px':'Com X momentum', 'Com_py':'Com Y momentum', 'current':'Current', 
                       'std_pz':'STD of Z momentum','std_x':'STD of x position',
                       'std_y':'STD of y position', 'beta_x':'beta x', 'beta_y':'beta y'}

        self.axis_labels['e_y'] = 'Emittance'

        x_data = self.directory[x_axis]
        y_data = self.directory[y_axis]
        title= ''.join([axis_titles[y_axis], ' against ', axis_titles[x_axis]])
        self.output_file(filename[:-3]+'.html')

        p = self.figure(title=title,
                        x_axis_label=self.axis_labels[x_axis], 
                        y_axis_label=self.axis_labels[y_axis])

        self.axis_labels['e_y'] = 'Y Emittance '

        p.yaxis.axis_label_text_color = text_color

        if not Legend:
            if plotter == 'circle':
                p.circle(x_data, y_data, color=color)
            elif plotter == 'line':
                p.line(x_data, y_data, color=color)


        else:
            if plotter == 'circle':
                p.circle(x_data, y_data, color=color, legend=Legend)
            elif plotter == 'line':
                p.line(x_data, y_data, color=color, legend=Legend)            

        if key == '':
            key = x_axis+'_'+y_axis
        self.plots.update({key:p})

        return p

    def prepare_defaults(self):
        if not self.SU_distribution.directory['SI']:
            self.SU_distribution.SU2SI()
            n = int(raw_input('Enter number of slices (integer): '))
            self.SU_distribution.Slice(n)
            self.SU_distribution.Calculate_Emittance()
            self.SU_distribution.Com()
            self.SU_distribution.get_current()

        self.output_file("tabs.html")

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

        e_y = self.custom_plot('slice_z', 'e_x', key='e_y', plotter='line', Legend='E_x')
        e_y.line(self.directory['slice_z'], self.directory['e_y'], color='blue', legend="E_y")

        mean_pos = self.custom_plot('slice_z', 'Com_x', key='mean_pos', 
                                    plotter='line', Legend="Com x")
        mean_pos.line(self.directory['slice_z'], self.directory['Com_y'], 
                      color='blue', legend="Com y")

        Com_p = self.custom_plot('slice_z', 'Com_px', key='Com_p', 
                                 plotter='line',Legend="Com px")
        Com_p.line(self.directory['slice_z'], self.directory['Com_py'], 
                   color='blue', legend="Com py")

        beta = self.custom_plot('slice_z', 'beta_x', key='beta', plotter='line', Legend="B(x)")
        beta.line(self.directory['slice_z'], self.directory['beta_y'], color='blue', legend="B(y)")
        Com_pz = self.custom_plot('slice_z', 'Com_pz', key='Com_pz', plotter='line')


    def plot_defaults(self):
        if hasattr(self, 'auto_plots'):
            pass
        else:
            self.prepare_defaults()

        for key, val in self.plots.iteritems():
            exec(key + '=val')

        l1 = self.layout([[x_y, px_py],
                         [x_px, y_py]], sizing_mode='fixed')
                    
        l2 = self.layout([[z_px, z_py],
                         [z_x, z_y]], sizing_mode='fixed')

        l3 = self.layout([[pz_x, pz_y],
                         [pz_px, pz_py]], sizing_mode='fixed')

        l4 = self.layout([[e_y, mean_pos],
                         [Com_p,Com_pz],
                         [current,std],
                         [beta]],sizing_mode='fixed')


        tab1 = self.Panel(child=l1, title="X Y ")
        tab2 = self.Panel(child=l2, title="Z ")
        tab3 = self.Panel(child=l3, title="transverse phase space")
        tab4 = self.Panel(child=l4, title="Emittances")
        tabs = self.Tabs(tabs=[tab1, tab2, tab3, tab4])

        self.curdoc().add_root(tabs)
        self.show(tabs)