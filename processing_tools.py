# -*- coding: utf-8 -*-

#import matplotlib.pyplot as plt
#
#Author: Daniel Bultrini, danielbultrini@gmail.com
from multiprocessing import Pool
import numpy as np
import sys
import tables
from bokeh.plotting import figure, save
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import layout
from bokeh.models import Toggle, BoxAnnotation, CustomJS, Tabs, Panel, LinearAxis
from bokeh.models import Range1d
from FEL_equations import *


c = 3.0e+8                    # Speed of light
m = 9.11e-31                  # mass of electron
e_ch = 1.602e-19
Pi = np.pi


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return np.sqrt(variance)


def norm(a):
    return 2*((a - min(a))/(max(a)-min(a)))-1


class SU_particle_distribution(object):
    '''Reads and provides processing to SU particle distributions

    initializes with a filename and the data,
    which then can be processed by in built methods'''

    def __init__(self, filename):
        self.filename = filename
        self.emit = []
        self.axis_labels = {'x':'X position [SU]', 'px':'X momentum [SU]', 'y':'Y position [SU]',
            'py':'Y momentum [SU]', 'z':'Z position [SU]','pz':'Z momentum [SU]', 'NE':'Weight',
            'e_y': 'Y emittance [SU]', 'e_x':'X emittance [SU]', 'slice_z':'Z position [SU]',
            'mean_x' : 'mean x position [SU]', 'mean_y' : 'mean y position [SU]'}

        with tables.open_file(filename,'r') as f:  
            self.SU_data =  f.root.Particles.read()
        
        self.directory = {'x':self.SU_data[:, 0], 'px':self.SU_data[:, 1],
                    'y':self.SU_data[:, 2], 'py':self.SU_data[:, 3],
                    'z':self.SU_data[:, 4],'pz':self.SU_data[:, 5],
                    'NE':self.SU_data[:, 6]}
    
            

    def SU2SI(self):
        '''Converts data to SI if needed'''
        self.SU_data[:, 1] = self.SU_data[:, 1]*(m*c)
        self.SU_data[:, 3] = self.SU_data[:, 3]*(m*c)
        self.SU_data[:, 5] = self.SU_data[:, 5]*(m*c)

        self.axis_labels = {'x':'X position [m]','px':'X momentum [mmrad]','y':'Y position [m]',
            'py':'Y momentum [mmrad]','z':'Z position [m]','pz':'Z momentum [mmrad]','NE':'Weight',
            'e_y': 'Y emittance [mmrad]', 'e_x':'X emittance [mmrad]', 'slice_z':'Z position [um]',
            'mean_x' : 'mean x position [m]', 'mean_y' : 'mean y position [m]'}

    def Slice(self, Num_Slices):
        '''set data slicing for use in certain routines if needed,
        returns 'self.z_pos' as array with z positions and self.Slice_Keys
        with boolean arrays for use in operations'''
        self.Num_Slices = Num_Slices
        self.Step_Z = (np.max(self.SU_data[:, 4])-np.min(self.SU_data[:, 4]))/self.Num_Slices
        self.Slice_Keys = np.ones((self.Num_Slices,self.SU_data.shape[0]),dtype=bool)
        self.z_pos = np.zeros(Num_Slices)

        for slice_no in xrange(self.Num_Slices):
            z_low = np.min(self.SU_data[:, 4])+(slice_no*self.Step_Z)
            z_high = np.min(self.SU_data[:, 4])+((slice_no+1)*self.Step_Z)
            z_pos = (z_high+z_low)/2.0
            self.z_pos[slice_no] = z_pos
            self.Slice_Keys[slice_no] = np.array((self.SU_data[:, 4]>=z_low) & (self.SU_data[:, 4]<z_high),dtype=bool)
        self.directory.update({'slice_z': self.z_pos})
    
    
    def Calculate_Emittance(self):
        '''Returns the emittance per slice as arrays self.eps_rms_x, self.eps_rms_y
        directories 'e_x' 'and e_y'''
        self.eps_rms_x, self.eps_rms_y  = np.empty(self.Num_Slices), np.empty(self.Num_Slices)
        
        i=0
        for comparison in self.Slice_Keys:
            m_POSx = self.SU_data[:,0][comparison]
            m_MOMx = self.SU_data[:,1][comparison]
            m_POSy = self.SU_data[:,2][comparison]
            m_MOMy = self.SU_data[:,3][comparison]

            x_2=((np.sum(m_POSx*m_POSx))/len(m_POSx))-(np.mean(m_POSx))**2.0
            px_2=((np.sum(m_MOMx*m_MOMx))/len(m_MOMx))-(np.mean(m_MOMx))**2.0
            xpx=np.sum(m_POSx*m_MOMx)/len(m_POSx)-np.sum(m_POSx)*np.sum(m_MOMx)/(len(m_POSx))**2.0

            y_2=((np.sum(m_POSy*m_POSy))/len(m_POSy))-(np.mean(m_POSy))**2.0
            py_2=((np.sum(m_MOMy*m_MOMy))/len(m_MOMy))-(np.mean(m_MOMy))**2.0
            ypy=np.sum(m_POSy*m_MOMy)/len(m_POSy)-np.sum(m_POSy)*np.sum(m_MOMy)/(len(m_POSy))**2.0

            self.eps_rms_x[i]=(1.0/(m*c))*np.sqrt((x_2*px_2)-(xpx*xpx))
            self.eps_rms_y[i]=(1.0/(m*c))*np.sqrt((y_2*py_2)-(ypy*ypy))
            i += 1
        self.directory.update( {'e_x': self.eps_rms_x, 
                                'e_y': self.eps_rms_y})

    def CoM(self):
        '''Returns the weighted average positions in 
            x and y as self.mean_x, self.mean_y'''

        # allocate arrays of appropiate size in memory
        self.CoM_x, self.CoM_y = np.empty(self.Num_Slices), np.empty(self.Num_Slices)
        self.CoM_px, self.CoM_py = np.empty(self.Num_Slices), np.empty(self.Num_Slices)
        self.CoM_pz, self.std_pz = np.empty(self.Num_Slices), np.empty(self.Num_Slices)
        self.std_x, self.std_y = np.empty(self.Num_Slices), np.empty(self.Num_Slices)

        #Loop through slices and apply weighed standard deviation and average as a sort of 'center of mass'
        i = 0
        for comparison in self.Slice_Keys:
            weight = self.SU_data[:, 6][comparison]

            self.CoM_x[i] = np.average(self.SU_data[:, 0][comparison], weights=weight)
            self.CoM_y[i] = np.average(self.SU_data[:, 2][comparison], weights=weight)
            self.CoM_px[i] = np.average(self.SU_data[:, 1][comparison], weights=weight)
            self.CoM_py[i] = np.average(self.SU_data[:, 3][comparison], weights=weight)
            self.CoM_pz[i] = np.average(self.SU_data[:, 5][comparison], weights=weight)
            self.std_pz[i] = weighted_std(self.SU_data[:, 5][comparison], weight)
            self.std_x[i]  = weighted_std(self.SU_data[:, 0][comparison], weight)
            self.std_y[i]  = weighted_std(self.SU_data[:, 2][comparison], weight)

            i+=1
        
        self.beta_x, self.beta_y = ((np.sqrt(4*np.log(2))*self.std_x)**2)/self.eps_rms_x,\
                                    ((np.sqrt(4*np.log(2))*self.std_y)**2)/self.eps_rms_y

        self.directory.update({ 'CoM_x': self.CoM_x, 
                                'CoM_y': self.CoM_y,
                                'CoM_px': self.CoM_px, 
                                'CoM_py': self.CoM_py,
                                'CoM_pz': self.CoM_pz,
                                'std_pz': self.std_pz,
                                'std_y': self.std_y,
                                'std_x': self.std_x,
                                'beta_x': self.beta_x,
                                'beta_y': self.beta_y,
                                })
        
        self.axis_labels.update({ 'CoM_x': 'CoM X position', 
                                'CoM_y': 'CoM Y position',
                                'CoM_px': 'CoM X momentum', 
                                'CoM_py': 'CoM Y momentum',
                                'CoM_pz': 'CoM Z momentum',
                                'std_pz': 'STD of Z momentum',
                                'std_x': 'STD of x position',
                                'std_y': 'STD of y position',
                                'beta_x': 'beta x',
                                'beta_y': 'beta y'})

    def get_current(self):
        '''Calculates current per slice and returns array - uses approximation 
        
        current = total charge per slice * speed of light '''
        self.current = np.empty(self.Num_Slices)
        bin_length = self.zpos[1]-self.z_pos[0]
        i = 0
        for comparison in self.Slice_Keys:
            self.current[i] = (np.sum(self.SU_data[:, 6][comparison])*e_ch)*c/(bin_length)
            i += 1

        self.directory.update({ 'current': self.current})
        self.axis_labels.update({ 'current': 'slice current [A]'})


    def undulator(self,undulator_period=0,magnetic_field=0,K=1):
        if magnetic_field != 0:
            self.K = undulator_parameter(magnetic_field,undulator_period)
        else:
            self.K = float(K)
        self.undulator_period = undulator_period
        self.gamma_res = resonant_electron_energy(np.average(
                        self.SU_data[:, 5], weights=self.SU_data[:, 6])*c,0)
        self.wavelen_res = resonant_wavelength(undulator_period,self.K,self.gamma_res)
        
    def pierce(self,slice_no):
        K_JJ2 = (self.K*EM_charge_coupling(self.K))**2
        pierce = self.current[slice_no]/(alfven*self.gamma_res**3)
        pierce = pierce*(self.undulator_period**2)/(2*const.pi*self.std_x[slice_no]*self.std_y[slice_no])
        pierce = (pierce*K_JJ2/(32*const.pi))**(1.0/3.0)
        gain_length = (self.undulator_period/(4*const.pi*np.sqrt(3.0)*pierce))
        return pierce, gain_length

    def gain_length(self):
        self.ming_xie_gain_length  = np.empty(self.Num_Slices)
        for i in xrange(self.Num_Slices):
            rho,gain = self.pierce(i)
            ne = scaled_e_spread(self.std_pz[i]/self.CoM_pz[i],gain,self.undulator_period)
            nd = scaled_transverse_size(self.std_x[i],gain,self.wavelen_res[0])
            ny = scaled_emittance(self.eps_rms_x[i],gain,self.wavelen_res[0],self.beta_x[i])
            print(gain,ne,nd,ny)
            self.ming_xie_gain_length[i] = gain*(1+Ming_Xie_factor(nd,ne,ny))


    def custom_plot(self,x_axis,y_axis, plotter = 'circle', color = 'green',
                    filename = ' ',direct_call=False, text_color = 'black',
                    Legend = False):

        '''Takes two strings "x,px,y,py,z,pz,NE" plots 
        corresponding values on x and y to file'''

        axis_titles = {'x':'X position','px':'X momentum','y':'Y position',
            'py':'Y momentum','z':'Z position','pz':'Z momentum','NE':'Weight',
            'e_x':'emittance','e_y':'emittance', 'slice_z':'Z position',
            'mean_x' : 'mean position', 'CoM_x': 'CoM X position', 'CoM_y': 'CoM Y position',
            'CoM_px': 'CoM X momentum', 'CoM_py': 'CoM Y momentum', 'CoM_pz': 'CoM Z momentum',
            'current': 'Current', 'std_pz': 'STD of Z momentum','std_x': 'STD of x position',
            'std_y': 'STD of y position', 'beta_x': 'beta x', 'beta_y': 'beta y'}
        
        self.axis_labels['e_y'] = 'Emittance'
        
        

        x_data = self.directory[x_axis]
        y_data = self.directory[y_axis]
        title=''.join([axis_titles[y_axis],' against ',axis_titles[x_axis]])
        output_file(self.filename[:-3]+'.html')

    
        p = figure(title=title,
                x_axis_label=self.axis_labels[x_axis], 
                y_axis_label=self.axis_labels[y_axis])
                # ,
                # x_range=Range1d(min(x_data), max(x_data)), 
                # y_range=Range1d(min(y_data), max(y_data))
                # )
        self.axis_labels['e_y'] = 'Y Emittance [SU]'

        p.yaxis.axis_label_text_color = text_color

        if not Legend:
            if plotter == 'circle':    
                p.circle(x_data,y_data,color=color)
            elif plotter == 'line':
                p.line(x_data,y_data,color=color)

        else:
            if plotter == 'circle':    
                p.circle(x_data,y_data,color=color,legend=Legend)
            elif plotter == 'line':
                p.line(x_data,y_data,color=color,legend=Legend)            
            
        if direct_call == True:
            save(self.p)

        return p

    def plot_defaults(self):
        self.Calculate_Emittance()
        self.CoM()
        self.get_current()

        output_file("tabs.html")

        x_y  = self.custom_plot('x','y')
        x_px = self.custom_plot('x','px')
        y_py = self.custom_plot('y','py')
        z_pz = self.custom_plot('z','pz')
        px_py = self.custom_plot('px', 'py')
        z_px = self.custom_plot('z','px')
        z_py = self.custom_plot('z','py')
        z_x = self.custom_plot('z','x')
        z_y = self.custom_plot('z','y')
        dp_x = self.custom_plot('pz','x')
        dp_y = self.custom_plot('pz','y')
        dp_px = self.custom_plot('pz','px')
        dp_py = self.custom_plot('pz','py')
        current = self.custom_plot('slice_z','current',plotter='line')
        std = self.custom_plot('slice_z','std_pz',plotter='line')

        e_y = self.custom_plot('slice_z','e_x',plotter='line',Legend ="E_x")
        e_y.line(self.directory['slice_z'],self.directory['e_y'],color='blue', legend ="E_y")
        
        mean_pos = self.custom_plot('slice_z','CoM_x',plotter='line',Legend ="CoM x")
        mean_pos.line(self.directory['slice_z'],self.directory['CoM_y'],color='blue', legend ="CoM y")

        CoM_p = self.custom_plot('slice_z','CoM_px',plotter='line',Legend ="CoM px")
        CoM_p.line(self.directory['slice_z'],self.directory['CoM_py'],color='blue', legend ="CoM py")

        beta = self.custom_plot('slice_z','beta_x',plotter='line',Legend ="B(x)")
        beta.line(self.directory['slice_z'],self.directory['beta_y'],color='blue', legend ="B(y)")


        CoM_pz = self.custom_plot('slice_z','CoM_pz',plotter='line')

        l1 = layout([[x_y, px_py],
                    [x_px, y_py]], sizing_mode='fixed')
                    
        l2 = layout([[z_px, z_py],
                    [z_x, z_y]], sizing_mode='fixed')

        l3 = layout([[dp_x, dp_y],
                    [dp_px, dp_py]], sizing_mode='fixed')

        l4 = layout([[e_y, mean_pos],
                    [CoM_p,CoM_pz],
                    [current,std],
                    [beta]],sizing_mode='fixed')

        tab1 = Panel(child=l1,title="X Y ")
        tab2 = Panel(child=l2,title="Z ")
        tab3 = Panel(child=l3,title="transverse phase space")
        tab4 = Panel(child=l4,title="Emittances")
        tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4 ])

        curdoc().add_root(tabs)
        show(tabs)


# class undulator(SU_particle_distribution):
    

#     def pierce(self,slice_no):
#         K_JJ2 = (K*EM_charge_coupling(self.K))**2
#         pierce = self.current[slice_no]/(alfven*self.gamma_res**3)
#         pierce = pierce*(self.undulator_period**2)/(2*const.pi*self.std_x*self.std_y)
#         pierce = (pierce*K_JJ2/(32*const.pi))**(1.0/3.0)
#         return pierce












    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










































































#


























































#     def Calculate_Emittance(self, slice_no):
#         '''Returns z, rms x and y emittances of SU_data for given slice 
#         calculated through a method by Floettmann, 2003'''

#         z_low=np.min(self.SU_data[:,4])+(slice_no*self.Step_Z)
#         z_high=np.min(self.SU_data[:,4])+((slice_no+1)*self.Step_Z)
#         z_pos=(z_high+z_low)/2.0
    

#         comparison = (self.SU_data[:,4]>=z_low) & (self.SU_data[:,4]<z_high)
    
#         m_POSx = self.SU_data[:,0][comparison]
#         m_MOMx = self.SU_data[:,1][comparison]
#         m_POSy = self.SU_data[:,2][comparison]
#         m_MOMy = self.SU_data[:,3][comparison]    

#         x_2=((np.sum(m_POSx*m_POSx))/len(m_POSx))-(np.mean(m_POSx))**2.0
#         px_2=((np.sum(m_MOMx*m_MOMx))/len(m_MOMx))-(np.mean(m_MOMx))**2.0
#         xpx=np.sum(m_POSx*m_MOMx)/len(m_POSx)-np.sum(m_POSx)*np.sum(m_MOMx)/(len(m_POSx))**2.0

#         y_2=((np.sum(m_POSy*m_POSy))/len(m_POSy))-(np.mean(m_POSy))**2.0
#         py_2=((np.sum(m_MOMy*m_MOMy))/len(m_MOMy))-(np.mean(m_MOMy))**2.0
#         ypy=np.sum(m_POSy*m_MOMy)/len(m_POSy)-np.sum(m_POSy)*np.sum(m_MOMy)/(len(m_POSy))**2.0


#         eps_rms_x=(1.0/(m*c))*np.sqrt((x_2*px_2)-(xpx*xpx))
#         eps_rms_y=(1.0/(m*c))*np.sqrt((y_2*py_2)-(ypy*ypy))
    
#         return z_pos,eps_rms_x,eps_rms_y  



#  def Store_Emittance(self):
#         '''Calculates emittance for ever slice and returns self.emittance with this data'''
#         for i in xrange(0,self.Num_Slices):
#             self.emit.append(self.Calculate_Emittance(i))
#         self.emittance = np.zeros((self.Num_Slices,3))

#         for i in xrange(0,self.Num_Slices):
#             self.emittance[i,0] = self.emit[i][0]
#             self.emittance[i,1] = self.emit[i][1]
#             self.emittance[i,2] = self.emit[i][2]
        
#         self.emittance = self.emittance[self.emittance[:,0].argsort()]
#         self.directory.update({'slice_z': self.emittance[:,0],
#                                 'e_x': self.emittance[:,1], 
#                                 'e_y': self.emittance[:,2]})










#     def plot_interactive(self,normal=False):
#         p = figure()
#         if not normal:
#             x_y = p.circle(self.SU_data[:,0], self.SU_data[:,2], color="blue")
#             x_px = p.circle(self.SU_data[:,0], self.SU_data[:,1], color="red")
#             y_py = p.circle(self.SU_data[:,2], self.SU_data[:,3], color="green")
#             z_pz = p.circle(self.SU_data[:,4], self.SU_data[:,5], color="yellow")

#         if normal:
#             x_y = p.circle(norm(self.SU_data[:,0]), norm(self.SU_data[:,2]), color="blue")
#             x_px = p.circle(norm(self.SU_data[:,0]), norm(self.SU_data[:,1]), color="red")
#             y_py = p.circle(norm(self.SU_data[:,2]), norm(self.SU_data[:,3]), color="green")
#             z_pz = p.circle(norm(self.SU_data[:,4]), norm(self.SU_data[:,5]), color="yellow")


#         We write coffeescript to link toggle with visible property of box and line
#         code = '''\
#         object.visible = toggle.active
#         '''

#         callback1 = CustomJS.from_coffeescript(code=code, args={})
#         toggle1 = Toggle(label="x vs y", button_type="success", callback=callback1)
#         callback1.args = {'toggle': toggle1, 'object': x_y}

#         callback2 = CustomJS.from_coffeescript(code=code, args={})
#         toggle2 = Toggle(label="x vs px", button_type="success", callback=callback2)
#         callback2.args = {'toggle': toggle2, 'object': x_px}

#         callback3 = CustomJS.from_coffeescript(code=code, args={})
#         toggle3 = Toggle(label="y vs py", button_type="success", callback=callback3)
#         callback3.args = {'toggle': toggle3, 'object': y_py}

#         callback4 = CustomJS.from_coffeescript(code=code, args={})
#         toggle4 = Toggle(label="z vs pz", button_type="success", callback=callback4)
#         callback4.args = {'toggle': toggle4, 'object': z_pz}


#         output_file("styling_visible_annotation_with_interaction.html")

#         show(layout([p], [toggle1, toggle2], [toggle3, toggle4]))
            














#     def Multi_Store_Emittance(self, processes=None):
#         p = Pool(processes)

#         for i in range(self.Num_Slices):
        
#             p.apply_async(unpack(self.Calculate_Emittance), args = (i,), callback = self.emit.append)
        
#         p.close()
#         p.join()
    
#         self.emittance = np.zeros((self.Num_Slices,3))

#         for i in xrange(self.Num_Slices):
#             self.emittance[i,0] = self.emit[i][0]
#             self.emittance[i,1] = self.emit[i][1]
#             self.emittance[i,2] = self.emit[i][2]
        
#         self.emittance = self.emittance[self.emittance[:,0].argsort()]


