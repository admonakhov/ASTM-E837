import numpy as np
from os.path import isfile
import pandas as pd
from scipy.interpolate import make_interp_spline, interp1d
import matplotlib.pyplot as plt


def std(s):
    s_std = 0
    for i in range(len(s)-3):
        s_std += ((s[i] - 3*s[i+1] + 3*s[i+2] - s[i+3])**2) 
    s_std = s_std / (20 *(len(s) - 3))
    return s_std


def mk_c(n=20):
    """
    Get the tri-diagonal “second derivative” matrix c
    ASTM E837 - 2013 10.3.2
    
    n: number of rows, int
    return: matrix n*n, np.array
    """
    c = np.zeros((n, n))
    der = np.array([-1, -2, -1])
    for i in range(1, n-1):
        c[i][i-1:i+2] = der
    return c


def rms(s):
    """
    Return root mean square
    ASTM E837 - 2013 10.3.7
    s: strain, float
    return: root mean square of strain, float
    """
    return np.sum(s**2)/s.shape[0]



class Coefficients:
    def __init__(self, a=None, b=None):
        if a == None and b == None:
            self.a = np.genfromtxt('coefficients/a.txt')
            self.b = np.genfromtxt('coefficients/b.txt')
        else:
            self.a = a
            self.b = b
            
    def diameter_correction(self, d):
        correction = (d / 2) ** 2
        self.a *= correction
        self.b *= correction
        
    def crop2(self, hole_depth=1):
        i = int(hole_depth // 0.05) + 1
        self.a = self.a[:i, :i]
        self.b = self.b[:i, :i]
        self.c = mk_c(i)
        
        

class Strain:
    def __init__(self, path_file):
        """
        Strain object
        path_file: path to csv file with strain data, str
        """
        if isfile(path_file):
            data = pd.read_csv(path_file, sep='\t', decimal='.')
            for column in data.columns:
                if ('Depth' in column) or ('depth' in column):
                    self.depth = np.array(data[column])
                    print('Depth - OK', end=', ')
                elif 'e1' in column:
                    self.e1 = np.array(data[column])
                    print('E1 - OK', end=', ')
                elif 'e2' in column:
                    self.e2 = np.array(data[column])
                    print('E2 - OK', end=', ')
                elif 'e3' in column:
                    self.e3 = np.array(data[column])
                    print('E3 - OK')
        else:
            print("File isn't exist")
        self.iter = 0
        
            
    def __next__(self):
        ans = (self.e1, self.e2, self.e3)[self.iter]
        self.iter = (self.iter + 1) % 3
        return ans
    
  
    def __getitem__(self, index):
        index %= 3
        return (self.e1, self.e2, self.e3)[index]
    
        
    def interpolation(self, end=1, step=0.05, kind='poly', deg=7):
 
        depth = np.arange(step, end+step, step)
        strains = np.zeros((3, depth.shape[0]))
        
        if kind == 'poly':
            for i in range(3):
                interp = np.poly1d(np.polyfit(x=self.depth, y=next(self), deg=deg))
                strains[i] = interp(depth)
                
        elif kind == 'spline':
            for i in range(3):
                interp = make_interp_spline(x=self.depth, y=next(self), k=deg)
                strains[i] = interp(depth)
                
        elif kind == 'none':
            for i in range(3):
                interp = interp1d(x=self.depth, y=next(self), kind='zero')
                strains[i] = interp(depth)
           
        return depth, strains
    

    def principal_strain(self, hole_depth=None, depth_step=0.05, interpolation='poly', k=7, show_strains=True):
        """
        Calculate the three combination strain by interpolated data by
        ASTM E837 − 13a 10.1.2 
        
        hole_depth: depth of hole, float | None
        depth_step: hole depth step, use different from 0.05 if have custom calibration matrices, float
        interpolation: intepolation kind 'poly', 'spline' or 'none', str,
        k: degree of the fitting, int
        show_strains: shows figure of strain, bool
        
        """
        if hole_depth == None:
            hole_depth = np.max(self.depth)
        depth, strains =  self.interpolation(end=hole_depth, step=depth_step, kind=interpolation, deg=k)
        self.e_depth = depth
        self.e_strains = strains
        self.p = (strains[2] + strains[0]) / 2 * 1e-6
        self.q = (strains[2] - strains[0]) / 2 * 1e-6
        self.t = (strains[2] + strains[0] - 2*strains[1]) / 2 * 1e-6
        
        if show_strains:
            colors=('blue', 'green', 'red')
            labels=('E1', 'E2', 'E3')
            for e in range(3):
                plt.plot(self.depth, self[e], 'o', markersize=3, color=colors[e], label=f'{labels[e]}')
                plt.plot(depth, strains[e], color=colors[e], label=f'{labels[e]} interpolation')
            plt.grid()
            plt.xlabel('Hole depth, mm')
            plt.ylabel('Strain, um/m')
            plt.legend()

            
class ResidualStress:
    def __init__(self, strain: Strain, E:int, nu:float, coef: Coefficients, smoothing:bool=False):
        """
        Calculate uniform residual stress by Hole-Drilling Strain-Gage Method ASTM E837 − 13a
        use normal_stress or principal_stress to get stresses,
        report to get pivot table, show_stresses to get figures 
        
        strain: Strain object, Strain
        E: Young’s modulus, int
        nu: Poisson ratio, float
        coef: calibration matrices, np.array
        smoothing: Tikhonov Regularization flag, boolean
        """
        self.strain = strain
        coef.crop2(np.max(strain.depth))
        self.E = E
        self.nu = nu
        self.a = coef.a
        self.b = coef.b
        self.c = coef.c
        self.alpha_P = 1e-5
        self.alpha_Q = 1e-5
        self.alpha_T = 1e-5
        self.smoothing = smoothing


    def PQTstresses(self):
        """
        Calculate the three combination stresses P, Q and T by
        ASTM E837-13a 10.3.1 if smoothing is False
        ASTM E837-13a 10.3.3 if smoothing is True
        
        smoothing: Tikhonov Regularization flag, boolean
        return: combination stresses, tuplle
        """
        if self.smoothing:
            alpha_P_correction = np.linalg.inv(np.dot(self.a.T, self.a) + self.alpha_P * np.dot(self.c.T, self.c))
            alpha_Q_correction = np.linalg.inv(np.dot(self.b.T, self.b) + self.alpha_Q * np.dot(self.c.T, self.c))
            alpha_T_correction = np.linalg.inv(np.dot(self.b.T, self.b) + self.alpha_T * np.dot(self.c.T, self.c))
            P = self.E / (1+ self.nu) * np.dot(np.dot(self.a.T, self.strain.p), alpha_P_correction)
            Q = self.E * np.dot(np.dot(self.b.T, self.strain.q), alpha_Q_correction)
            T = self.E * np.dot(np.dot(self.b.T, self.strain.t), alpha_T_correction)
        else:
            P = (np.linalg.inv(self.a)).dot(self.strain.p) * self.E / (1+ self.nu)
            Q = (np.linalg.inv(self.b)).dot(self.strain.q) * self.E
            T = (np.linalg.inv(self.b)).dot(self.strain.t) * self.E
        return P, Q, T


    def normal_stress(self):
        """
        Calculate normal stresses 
        ASTM E837-13a 10.3.10
        """
        P, Q, T = self.PQTstresses()
        S_x = P - Q
        S_y = P + Q
        tau_xy = T
        return  S_x, S_y, tau_xy

    
    def principal_stress(self):
        """
        Calculate principal stresses 
        ASTM E837-13a 10.3.11
        """
        P, Q, T = self.PQTstresses()
        S_max = P + np.sqrt(Q**2 + T**2)
        S_min = P - np.sqrt(Q**2 + T**2)
        beta = np.arctan(T/Q)/2
        return S_max, S_min, beta
    
    
    def show_stresses(self, kind='principal'):
        """
        Shows figure of stresses
        """
        if kind == 'principal':
            labels = ('S1', 'S2')
            S_1, S_2, _ = self.principal_stress()
        elif kind == 'normal':
            labels = ('Sx', 'Sy')
            S_1, S_2, _ = self.normal_stress()
        plt.plot(self.strain.e_depth, S_1, label=labels[0])
        plt.plot(self.strain.e_depth, S_2, label=labels[1])
        plt.xlabel('Hole depth, mm')
        plt.ylabel('Stress, MPa')
        plt.grid()
        plt.legend()
        
    def get_misfit(self):
        """
        Return misfits between real and calculated strains
        ASTM E837-13a 10.3.6
        """
        P, Q, T = self.PQTstresses()
        p_misfit = self.strain.p - ((1+self.nu) / self.E) * np.dot(self.a, P)
        q_misfit = self.strain.q - (1 / self.E) * np.dot(self.b, Q)
        t_misfit = self.strain.t - (1 / self.E) * np.dot(self.b, T)
        return p_misfit, q_misfit, t_misfit
        
    def new_alpha(self):
        """
        Calculate new values of Tikhonov Regularization factors.
        ASTM E837-13a 10.3.8
        """
        
        p_misfit, q_misfit, t_misfit = self.get_misfit()
        p_std2, q_std2, t_std2 = map(std, (self.strain.p, self.strain.q, self.strain.t))
        p_rms2, q_rms2, t_rms2 = map(rms, (p_misfit, q_misfit, t_misfit))
        # print(p_rms2, q_rms2, t_rms2)
        criterion_p = p_std2/p_rms2
        criterion_q = q_std2/q_rms2
        criterion_t = t_std2/t_rms2
        max_criterion = np.abs(1 - np.min((criterion_p, criterion_p, criterion_t)))
        if max_criterion > 0.05:
            self.alpha_P *=  criterion_p
            self.alpha_Q *=  criterion_q
            self.alpha_T *=  criterion_t
        return max_criterion
            
        
    def tikhonov_correction(self):
        """
        Corect Tikhonov Regularization factors until  the 5 % criterion is obeyed.
        ASTM E837-13a 10.3.9
        """
        err = self.new_alpha()
        i = 0
        while err > 0.05 and i <= 100:
            err = self.new_alpha()
            i += 1
        print(f'Error: {err}, alpha P: {self.alpha_P}, alpha Q: {self.alpha_Q}, alpha T: {self.alpha_T}, iteration_count: {i}')
        
        
    def report(self):
        """
        Make report
        return: pandas dataframe
        """
        report = pd.DataFrame()
        
        report['Depth, mm'] = self.strain.e_depth
        report['E1, um/m'] = self.strain.e_strains[0]
        report['E2, um/m'] = self.strain.e_strains[1]
        report['E3, um/m'] = self.strain.e_strains[2]
        
        Sx, Sy, tau = map(lambda a: np.round(a, 1), self.normal_stress())
        report['Sx, MPa'] = Sx
        report['Sy, MPa'] = Sy
        report['tau_xy, MPa'] = tau
        
        S1, S2, beta = map(lambda a: np.round(a, 1), self.principal_stress())
        report['S1, MPa'] = S1
        report['S2, MPa'] = S2
        report['beta, deg'] = beta
        
        return report
    
    
    def save_report(self, filename:str):
        """
        Write report to a text file
        filename:  path, str
        """
        report = self.report()
        report.to_csv(filename, sep='\t', index=False)
        with open(filename, 'a') as file:
            file.write(f'\nE: {self.E} MPa, nu: {self.nu}\n')
            file.write(f'alpha P: {self.alpha_P}, alpha Q: {self.alpha_Q}, alpha T: {self.alpha_T}\n')
            
            
    
    