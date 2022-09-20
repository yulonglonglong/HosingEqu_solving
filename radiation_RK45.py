# title：RK solve radiation dampling force ode
# name：Yulong
# time：2022.9.12 13：22
# address：ihep
# ----------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.integrate import odeint, RK45
import matplotlib.pyplot as plt

f = 1.e-3
xc = 0.
re = 1.e-10
k_square = 0.5
re_k4 = re*k_square*k_square

def func_xb(s, y):
    '''
        define the ordinary differential equations to be solved.
        y = [xb, gamma, dxb/ds]
    '''
    return [y[2], f-2./3.*re_k4*y[1]*y[1]*y[0]*y[0], -f/y[1]*y[2]-k_square/y[1]*y[0]]
    
def func_ampli(s, y):
    '''
        define the ordinary differential equations of amplitude to be solved.
        y = [gamma, Uxb]
    '''
    return[f-1./3.*re_k4*y[0]*y[0]*y[1]*y[1], -1./4.*f/y[0]*y[1]-1./24.*re_k4*y[0]*y[1]*y[1]*y[1]]
    
def solve_ode_RK45(func, t0, y0, t_bound, max_step, first_step):
    '''
                solve the ordinary differential equations defined by func, using RK45
    '''
    solver = RK45(func, t0=t0, y0=y0, t_bound=t_bound, max_step=max_step, first_step=first_step)
    t_list = []
    y_list = []
    while 'running' == solver.status:
        t_list.append(solver.t)
        y_list.append(solver.y)
        solver.step()
    Y = np.array(y_list)
    T = np.array(t_list)
    return T, Y

def plot_ode(x, y, axis_name, label_name, curve_style):
    '''
        plot xb-s, gamma-s
    '''
    plt.plot(x, y, curve_style, label='numsolving_{}'.format(label_name))
    plt.title('RKsolveode')
    plt.ylabel('{}'.format(axis_name))
    plt.xlabel('s[c/$\\omega$p]')
    plt.legend()

if __name__ == "__main__":  
    #y0=[xb0, gamma0, vxb0]
    a = 1.e-3    #amplitude gamma are big, f is small
    gamma0 = 2.e10
    v0 = 0.
    y0 = [a, gamma0, v0]
    
    #Ls, determain s_bound
    #s_bound = 16./(re*gamma0*a*a)
    s_bound = 3.e7
    
    #omega_beta, determain maxstep and first step
    omega_beta = np.power(k_square/gamma0, 1/2)
    max_step = 0.1/omega_beta
    first_step = 0.05/omega_beta
    
    s0=0
    print(s_bound, max_step, first_step)
    output_s, output_y = solve_ode_RK45(func_xb, s0, y0, s_bound, max_step, first_step)
    y0_amp = [gamma0, a]
    output_amp_s, output_amp_y = solve_ode_RK45(func_ampli, s0, y0_amp, s_bound, s_bound*1e-2, s_bound*1e-3)
    
    plt.figure()
    plot_ode(output_s, output_y[:, 0], 'Xb[c/$\\omega$pp]', 'Xb', 'g-')
    plot_ode(output_amp_s, output_amp_y[:, 1], 'Xb[c/$\\omega$p]', 'Uxb', 'r:')
    plt.figure()
    plot_ode(output_s, output_y[:, 1], '$\\gamma$', 'Xb', 'g-')
    plot_ode(output_amp_s, output_amp_y[:, 0], '$\\gamma$', 'Uxb', 'r:')
    
    
    print('t_xb', len(output_s))
    #print('t_xb', output_s)
    #print('y_xb', output_y)
    print('t_amp', len(output_amp_s))
    #print('t_amp', output_amp_s)
    #print('y_amp', output_amp_y)
    
    plt.show()

















