import numpy as np
from scipy.integrate import RK45
from scipy import pi
import matplotlib.pyplot as plt

float_type = float
#Defining global constants
#Longitudinal force provided by the wakefield
f_z_ext = 0.
lam = 0.
#Transverse restoring parameter, that f_x = - K2*x
K2 = 0.5
K4 = K2*K2
K = K2**0.5
#Classical electron radius normalized to k_p^-1
re = 0
#Calculate re * K^4
re_K4 = re*K4
#Calculate 2/3 * re * K^4
two_over_three_re_K4 = 2./3.*re_K4
#Calculate 1/24 * re * K^4
re_K4_over_24 = re_K4/24.
#Calculate 1/3 * re * K^4
re_K4_over_3 = re_K4/3.

def Xc_integral(Xb, dxi):
    '''
        Calculate channel centroid Xc(xi) with given beam centriod Xb(xi) using integral
        Input:
        Xb: array of floats
            The xi distribution of beam centriod
        dxi: float
            Interval of xi
        Output:
        array of floats with the same shape as Xb
            Xc at the grid points being the same as Xb
    '''
    #Initialize complex integral
    I = np.zeros_like(Xb, dtype='complex')
    #Calculate I = integral_0^xi(Xb(xi')exp(iK(xi-xi'))dxi') using trapezoidal rule
    half_dxi = 0.5*dxi
    exp_iKdxi = np.exp(K*dxi*1j)
    for i in range(len(Xb)-1):
        I[i+1] = I[i]*exp_iKdxi + (Xb[i]*exp_iKdxi+Xb[i+1])*half_dxi
    return np.imag(I)*K

def Xc_cd(Xb, K2dxi2):
    '''
        Calculate channel centroid Xc(xi) with given beam centriod Xb(xi) using central difference scheme.
        Notice Xb[0]=0 is asserted, and non-zero value of Xb[0] is ignored, which guarentees Xc[1]=0 and Xc[0]=0.
        Input:
        Xb: array of floats
            The xi distribution of beam centriod
        K2dxi2: float
            K2*dxi*dxi, where dxi is the interval of xi
        Output:
        ndarray of floats with the same shape as Xb
            Xc at the grid points being the same as Xb
    '''
    two_m_K2dxi2 = 2. - K2dxi2
    Xc = np.zeros_like(Xb)
    for i in range(len(Xb)-2):
        Xc[i+2] = K2dxi2*Xb[i+1] - Xc[i] + two_m_K2dxi2*Xc[i+1]

    return Xc

def solveBORR(Xb0, gamma_init, s_max, xi_max, ns, nxi):
    '''
        Solving hosing with radiation reaction
        Input:
        Xb0: float
            The maximum offset of a linear tilt beam, with 0 velocity
        gamma_init: float
            Intial gamma
        s_max: float
            Maximum of s
        xi_max: float
            Maximum of xi
        ns: integer
            Number of s series
        nxi: integer
            Number of xi series
        Output:
        time: ndarray of floats
            Time array
        Xb, gamma: ndarray of floats
            Solution numpy arrais with shape of [ns,nxi]
            Xb and gamma, with [i,j] being at s = ds*i and xi = dxi*j
    '''
    #Calculate initial betatron frequency
    omega_beta=(K2/gamma_init)**0.5
    #Define xi series and dxi
    xi = np.linspace(0., xi_max, nxi)
    dxi = xi[1] - xi[0]
    #Define s series and ds
    s = np.linspace(0., s_max, ns)
    ds = s[1] - s[0]
    #Initialize
    Xb = np.zeros([ns, nxi], dtype=float_type)
    #At s = ds, Xb[1,:] are set to 0, and partial_Xb_partial_s are set to Xb*omega_beta for a correct betatron amplitude
    #Thus Xb[0,:] = -partial_Xb_partial_s*ds = -Xb*omega_beta*ds
    Xb[0,:] = -omega_beta*ds*np.linspace(0., Xb0, nxi)
    #gamma[1,:] are set to gamma_init, and gamma[0,:] = gamma_init-partial_gamma_partial_s*ds = gamma_init-ds*f_z_ext
    gamma = np.zeros([ns, nxi], dtype=float_type)
    gamma[1,:] = np.full(nxi, fill_value = gamma_init, dtype=float_type)
    gamma[0,:] = np.full(nxi, fill_value = gamma_init-ds*f_z_ext, dtype=float_type)
    
    K2dxi2 = K2*dxi*dxi
    two_ds = 2*ds
    ds_f_z_ext_over_4 = 0.25*f_z_ext*ds
    ds2K2_half = ds*ds*K2*0.5
    Xc = np.zeros((ns, nxi), dtype=float)
    for i in range(1, ns-1):
        Xc_minus_Xb = Xc_cd(Xb[i,:], K2dxi2) - Xb[i,:]
        Xc[i, :] = Xc_cd(Xb[i,:], K2dxi2)
        #Central difference scheme for partial derivative of s
        gamma[i+1,:] = (f_z_ext - two_over_three_re_K4 * (np.square(gamma[i,:]*Xc_minus_Xb)))*two_ds + gamma[i-1,:]
        ds_f_z_ext_over_4gamma = ds_f_z_ext_over_4/gamma[i,:]
        Xb[i+1,:] = (ds2K2_half*(Xc_minus_Xb/gamma[i,:]) + (ds_f_z_ext_over_4gamma - 0.5)*Xb[i-1,:] + Xb[i,:]) / (ds_f_z_ext_over_4gamma + 0.5)
    return Xb, gamma, xi, s

def solveBORR_plus(Xb0, gamma_init, s_max, xi_max, ns, nxi):
    '''
        Solving hosing with radiation reaction
        Input:
        Xb0: float
            The maximum offset of a linear tilt beam, with 0 velocity
        gamma_init: float
            Intial gamma
        s_max: float
            Maximum of s
        xi_max: float
            Maximum of xi
        ns: integer
            Number of s series
        nxi: integer
            Number of xi series
        Output:
        time: ndarray of floats
            Time array
        Xb, gamma: ndarray of floats
            Solution numpy arrais with shape of [ns,nxi]
            Xb and gamma, with [i,j] being at s = ds*i and xi = dxi*j
    '''
    #Calculate initial betatron frequency
    omega_beta=(K2/gamma_init)**0.5
    #Define xi series and dxi
    xi = np.linspace(0., xi_max, nxi)
    dxi = xi[1] - xi[0]
    #Define s series and ds
    s = np.linspace(0., s_max, ns)
    ds = s[1] - s[0]
    #Initialize
    Xb = np.zeros([ns, nxi], dtype=float_type)
    #At s = ds, Xb[1,:] are set to 0, and partial_Xb_partial_s are set to Xb*omega_beta for a correct betatron amplitude
    #Thus Xb[0,:] = -partial_Xb_partial_s*ds = -Xb*omega_beta*ds
    Xb[0,:] = -omega_beta*ds*np.linspace(0., Xb0, nxi)
    #gamma[1,:] are set to gamma_init, and gamma[0,:] = gamma_init-partial_gamma_partial_s*ds = gamma_init-ds*f_z_ext
    gamma = np.zeros([ns, nxi], dtype=float_type)
    gamma[1,:] = np.full(nxi, fill_value = gamma_init, dtype=float_type)
    gamma[0,:] = np.full(nxi, fill_value = gamma_init-ds*f_z_ext, dtype=float_type)
    
    K2dxi2 = K2*dxi*dxi
    K2ds2 = K2*ds*ds
    two_ds = 2*ds
    ds_f_z_ext = f_z_ext*ds
    ds2K2_half = ds*ds*K2*0.5
    Xc = np.zeros((ns, nxi), dtype=float)
    for i in range(1, ns-1):
        Xc_minus_Xb = Xc_cd(Xb[i,:], K2dxi2) - Xb[i,:]
        #Xc[i, :] = Xc_cd(Xb[i,:], K2dxi2)
        #Central difference scheme for partial derivative of s
        A = (K2*0.5-lam*0.25)*(Xc_minus_Xb)*(Xb[i, :]-Xb[i-1, :])/ds
        gamma[i+1,:] = (f_z_ext - two_over_three_re_K4*np.square(gamma[i,:]*Xc_minus_Xb) + A)*two_ds + gamma[i-1,:]
        B = -(K2*0.5-lam*0.25)*Xc_minus_Xb*(Xb[i, :]-Xb[i-1, :])**2/gamma[i, :]
        Xb[i+1,:] = K2ds2*(Xc_minus_Xb/gamma[i,:]) + (ds_f_z_ext/gamma[i, :]-1.)*Xb[i-1,:] - (ds_f_z_ext/gamma[i, :]-2.)*Xb[i,:] + B   
    return Xb, gamma
    
if '__main__'==__name__:
    #initial conditions
    Xb0=1e-3
    gamma_init=1e5
    omega_beta=(K2/gamma_init)**0.5
    print('lambda_beta = ', 2*pi/omega_beta)
    
    #Calculate the charactoristic time
    '''
    x1=4.
    LS=16./re/gamma_init/x1/x1
    print('LS = ', LS)
    print('f_z_rad2 = ', re_K4_over_3*gamma_init*gamma_init*x1*x1)
    '''
    
    s_max=1e4
    xi_max=1.
    ns=1024
    nxi=512
    print('lambda_beta/ds = ', 2*pi/omega_beta/(s_max/ns))
    print('lambda_0/dxi = ', 2*pi/K/(xi_max/nxi))
    Xb, gamma, xi, s = solveBORR(Xb0=Xb0, gamma_init=gamma_init, s_max=s_max, xi_max=xi_max, ns=ns, nxi=nxi)
    Xb_plus, gamma_plus= solveBORR_plus(Xb0, gamma_init, s_max, xi_max, ns, nxi)

    xi_ind = -1
    s_ind = -1
        
    plt.figure(figsize=[12,8])
    plt.subplot(321)
    plt.imshow(Xb,aspect='auto', origin='lower', extent=(0., xi_max, 0., s_max))
    plt.title('$X_b$')
    cbar = plt.colorbar()
    cbar.set_label('$X_b$')
    plt.xlabel('$\\xi$')
    plt.ylabel('$s$')
    plt.legend()
    
    plt.subplot(322)
    plt.imshow(gamma,aspect='auto', origin='lower', extent=(0., xi_max, 0., s_max))
    plt.title('$\\gamma$')
    cbar = plt.colorbar()
    cbar.set_label('$\\gamma$')
    plt.xlabel('$\\xi$')
    plt.ylabel('$s$')
    plt.legend()
    
    plt.subplot(323)
    plt.plot(s, Xb[:, xi_ind], 'r-', label='Xb-s')
    plt.plot(s, Xb_plus[:, xi_ind], 'g:', label='Xbplus-s')
    plt.title('$\\xi = {}$'.format(xi[xi_ind]))
    plt.xlabel('$s$')
    plt.ylabel('$X_b$')
    A = 3.**1.5/4. * ((omega_beta*s[1:])*(K*xi[xi_ind])**2)**(1./3.)
    #plt.plot(s[1:], 0.341*Xb0/A**1.5*np.exp(A)*np.sin(omega_beta*s[1:]-A/3**0.5+pi/4.), label = 'Asymptotic')
    plt.legend()
    
    plt.subplot(324)
    plt.plot(xi, Xb[s_ind,:], 'r-', label='Xb-$\\xi$')
    plt.plot(xi, Xb_plus[s_ind,:], 'g:', label='Xbplus-$\\xi$')
    plt.title('$s = {}$'.format(s[s_ind]))
    plt.xlabel('$\\xi$')
    plt.ylabel('$X_b$')
    plt.legend()
    
    plt.subplot(325)
    plt.plot(s, gamma[:, xi_ind], 'r-', label = '$\\gamma$-s')
    plt.plot(s, gamma_plus[:, xi_ind], 'g:', label = '$\\gamma$plus-s')
    plt.title('$\\xi = {}$'.format(xi[xi_ind]))
    plt.xlabel('$s$')
    plt.ylabel('$\\gamma$')
    A = 3.**1.5/4. * ((omega_beta*s[1:])*(K*xi[xi_ind])**2)**(1./3.)
    #plt.plot(s[1:], 0.341*Xb0/A**1.5*np.exp(A)*np.sin(omega_beta*s[1:]-A/3**0.5+pi/4.), label = 'Asymptotic')
    plt.legend()
    
    plt.subplot(326)
    plt.plot(xi, gamma[s_ind,:], 'r-', label='$\\gamma-\\xi$')
    plt.plot(xi, gamma_plus[s_ind,:], 'g:', label='$\\gamma$plus-$\\xi$')
    plt.title('$s = {}$'.format(s[s_ind]))
    plt.xlabel('$\\xi$')
    plt.ylabel('$\\gamma$')
    plt.legend()

    
    plt.figure(figsize=[12,8])
    plt.subplot(121)
    plt.plot(s, Xb_plus[:, xi_ind]-Xb[:, xi_ind], 'g-', label='Xb-Xbplus-s')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(s, gamma[:, xi_ind]-gamma_plus[:, xi_ind], 'g-', label='$\\gamma$-$\\gamma$plus-s')
    plt.plot(s, Xb_plus[:, xi_ind]**2*(K2*0.5-lam*0.5)*0.5, 'r-', label='theory')
    plt.legend()
   
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
