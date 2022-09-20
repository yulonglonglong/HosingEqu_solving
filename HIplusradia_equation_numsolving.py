# 程序名：有限差分法解hosing+radiation耦合方程组2.0版本
# 作者：刘玉龙
# 时间：2022.9.13 20：50
# 地点：高能所
# ----------------------------------------------------------------------------------------------------------
# Numerical sovinghosing of HI coupled equation:
# ∂ξ^2(Xc)+CrCψω0^2(Xc)=ω0^2(Xb)
# ∂s^2(Xb)+f/gamma∂s(Xb)+k^2/gamma(Xb)=k^2/gamma(Xc)
# ∂s(gamma)=f-2/3regamma^2k^4(xb-xc)^2
# Three variables, Xc(ξ,s), Xb(ξ,s), gamma(ξ,s)

import numpy as np
import matplotlib.pyplot as plt

##################### Global variables ####################
k2 = 0.5
w02 = 0.5
Cr = 1.
Cphi = 1.
re = 1.e-10
f = 1.e-3

##################### define function #####################

def numsolve(nxi, ns, Ximax, Smax, XbI, gamma0):
    '''
    calculate all Xb[:, :], Xc[:, :], gamma[:, :]
    
    Parameters
    ----------
    :param nxi, ns: the number of meshes of ξ and S in [0, max]
    :param hxi, hs: the step of ξ and S
    :param Ximax, Smax: the maximum value of ξ and S,from the beginning of 0
    :param XbI, XcI, gamma0: value at initial time, that is Xb[:, 1], Xc[1,:], gamma[:, 1]
    :param a: k^2*delts^2
    :param b: f*delts
    :param c: CrCphi*w0*deltxi^2
    :param d: 2/3*re*k^4*delts
    
    :return:Xb[i, j+1], Xc[i+1, j], gamma[i, j+1]
    
    ----------
    '''
    
    xi_spread = np.linspace(0., Ximax, nxi)
    s_spread = np.linspace(0., Smax, ns)
    hs = s_spread[1]-s_spread[0]
    hxi = xi_spread[1]-xi_spread[0]

    #define some coefficients
    k2ds2 = k2*hs*hs #k^2*delts^2, a
    fds = f*hs #f*delts, b
    #dxi need samll, ds need big
    #CrCphiW0dxi2 = 0
    CrCphiW0dxi2 = Cr*Cphi*w02*hxi*hxi #CrCphi*w0*deltxi^2, c
    co_rek4ds = 2./3.*re*k2*k2*hs #2/3*re*k^4*delts, co represent 2./3., d
    print(k2ds2, fds, CrCphiW0dxi2, co_rek4ds)
    
    #define variables
    Xb = np.zeros((nxi, ns), dtype=float)  # define dtype for less running time
    Xc = np.zeros((nxi, ns), dtype=float)
    gamma = np.zeros((nxi, ns), dtype=float)
    gamma[:, 1] = gamma0
    gamma[:, 0] = gamma0
    
    omegabeta_ds=(k2/gamma0)**0.5*hs
    for i in range(0, nxi):
        Xb[i, 0] = -omegabeta_ds*XbI[i]
        #Xb[i, 1] = XbI[i]
        
    for i in range(0, 2): 
        for j in range(2, ns):
            Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] - Xb[i, j-2] + (fds/gamma[i, j-1]+2.)*Xb[i, j-1])/(1.+fds/2./gamma[i, j-1])
            gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1]
           
    for i in range(2, nxi):
    	for j in range(2, ns):
    	    #Xb[i, j] = (2*k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] + 4*Xb[i, j-2] + (fds/gamma[i, j-1]-2)*Xb[i, j-1])/(2+fds/gamma[i, j-1])
    	    Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] - Xb[i, j-2] + (fds/gamma[i, j-1]+2)*Xb[i, j-1])/(1+fds/gamma[i, j-1])
    	    Xc[i, j] = CrCphiW0dxi2*Xb[i-1, j] + (2-CrCphiW0dxi2)*Xc[i-1, j] - Xc[i-2, j]
    	    gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1]  

    return Xb, gamma, xi_spread, s_spread
    
def draw_3d(Y, ximin, ximax, smin, smax, label):
        plt.imshow(Y, origin='lower', extent=[ximin, ximax, smin, smax],aspect='auto')
        plt.title(label='{}'.format(label), fontsize=19)
        plt.ylabel('$k_p S$', fontsize=19)
        plt.xlabel('$k_p \\xi$', fontsize=19)
        plt.tight_layout()

if __name__ == "__main__":
    #set some variabilities
    Ximax = 4.
    nxi = 512 #total grids
    gamma0 = 1.e5 #for bigger co_rek4dsgamma2 
    Smax = 1.e5
    ns = 1024
    
    Xb0 = 1.e-3
    XbI = np.linspace(0., Xb0, nxi)    #XbI are one-dimensional transverse
       
    #constant with xi(i=Ixi) while varying s; constant with s(j=Js) while varying xi;
    Ixi = -1 #i=257, ξ = (i-1)*hxi
    Js = -1
    
    #Generally, at least, one period(lambda_beta) has twenty grids(hs=300/20=15)
    L_hosing = np.power(2, 15/4)*np.power(gamma0, 1/2)*np.power(3, -9/4)*np.power(XbI[Ixi], -1/2)
    lambda_beta = 2.*np.pi*np.power(2.*gamma0, 1/2) #the period of S roughly = 0.4*sqrt(gamma0)
    L_radiation = 16./(re*gamma0*XbI[Ixi]*XbI[Ixi])
    print('XbI[Ixi]', XbI[Ixi])
    print('Lb_hosing', L_hosing)
    print('Lb_radiation', L_radiation)
   
    
    #at here, get all the value of Xb, xi, s
    Xb_fore, Gamma_fore, xi_fore, s_fore = numsolve(nxi, ns, Ximax, Smax, XbI, gamma0)
    
    #draw plot
    plt.figure(figsize=[12,8])
    plt.subplot(321)
    plt.plot(s_fore, Xb_fore[Ixi, :], label='Xb-s')
    plt.title(label='$\\xi$={}'.format(xi_fore[Ixi]))
    plt.ylabel('Xb')
    plt.xlabel('s')
    plt.legend()
    
    plt.subplot(322)
    plt.plot(s_fore, Gamma_fore[Ixi, :], label='$\\gamma$-s')
    plt.title(label='$\\xi$={}'.format(xi_fore[Ixi]))
    plt.ylabel('$\\gamma$')
    plt.xlabel('s')
    plt.legend()

    plt.subplot(323)
    plt.plot(xi_fore, Xb_fore[:, Js], label='Xb-$\\xi$')
    plt.title(label='s={}'.format(s_fore[Js]))
    plt.ylabel('Xb')
    plt.xlabel('$\\xi$')
    plt.legend()
    
    plt.subplot(324)
    plt.plot(xi_fore, Gamma_fore[:, Js], label='$\\gamma-\\xi$')
    plt.title(label='s={}'.format(s_fore[Js]))
    plt.ylabel('$\\gamma$')
    plt.xlabel('$\\xi$')
    plt.legend()
    
    plt.subplot(325)
    draw_3d(Xb_fore, 0, Ximax, 0, Smax, 'Xb')
    
    plt.subplot(326)
    draw_3d(Gamma_fore, 0, Ximax, 0, Smax, '$\\gamma$')
    
    plt.tight_layout()
    plt.show()

    
























