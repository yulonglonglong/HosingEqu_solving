# 程序名：有限差分法解hosing+radiation耦合方程组2.0版本，加上额外项
# 作者：刘玉龙
# 时间：2022.9.13 20：50
# 地点：高能所
# ----------------------------------------------------------------------------------------------------------
# Numerical sovinghosing of HI coupled equation:
# ∂ξ^2(Xc)+CrCψω0^2(Xc)=ω0^2(Xb)
# ∂s^2(Xb)+f/gamma∂s(Xb)+k^2/gamma(Xb)=k^2/gamma(Xc)
# ∂s(gamma)=f-2/3regamma^2k^4(xb-xc)^2
# Three variables, Xc(ξ,s), Xb(ξ,s), gamma(ξ,s)
'''
#Xb[i, j] = (fds/gamma[i, j-1]+1 - ((fds/gamma[i, j-1]+1)**2 + 4*(k2ds2/gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1])/hs)**2 + 4*k2/gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1])*(Xb[i, j-2]-Xb[i, j-1]))**(1/2))/(2*k2ds2*(Xb[i, j-1]-Xc[i, j-1])/gamma[i, j-1]/hs/hs) + Xb[i, j-1] #forward -b(+/-)sqrt(b2-4ac), the first blank
    	    #Xb[i, j] = k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] + (2-fds/gamma[i, j-1])*Xb[i, j-1] - (1-fds/gamma[i, j-1])*Xb[i, j-2] + k2ds2/gamma[i, j-1]*(Xc[i, j-1]-Xb[i, j-1])*((Xb[i, j-1]-Xb[i, j-2])/hs)**2 #backward
    	    Xb[i, j] = (fds/gamma[i, j-1]+1 - ((fds/gamma[i, j-1]+1)**2 + (k2ds2/gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1])/hs)**2 + 2*k2/gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1])*(Xb[i, j-2]-Xb[i, j-1]))**(1/2))/(k2*(Xb[i, j-1]-Xc[i, j-1])/gamma[i, j-1]/2) + Xb[i, j-1] #center -b(+/-)sqrt(b2-4ac), the first blank
    	    Xc[i, j] = CrCphiW0dxi2*Xb[i-1, j] + (2-CrCphiW0dxi2)*Xc[i-1, j] - Xc[i-2, j]
    	    #gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1] - k2*(Xb[i, j]-Xb[i, j-1])*(Xb[i, j-1]-Xc[i, j-1]) #forward
    	    #gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1] - k2*(Xb[i, j-1]-Xb[i, j-2])*(Xb[i, j-1]-Xc[i, j-1]) #backward
            gamma[i, j] = 2*fds - 2*co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1] - k2*(Xb[i, j]-Xb[i, j-2])*(Xb[i, j-1]-Xc[i, j-1]) #center
'''
import numpy as np
import matplotlib.pyplot as plt

##################### Global variables ####################
lam = 0.
k2 = 0.5
w02 = 0.5
Cr = 1.
Cphi = 1.
re = 1e-10
f = 0.

##################### define function #####################
def numsolve_fore(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds): 
    '''
    forward
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
    #define variables
    Xb = np.zeros((nxi, ns), dtype=float)  # define dtype for less running time
    Xc = np.zeros((nxi, ns), dtype=float)
    gamma = np.zeros((nxi, ns), dtype=float)
    gamma[:, 1] = gamma0
    gamma[:, 0] = gamma0
    
    for i in range(0, nxi):
        Xb[i, 0] = -omegabeta_ds*XbI[i]
        #Xb[i, 1] = XbI[i]
        
    for i in range(0, 2): 
        for j in range(2, ns):
            Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] - Xb[i, j-2] + (fds/gamma[i, j-1]+2.)*Xb[i, j-1])/(1.+fds/gamma[i, j-1])
            gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1]
           
    for i in range(2, nxi):
    	for j in range(2, ns):
    	    Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] - Xb[i, j-2] + (fds/gamma[i, j-1]+2.)*Xb[i, j-1])/(1.+fds/gamma[i, j-1])
    	    Xc[i, j] = CrCphiW0dxi2*Xb[i-1, j] + (2-CrCphiW0dxi2)*Xc[i-1, j] - Xc[i-2, j]
    	    gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1]  

    return Xb, gamma
    
def numsolve_back(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds): 
    '''
    forward
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
    #define variables
    Xb = np.zeros((nxi, ns), dtype=float)  # define dtype for less running time
    Xc = np.zeros((nxi, ns), dtype=float)
    gamma = np.zeros((nxi, ns), dtype=float)
    gamma[:, 1] = gamma0
    gamma[:, 0] = gamma0
    
    for i in range(0, nxi):
        Xb[i, 0] = -omegabeta_ds*XbI[i]
        #Xb[i, 1] = XbI[i]
        
    for i in range(0, 2): 
        for j in range(2, ns):
            Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] - (fds/gamma[i, j-1]-2.)*Xb[i, j-1] + (fds/gamma[i, j-1]-1.)*Xb[i, j-2])
            gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1]  
           
    for i in range(2, nxi):
    	for j in range(2, ns):
    	    Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] - (fds/gamma[i, j-1]-2.)*Xb[i, j-1] + (fds/gamma[i, j-1]-1.)*Xb[i, j-2])
    	    Xc[i, j] = CrCphiW0dxi2*Xb[i-1, j] + (2-CrCphiW0dxi2)*Xc[i-1, j] - Xc[i-2, j]
    	    gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1]  

    return Xb, gamma

def numsolve_cent(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds): 
    '''
    forward
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
    #define variables
    Xb = np.zeros((nxi, ns), dtype=float)  # define dtype for less running time
    Xc = np.zeros((nxi, ns), dtype=float)
    gamma = np.zeros((nxi, ns), dtype=float)
    gamma[:, 1] = gamma0
    gamma[:, 0] = gamma0
    
    for i in range(0, nxi):
        Xb[i, 0] = -omegabeta_ds*XbI[i]
        #Xb[i, 1] = XbI[i]
        
    for i in range(0, 2): 
        for j in range(2, ns):
            Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] + 2*Xb[i, j-1] + (fds/gamma[i, j-1]/2.-1.)*Xb[i, j-2])/(1+fds/gamma[i, j-1]/2.)
            gamma[i, j] = 2*fds - 2*co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-2]
           
    for i in range(2, nxi):
    	for j in range(2, ns):
    	    Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] + 2*Xb[i, j-1] + (fds/gamma[i, j-1]/2.-1.)*Xb[i, j-2])/(1+fds/gamma[i, j-1]/2.)
    	    Xc[i, j] = CrCphiW0dxi2*Xb[i-1, j] + (2-CrCphiW0dxi2)*Xc[i-1, j] - Xc[i-2, j]
    	    gamma[i, j] = 2*fds - 2*co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-2]  

    return Xb, gamma

def numsolve1_fore(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds): 
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
    #define variables
    Xb = np.zeros((nxi, ns), dtype=float)  # define dtype for less running time
    Xc = np.zeros((nxi, ns), dtype=float)
    gamma = np.zeros((nxi, ns), dtype=float)
    gamma[:, 1] = gamma0
    gamma[:, 0] = gamma0
    
    for i in range(0, nxi):
        Xb[i, 0] = -omegabeta_ds*XbI[i]
        #Xb[i, 1] = XbI[i]
        
    for i in range(0, 2): 
        for j in range(2, ns):
            A = k2/gamma[i, j-2]*(Xb[i, j-2]-Xc[i, j-2])*(Xb[i, j-1]-Xb[i, j-2])**2*(k2*0.5-lam*0.25)
            Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] - Xb[i, j-2] + (fds/gamma[i, j-1]+2.)*Xb[i, j-1] + A)/(1+fds/gamma[i, j-1]) 
            gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1] - k2*(Xb[i, j-1]-Xc[i, j-1])*(Xb[i, j]-Xb[i, j-1])*(k2*0.5-lam*0.25)
           
    for i in range(2, nxi):
    	for j in range(2, ns):
    	    A = k2/gamma[i, j-2]*(Xb[i, j-2]-Xc[i, j-2])*(Xb[i, j-1]-Xb[i, j-2])**2*(k2*0.5-lam*0.25)
    	    Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] - Xb[i, j-2] + (fds/gamma[i, j-1]+2.)*Xb[i, j-1] + A)/(1+fds/gamma[i, j-1]) #forward -b(+/-)sqrt(b2-4ac), the first blank
    	    Xc[i, j] = CrCphiW0dxi2*Xb[i-1, j] + (2-CrCphiW0dxi2)*Xc[i-1, j] - Xc[i-2, j]
    	    gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1] - k2*(Xb[i, j-1]-Xc[i, j-1])*(Xb[i, j]-Xb[i, j-1])*(k2*0.5-lam*0.25) #forward
    return Xb, gamma
    
def numsolve1_back(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds):  
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
    #define variables
    Xb = np.zeros((nxi, ns), dtype=float)  # define dtype for less running time
    Xc = np.zeros((nxi, ns), dtype=float)
    gamma = np.zeros((nxi, ns), dtype=float)
    gamma[:, 1] = gamma0
    gamma[:, 0] = gamma0
    
    for i in range(0, nxi):
        Xb[i, 0] = -omegabeta_ds*XbI[i]
        #Xb[i, 1] = XbI[i]
        
    for i in range(0, 2): 
        for j in range(2, ns):
            Xb[i, j] = k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] + (2.-fds/gamma[i, j-1])*Xb[i, j-1] - (1.-fds/gamma[i, j-1])*Xb[i, j-2] + k2/gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1])*(Xb[i, j-1]-Xb[i, j-2])**2*(k2*0.5-lam*0.25)
            gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1] - k2*(Xb[i, j-1]-Xb[i, j-2])*(Xb[i, j-1]-Xc[i, j-1])*(k2*0.5-lam*0.25)
           
    for i in range(2, nxi):
    	for j in range(2, ns):
    	    Xb[i, j] = k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] + (2.-fds/gamma[i, j-1])*Xb[i, j-1] - (1.-fds/gamma[i, j-1])*Xb[i, j-2] + k2/gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1])*(Xb[i, j-1]-Xb[i, j-2])**2*(k2*0.5-lam*0.25) #backward
    	    Xc[i, j] = CrCphiW0dxi2*Xb[i-1, j] + (2-CrCphiW0dxi2)*Xc[i-1, j] - Xc[i-2, j]
    	    gamma[i, j] = fds - co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-1] - k2*(Xb[i, j-1]-Xb[i, j-2])*(Xb[i, j-1]-Xc[i, j-1])*(k2*0.5-lam*0.25) #backward
    return Xb, gamma
    
def numsolve1_cent(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds): 
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
    #define variables
    Xb = np.zeros((nxi, ns), dtype=float)  # define dtype for less running time
    Xc = np.zeros((nxi, ns), dtype=float)
    gamma = np.zeros((nxi, ns), dtype=float)
    gamma[:, 1] = gamma0
    gamma[:, 0] = gamma0
   
    for i in range(0, nxi):
        Xb[i, 0] = -omegabeta_ds*XbI[i]
        #Xb[i, 1] = XbI[i]
        
    for i in range(0, 2): 
        for j in range(2, ns):
            if j==2:
                A = 0
            else:
                A = k2/gamma[i, j-2]*(Xb[i, j-2]-Xc[i, j-2])*(Xb[i, j-1]-Xb[i, j-3])**2/4.*(k2*0.5-lam*0.25)
            Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] + 2*Xb[i, j-1] + (fds/gamma[i, j-1]/2.-1.)*Xb[i, j-2] + A)/(1+fds/gamma[i, j-1]/2.)
            gamma[i, j] = 2*fds - 2*co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-2] - k2*(Xb[i, j-1]-Xc[i, j-1])*(Xb[i, j]-Xb[i, j-2])*(k2*0.5-lam*0.25)
            
    for i in range(2, nxi):
    	for j in range(2, ns):
            if j==2:
                A = 0
            else:
                A = k2/4./gamma[i, j-2]*(Xb[i, j-2]-Xc[i, j-2])*(Xb[i, j-1]-Xb[i, j-3])**2/4.*(k2*0.5-lam*0.25)
            Xb[i, j] = (k2ds2*(Xc[i, j-1]-Xb[i, j-1])/gamma[i, j-1] + 2*Xb[i, j-1] + (fds/gamma[i, j-1]/2.-1.)*Xb[i, j-2] + A)/(1+fds/gamma[i, j-1]/2.) #center -b(+/-)sqrt(b2-4ac), the first blank
            Xc[i, j] = CrCphiW0dxi2*Xb[i-1, j] + (2-CrCphiW0dxi2)*Xc[i-1, j] - Xc[i-2, j]
            gamma[i, j] = 2*fds - 2*co_rek4ds*(gamma[i, j-1]*(Xb[i, j-1]-Xc[i, j-1]))**2 + gamma[i, j-2] - k2*(Xb[i, j-1]-Xc[i, j-1])*(Xb[i, j]-Xb[i, j-2])*(k2*0.5-lam*0.25) #center
    return Xb, gamma
    
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
    omegabeta_ds=(k2/gamma0)**0.5*hs
    print(k2ds2, fds, CrCphiW0dxi2, co_rek4ds)
    

   
    #at here, get all the value of Xb, xi, s
    Xb_fore, Gamma_fore = numsolve_fore(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds)
    Xb_back, Gamma_back = numsolve_back(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds)
    Xb_cent, Gamma_cent = numsolve_cent(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds)
    Xb1_fore, Gamma1_fore = numsolve1_fore(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds)
    Xb1_back, Gamma1_back = numsolve1_back(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds)
    Xb1_cent, Gamma1_cent = numsolve1_cent(nxi, ns, k2ds2, fds, CrCphiW0dxi2, co_rek4ds, omegabeta_ds)
    
    #draw plot
    plt.figure(figsize=[12,8])
    plt.subplot(321)
    
    plt.plot(s_spread, Xb_fore[Ixi, :], 'r-', label='Xb-s')
    plt.plot(s_spread, Xb1_fore[Ixi, :], 'g:', label='Xb1_fore-s')
    plt.plot(s_spread, Xb1_back[Ixi, :], 'y-.', label='Xb1_back-s')
    plt.plot(s_spread, Xb1_cent[Ixi, :], 'b--', label='Xb1_cent-s')
    plt.title(label='$\\xi$={}'.format(xi_spread[Ixi]))
    plt.ylabel('Xb')
    plt.xlabel('s')
    plt.legend()
    
    plt.subplot(322)
    plt.plot(s_spread, Gamma_fore[Ixi, :], 'r-', label='$\\gamma$-s')
    plt.plot(s_spread, Gamma1_fore[Ixi, :], 'g:', label='$\\gamma$1_fore-s')
    plt.plot(s_spread, Gamma1_back[Ixi, :], 'y-.', label='$\\gamma$1_back-s')
    plt.plot(s_spread, Gamma1_cent[Ixi, :], 'b--', label='$\\gamma$1_cent-s')
    plt.title(label='$\\xi$={}'.format(xi_spread[Ixi]))
    plt.ylabel('$\\gamma$')
    plt.xlabel('s')
    plt.legend()

    plt.subplot(323)
    plt.plot(xi_spread, Xb_fore[:, Js],  'r-', label='Xb-$\\xi$')
    plt.plot(xi_spread, Xb1_fore[:, Js], 'g:', label='Xb1_fore-$\\xi$')
    plt.plot(xi_spread, Xb1_back[:, Js], 'y-.', label='Xb1_back-$\\xi$')
    plt.plot(xi_spread, Xb1_cent[:, Js], 'b--', label='Xb1_cent-$\\xi$')
    plt.title(label='s={}'.format(s_spread[Js]))
    plt.ylabel('Xb')
    plt.xlabel('$\\xi$')
    plt.legend()
    
    plt.subplot(324)
    plt.plot(xi_spread, Gamma_fore[:, Js], 'r-', label='$\\gamma-\\xi$')
    plt.plot(xi_spread, Gamma1_fore[:, Js], 'g:', label='$\\gamma$1_fore-$\\xi$')
    plt.plot(xi_spread, Gamma1_back[:, Js], 'y-.', label='$\\gamma$1_back-$\\xi$')
    plt.plot(xi_spread, Gamma1_cent[:, Js], 'b--', label='$\\gamma$1_cent-$\\xi$')
    plt.title(label='s={}'.format(s_spread[Js]))
    plt.ylabel('$\\gamma$')
    plt.xlabel('$\\xi$')
    plt.legend()
    
    plt.subplot(325)
    draw_3d(Xb_fore, 0, Ximax, 0, Smax, 'Xb')
    
    plt.subplot(326)
    draw_3d(Gamma_fore, 0, Ximax, 0, Smax, '$\\gamma$')
    
    plt.figure(figsize=[12,8])
    plt.subplot(121)
    plt.plot(s_spread, Xb_fore[Ixi, :]-Xb1_fore[Ixi, :], 'g-', label='Xb_fore-Xb1_fore-s')
    plt.plot(s_spread, Xb_back[Ixi, :]-Xb1_back[Ixi, :], 'y-', label='Xb_back-Xb1_back-s')
    plt.plot(s_spread, Xb_cent[Ixi, :]-Xb1_cent[Ixi, :], 'b-', label='Xb_cent-Xb1_cent-s')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(s_spread, Gamma_fore[Ixi, :]-Gamma1_fore[Ixi, :], 'g-', label='$\\gamma$_fore-$\\gamma$1_fore-s')
    plt.plot(s_spread, Xb_fore[Ixi, :]**2*(k2*0.5-lam*0.25)*0.25*(-1), 'r-', label='$\\gamma$_fore-$\\gamma$1_fore-s')
    plt.plot(s_spread, Gamma_back[Ixi, :]-Gamma1_back[Ixi, :], 'y-', label='$\\gamma$_back-$\\gamma$1_back-s')
    plt.plot(s_spread, Gamma_cent[Ixi, :]-Gamma1_cent[Ixi, :], 'b-', label='$\\gamma$_cent-$\\gamma$1_cent-s')
    plt.legend()
    
    
    plt.tight_layout()
    plt.show()

    
























