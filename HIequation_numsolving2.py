# 程序名：有限差分法解hosing耦合方程组, 1991年中解的形式拟合
# 作者：刘玉龙
# 时间：2022.9.5 18：45
# 地点：高能所
# ----------------------------------------------------------------------------------------------------------
# Numerical sovinghosing of HI coupled equation:
# ∂ξ^2(Xc)+CrCψω0^2(Xc)=ω0^2(Xb)
# ∂s^2(Xb)+ωβ^2(Xb)=ωβ^2(Xc)
# Two variables, Xc(ξ,s), Xb(ξ,s)

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

##################### define function #####################
def calcuXb(XI, nxi, ns, Ximax, Smax, gamma, hxi, hs):
    '''
    calculate all Xb[:, :]
    
    Parameters
    ----------
    :param XI: value at initial time, that is Xb[:, 0]
    :param nxi, ns: the number of meshes of ξ and S
    :param Ximax, Smax: the maximum value of ξ and S,from the beginning of 0
        
    :return:Xb[i+1, j+1]
    
    ----------
    the coupled hosing equation:
    ∂ξ^2(Xc)+CrCψω0^2(Xc)=ω0^2(Xb)
    ∂s^2(Xb)+ωβ^2(Xb)=ωβ^2(Xc)
    vanish Xc, only using Xb
    ##################### define function #####################
    def fxb_xiands(xb_iM1_j, xb_iM2_j, xb_i_jM1, xb_iM1_jM1, xb_iM2_jM1 , xb_i_jM2, xb_iM1_jM2, xb_iM2_jM2, Adeltxi2, Bdelts2):
    ∂ξ^2∂s^2(Xb)+k2/gamma* ∂ξ^2(Xb)+k1∂s^2(Xb)=0
    k1=CrCψ/2, k2=1/2    
    [1        , -2+A*Δξ^2          , 1        ]
    [-2+B*ΔS^2, 4-2*B*ΔS^2-2*A*Δξ^2, -2+B*ΔS^2]
    [1        , -2+A*Δξ^2          , 1        ]
    A = CrCψ/2, B = 1/(2*gamma)
    
    Parameters
    ----------
    :param Adeltksi2: A*Δξ^2
    :param Bdelts2: B*ΔS^2
    
    :return:Xb[i+1, j+1]
    
    ----------
    xb_iP1_jP1 = (2.-Adeltxi2)*xb_iM1_j - xb_iM2_j \
    		+ (2.-Bdelts2)*xb_i_jM1 - 2.*(2.-Bdelts2-Adeltxi2)*xb_iM1_jM1 + (2.-Bdelts2)*xb_iM2_jM1 \
    		- xb_i_jM2 + (2.-Adeltxi2)*xb_iM1_jM2 - xb_iM2_jM2
    
    return xb_iP1_jP1
    '''
    X = np.zeros((nxi+2, ns+2), dtype=float)  # define dtype for less running time
    twoMA_hxi2 = 2.-hxi*hxi/2.
    twoMB_hs2 = 2.-hs*hs/(2.*gamma)
    
    #initial value， xb(-Δξ, :), xb(0, :)
    for j in range(0, ns+2):
    	X[0, j] = 0.
    	X[1, j] = XI[0]
    	
    for i in range(2, nxi+2):
        X[i, 1] = XI[i-1]
        X[i, 0] = XI[i-1]
        for j in range(2, ns+2):
            X[i, j] = twoMA_hxi2*X[i-1, j] - X[i-2, j] \
                      + twoMB_hs2*X[i, j-1] - 2*(twoMB_hs2+twoMA_hxi2-2)*X[i-1, j-1] + twoMB_hs2*X[i-2, j-1] \
                      - X[i, j-2] + twoMA_hxi2*X[i-1, j-2] - X[i-2, j-2]
            #X[i, j] = fxb_xiands(X[i-1, j], X[i-2, j], X[i, j-1], X[i-1, j-1], X[i-2, j-1], X[i, j-2], X[i-1, j-2], X[i-2, j-2], A_hxi2, B_hs2)
    #print(X)
    xi_spread = np.linspace(-hxi, Ximax, nxi+2)
    s_spread = np.linspace(-hs, Smax, ns+2)
    return X, xi_spread, s_spread
    
def draw_s(Y, s, xi_const): #the value of i when setting constant ξ
    #get Xb[Ixi, :], constant ξ 
    plt.figure()
    plt.plot(s, Y, label='Numerical_constantξ') #X[:,m:n] #from m to n-1
    plt.title(label='$k_p\\xi={}$'.format(xi_const) , fontsize=19)
    plt.ylabel('$k_p X_b$', fontsize=19)
    plt.xlabel('$k_p s$', fontsize=19)
    plt.legend()
    plt.tight_layout()
    
def draw_xi(Y, xi, s_const):
    #get Xb[:, Js], constant S
    plt.figure()
    plt.plot(xi, Y, label='Numerical_constantS')
    plt.title(label='$k_pS={}$'.format(s_const), fontsize=19)
    plt.ylabel('$k_p X_b$', fontsize=19)
    plt.xlabel('$k_p \\xi$' , fontsize=19)
    plt.legend()
    plt.tight_layout()
    
def draw_3d(Y, ximin, ximax, smin, smax):
    plt.figure()
    plt.imshow(Xb,origin='lower',extent=[ximin, ximax, smin, smax],aspect='auto')
    plt.title(label='Xb', fontsize=19)
    plt.ylabel('$k_p S$', fontsize=19)
    plt.xlabel('$k_p \\xi$', fontsize=19)
    plt.legend()
    plt.tight_layout()

def Getpeak(Y, s, ds):
    '''
    get the peak of Xb and abs(Xb)
    [peaks] are the all Corresponding x of peakXb
    ds can be calculated by period omega_beta*1.5, but it is the number of grids
    '''
    peaks, _ = find_peaks(abs(Y), distance=ds) #返回的peaks是网格点数，在画图中要乘以步长变成实际距离
    
    # 画xb在ξ=8(c/wp)时随s的变化图象
    plt.plot(s[peaks], Y[peaks], "oy",
             label='peakdot')  # "b"为蓝色, "o"为圆点, ":"为点线 'b' 蓝色 'm' 洋红色 magenta 'g' 绿色 'y' 黄色'r' 红色 'k' 黑色 'w' 白色 'c' 青绿色 cyan
    plt.plot(s[peaks], abs(Y[peaks]), ".g",
             label='peakdot')  # "b"为蓝色, "o"为圆点, ":"为点线 'b' 蓝色 'm' 洋红色 magenta 'g' 绿色 'y' 黄色'r' 红色 'k' 黑色 'w' 白色 'c' 青绿色 cyan
    plt.tight_layout()
    return abs(Y[peaks]), s[peaks]
    
def peak_fitfuct(x, A, B):
    return np.log(x)/3. + np.power(B, -2./3.)*np.power(x, 2./3.) + A - 1/3.*np.log(B) #this is the formula in 1991 paper
    #print(-np.log(x)/6. + np.power(B, -.1/3.)*np.power(x, 1./3.) + A + 1/6.*np.log(B))
    #return -np.log(x)/6. + np.power(B, -.1/3.)*np.power(x, 1./3.) + A + 1/6.*np.log(B)
           
def peak_fit(Y_peak, s_peak, xi_const, init_para): 
    '''
    init_para can be calculated by theory
    '''
    #plt.figure()
    popt, pcov = curve_fit(peak_fitfuct, s_peak, Y_peak, p0=init_para) #注意将网格数转变成实际长度
    print('popt', popt)
    plt.plot(s_peak, peak_fitfuct(s_peak, *popt), 'r-', label='fit: A={}, Lg={}'.format(popt[0], popt[1]))
    #plt.plot(s_peak, peak_fitfuct(s_peak, init_para[0], init_para[1]), "y-", label='Theoretical value: A={}, Lg={}'.format(init_para[0], init_para[1]))
    plt.title(label='$k_p\\xi={}$'.format(xi_const) , fontsize=19)
    plt.ylabel('$k_p\\_peak Xb$', fontsize=19)
    plt.xlabel('$k_p\\_peak s$', fontsize=19)
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    #set some variabilities
    nxi = 5000
    ns = 700 
    Ximax = 1200.
    Smax = 8000. #depending on characteristic length
    hxi = Ximax/nxi #hxi, hs are steps. hxi=4/20=0.5
    hs = Smax/ns #Generally, at least, one period(300) has twenty grids(hs=300/20=15)
    gamma = 1000.
   
    #XI is one-dimensional
    XI = np.zeros((nxi+1), dtype=float)
    #XI[0] = 1e-4
    XI[1] = 1e-3
    
    #at here, get all the value of Xb, xi, s
    Xb, xi, s = calcuXb(XI, nxi, ns, Ximax, Smax, gamma, hxi, hs)
    
    #constant with xi(i=Ixi) while varying s; constant with s(j=Js) while varying xi;
    Ixi = -1 #i=257, ξ = (i-1)*hxi
    Js = -1  #j=701, s = (j-1)*hs
    
    #auto determine grids grids between two peaks(absY, that is half period)
    lambda_beta = 2.*np.pi*np.power(2.*gamma, 1/2) 
    ds = math.floor(lambda_beta/(2.*hs)) #Longitudinal grids between two peaks
    print('lambda$_beta', lambda_beta)
    print('ds', ds)    
    
    #get peaks of Xb(s) and abs(Xb(s))
    draw_s(Xb[Ixi, :], s, xi[Ixi])
    absY_peak, s_peak = Getpeak(Xb[Ixi, :], s, ds)
    
    #set initial value for fit parameter from theory
    A_init = np.log(2*np.power(3, -5/4)*np.power(np.pi, -1/2)/xi[Ixi])
    B_init = np.power(2, 15/4)*np.power(gamma, 1/2)*np.power(3, -9/4)*np.power(xi[Ixi], -1/2)
    init_para = [A_init, B_init]
    print('fit_init_para', init_para)
    
    for i in range(0, len(s_peak)):
        if(s_peak[i]>=6000):
            s_peakcut = i
            break
 
    #Fit function of logXb-S, and compared with theory
    draw_s(np.log(absY_peak),s_peak, xi[Ixi])
    peak_fit(np.log(absY_peak[s_peakcut:]), s_peak[s_peakcut:], xi[Ixi], init_para)
   
    draw_xi(Xb[:, Js], xi, s[Js])
    draw_3d(Xb, xi[0], xi[-1], s[0], s[-1])
    plt.show()








