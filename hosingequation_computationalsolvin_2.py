#程序名：有限差分法解hosing耦合方程组2.0版本
#作者：刘玉龙
#时间：2022.9.5 18：45
#地点：高能所
# ----------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
import copy

# 所求hosing耦合方程为:
# ∂ξ^2(Xc)+CrCψω0^2(Xc)=ω0^2(Xb)
# ∂s^2(Xb)+ωβ^2(Xb)=ωβ^2(Xc)
# Xc(ξ,s), Xb(ξ,s), 两个变量两个未知数
# 将Xc带入Xb方程中消去
# 先固定ξ（大循环），改变s(小循环)

def fxb_ksiands(XL0, XL1, XL2, XC0, XC1, XC2, XR0, XR1, k1, k2, gamma, deltksi, delts):
#∂ξ^2∂s^2(Xb)+k2/gamma* ∂ξ^2(Xb)+k1∂s^2(Xb)=0
#其中k1=CrCψ/2, k2=1/2
    xb_ksiplus1_splus1 = -XL0 + (2-k2*delts**2/gamma)*XL1 - XL2 + (2-k1*deltksi**2)*XC0 - (4-2*k2*delts**2/gamma-2*k1*deltksi**2)*XC1 + (2-k1*deltksi**2)*XC2 - XR0 + (2-k2*delts**2/gamma)*XR1
    return xb_ksiplus1_splus1

def calcuXb_ksieql8(XbI, ns, Zmax):
    XL = np.zeros((ns), dtype=float)
    XC = np.zeros((ns), dtype=float)
    XR = np.zeros((ns), dtype=float)
    
    #为了避免大循环中出现ξ=-1和s=-1（因为不在所设数组之内，即溢出），需要提前算出ξ=1的全列Xb
    XR[0] = XbI[1] #Xb[1,0]
    XR[1] = fxb_ksiands(0, 0, 0, XbI[0], XbI[0], XbI[0], XbI[1], XbI[1], 1/2, 1/2, 20000, 0.1, Zmax/ns) #Xb[1,1]
    for j in range(2, ns-1): #Xb[1, j] j=2,...,ns-1
    	XR[j] = fxb_ksiands(0, 0, 0, XbI[0], XbI[0], XbI[0], XR[0], XR[1], 1/2, 1/2, 20000, 0.1, Zmax/ns) #至此,ξ=-1\0\1三列的Xb都已知
    	
    #但是还有s=1全行的Xb
    XL = XC.copy()
    XC = XR.copy()
    for i in range(2, 8):
    	#为了求s=1处的Xb，此为XR[1]，从i=2开始
    	#下面是为了确定s=-1全行的Xb
    	XC[0] = XbI[i-1]
    	XR[0] = XbI[i]
    	XL[0] = XbI[i-2]
    	XR[1] = fxb_ksiands(XL[0], XL[1], XL[2], XC[0], XC[1], XC[2], XR[0], XR[1], 1/2, 1/2, 20000, 0.1, 100)
    	#确定Xb[i,j],i=2,...,8, j=2,...,ns-1
    	for j in range(2,ns-1):
    		XR[j] = fxb_ksiands(XL[j-2], XL[j-1], XL[j], XC[j-2], XC[j-1], XC[0], XR[j-2], XR[j-1], 1/2, 1/2, 20000, 0.1, 100)
    		print(j)
    		print(XR[j])
    	XL = XC.copy()
    	XC = XR.copy()
    
    # 画xb在ξ=8(c/wp)时随s的变化图象
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    s = np.linspace(0, Zmax, ns) #定义样本点a，从0到1,间隔1/a, ns是网格数

    plt.plot(s, XR, label='Theoretical')
    plt.title('ξ=8[c/ωp]',fontsize=19)
    plt.ylabel('Xb[c/ωp]', fontsize=19)
    plt.xlabel('z[c/ωp]', fontsize=19)
    plt.legend()
    plt.show()
  

if __name__ == "__main__":
	a = 0.000000000096 #Xb微小扰动
	XbI = np.array([a, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
	ns = 100
	Zmax = 10000 #步长为100
	calcuXb_ksieql8(XbI, ns, Zmax)









