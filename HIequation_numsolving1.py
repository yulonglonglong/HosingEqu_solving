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
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# 所求hosing耦合方程为:
# ∂ξ^2(Xc)+CrCψω0^2(Xc)=ω0^2(Xb)
# ∂s^2(Xb)+ωβ^2(Xb)=ωβ^2(Xc)
# Xc(ξ,s), Xb(ξ,s), 两个变量两个未知数
# 将Xc带入Xb方程中消去
# 先固定ξ（大循环），改变s(小循环)

#峰值拟合函数形式为Xb=2/[3^(5/4)*pi^(1/2)*ξ]*(s/Lg)^(1/3)*e^((s/Lg)^(2/3))
#其中Lg=2^(15/4)*gamma^(1/2)/[3^(9/4)*ξ^(1/4)*Kp]*b
def peak_fit(x, A, B):
  return A * np.power(x/B, 1/3) * np.exp(np.power(x/B, 2/3))
  
def fxb_ksiands(XL0, XL1, XL2, XC0, XC1, XC2, XR0, XR1, k1, k2, gamma, deltksi, delts):
#∂ξ^2∂s^2(Xb)+k2/gamma* ∂ξ^2(Xb)+k1∂s^2(Xb)=0
#其中k1=CrCψ/2, k2=1/2， ds是取极值横轴范围
	xb_ksiplus1_splus1 = -XL0 + (2-k2*delts**2/gamma)*XL1 - XL2 + (2-k1*deltksi**2)*XC0 - (4-2*k2*delts**2/gamma-2*k1*deltksi**2)*XC1 + (2-k1*deltksi**2)*XC2 - XR0 + (2-k2*delts**2/gamma)*XR1
	return xb_ksiplus1_splus1

def calcuXb_ksieql8(XbI, ns, Zmax, k1, k2, gamma, deltksi, delts, ds):
	XL = np.zeros((ns), dtype=float) #若不规定类型，则开始默认整数，传入一个小数后全部转变成小数类型，占用运行时间
	XC = np.zeros((ns), dtype=float)
	XR = np.zeros((ns), dtype=float)
    
	#为了避免大循环中出现ξ=-1和s=-1（因为不在所设数组之内，即溢出），需要提前算出ξ=1的全列Xb
	XR[0] = XbI[1] #Xb[1,0]
	XR[1] = fxb_ksiands(0, 0, 0, XbI[0], XbI[0], XbI[0], XbI[1], XbI[1], k1, k2, gamma, deltksi, Zmax/ns) #Xb[1,1]
	for j in range(2, ns-1): #Xb[1, j] j=2,...,ns-1
		XR[j] = fxb_ksiands(0, 0, 0, XbI[0], XbI[0], XbI[0], XR[0], XR[1], k1, k2, gamma, deltksi, Zmax/ns) #至此,ξ=-1\0\1三列的Xb都已知
    	
	#但是还有s=1全行的Xb
	XL = XC.copy()
	XC = XR.copy()
	for i in range(2, 8):
    		#为了求s=1处的Xb，此为XR[1]，从i=2开始
    		#下面是为了确定s=-1全行的Xb
    		XC[0] = XbI[i-1]
    		XR[0] = XbI[i]
    		XL[0] = XbI[i-2]
    		XR[1] = fxb_ksiands(XL[0], XL[1], XL[2], XC[0], XC[1], XC[2], XR[0], XR[1], k1, k2, gamma, deltksi, Zmax/ns)
    		#确定Xb[i,j],i=2,...,8, j=2,...,ns-1
    		for j in range(2,ns-1):
    			XR[j] = fxb_ksiands(XL[j-2], XL[j-1], XL[j], XC[j-2], XC[j-1], XC[0], XR[j-2], XR[j-1], k1, k2, gamma, deltksi, Zmax/ns)
    			#print(j)
    			#print(XR[j])
    		XL = XC.copy() #若不用.copy(),默认传地址，等号两边两个变量一起变
    		XC = XR.copy()
    		
    	#取各小范围极值
	print(XC)
	peaks, _ = find_peaks(abs(XC), distance=ds) #返回的peaks是网格点数，在画图中要乘以步长变成实际距离
	print('peak_id', peaks*delts)
	print('peak_value', XC[peaks])
	print('peak_value', abs(XC[peaks]))
	
	# 画xb在ξ=8(c/wp)时随s的变化图象
	#plt.rcParams['font.sans-serif'] = ['SimHei']
	#plt.rcParams['axes.unicode_minus'] = False
	s = np.linspace(0, Zmax, ns) #定义样本点a，从0到1,间隔1/a, ns是网格数

	plt.plot(s, XC, label='Theoretical')
	plt.plot(peaks*delts, XC[peaks], "oy", label='peakdot')#"b"为蓝色, "o"为圆点, ":"为点线 'b' 蓝色 'm' 洋红色 magenta 'g' 绿色 'y' 黄色'r' 红色 'k' 黑色 'w' 白色 'c' 青绿色 cyan
	plt.plot(peaks*delts, abs(XC[peaks]), ".g", label='peakdot')#"b"为蓝色, "o"为圆点, ":"为点线 'b' 蓝色 'm' 洋红色 magenta 'g' 绿色 'y' 黄色'r' 红色 'k' 黑色 'w' 白色 'c' 青绿色 cyan
	
	'''#plt.plot(s, peak_fit(s, 0.0000007, 20), 'r-', label='fit')
	popt, pcov = curve_fit(peak_fit, peaks*delts, abs(XC[peaks])) #注意将网格数转变成实际长度
	print(popt)
	plt.plot(peaks*delts, peak_fit(peaks*delts, *popt), 'r-', label='fit: A=%5.3f, Lg=%5.3f' % tuple(popt))'''
		
	plt.title('ξ=8[c/ωp]',fontsize=19)
	plt.ylabel('Xb[c/ωp]', fontsize=19)
	plt.xlabel('z[c/ωp]', fontsize=19)
	plt.legend()
	plt.show()

if __name__ == "__main__":
	#设置一些初始量
	a = 0.0000000098 #Xb微小扰动
	XbI = np.array([a, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
	ns = 500
	Zmax = 10000 
	h = Zmax/ns #s步长20
	k1 = 1/2
	k2 = 1/2
	gamma = 20000
	deltksi = 0.1
	distance = 4 #相邻峰之间的最小水平距离，这里是网格点数，不是实际长度
	
	#输出ξ=8时的Xb
	calcuXb_ksieql8(XbI, ns, Zmax, k1, k2, gamma, deltksi, h, distance)
	










