#程序名：有限差分法解hosing耦合方程组
#作者：刘玉龙
#时间：2022.9.3 22：37
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

# 所求hosing耦合方程为:
# ∂ξ^2(Xc)+CrCψω0^2(Xc)=ω0^2(Xb)
# ∂s^2(Xb)+ωβ^2(Xb)=ωβ^2(Xc)
# Xc(ξ,s), Xb(ξ,s), 两个变量两个未知数
# 先固定ξ（大循环），改变s(小循环)
def fxc_s(xb_splus1, xb_s, xb_sminus1, a):
    xc_splus1 = (xb_splus1 - 2*xb_s + xb_sminus1)*40000/a**2 + xb_s
    return xc_splus1

def fxb_ksi(xc_ksiplus1, xc_ksi, xc_ksiminus1, h):
    xb_ksiplus1 = (xc_ksiplus1 - 2*xc_ksi + xc_ksiminus1)*np.sqrt(2)/h**2 + xc_ksi
    return xb_ksiplus1

if __name__ == "__main__":
# 耦合方程：
# ∂ξ^2(Xc)+(1/√2)(Xc)=(1/√2)(Xb)
# ∂s^2(Xb)+(1/40000)(Xb)=(1/40000)(Xc)
# 其中np=2*10^16cm-1, gamma假设恒定为20Gev(CEPC注入束能量), ξ取值范围[0,20]、步长h=1, s取值范围[0,32000]、步长a=2000
# 给电子束一个小偏角Θ=1.96*10^3rad,tanΘ=1.7*10^-3
    nksi = 20
    ns = 16
    h = 20/nksi  #ξ的步长1，[0, 20]
    a = 32000/ns  #s的步长2000 [0, 32000]
    xc = np.zeros((nksi+1, ns+1))  #创建一个Xc(ξi, sj)的数组，最大值是[nksi, ns]
    xb = np.zeros((nksi+1, ns+1))  #创建一个Xb(ξi, sj)的数组，最大值是[nksi, ns]

    # 以下用迭代对Xc，Xb赋值
    # 赋初值Xb(0, s)=1.7*10^-3*s
    for j in range(0, ns-2):
        xb[0, j] = 1.7e-3*(h*nksi-h*j)

    # 赋初值Xc(0, s)=[Xb(0, s+a)-2*Xb(0, s)+Xb(0, s-h)]*40000/a^2+Xb(0, s)
    for j in range(0, ns-2):
        xc[0, j+1] = fxc_s(xb[0, j+2], xb[0, j+1], xb[0, j], a) #Xc(ξ0, s0)=0,xc从s1开始赋值的

####
####
#这里应添加终点处Xc[nξ, ns], Xb[nξ, ns]的值
####
####
    # 计算Xc,Xb,迭代
    for i in range(0, nksi-2):    #先控制ξ不变，小循环里变s
        for j in range(0, ns-2):  #一直求到[nksi-1, ns-1], 终点处[nksi, ns]应该直接给出，不能参与迭代,这样才能闭环
            xb[i+1, j+1] = fxb_ksi(xc[i+2, j+1], xc[i+1, j+1], xc[i, j+1], h)
            xc[i+1, j+1] = fxc_s(xb[i+1, j+2], xb[i+1, j+1], xb[i+1, j], a)

    # 画xc在ξ=8(c/wp)时随s的变化图象
    # 画xb在ξ=8(c/wp)时随s的变化图象
    xc_ksiequ8 = np.zeros((ns+1))
    for j in range(0, ns):
            xc_ksiequ8[j] = xc[8, j]
            #xb_ksiequ8[j] = xc[8, j]

    #画hosing曲线
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    s = np.linspace(0, 32000, ns+1) #定义样本点a，从0到1,间隔1/a, ns是网格数

    plt.plot(s, xc_ksiequ8, label='Theoretical')
    #plt.plot(s, xb_ksiequ8, label='Theoretical')
    plt.title('ξ=8[c/ωp]',fontsize=19)
    plt.ylabel('Xc[c/ωp]', fontsize=19)
    #plt.ylabel('Xb[c/ωp]', fontsize=19)
    plt.xlabel('z[c/ωp]', fontsize=19)
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
