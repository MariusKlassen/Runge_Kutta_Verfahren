import numpy as np 
import matplotlib.pyplot as plt


h=0.025
T=35
epsilon_1=2.0
epsilon_2=0.8
gamma_1=0.02
gamma_2=0.0002
ts=np.arange(0,T+h,h)
x=np.zeros(int(T/h)+1)
y=np.zeros(int(T/h)+1)
x[0]=40
y[0]=40


def f_xpunkt(x,y,epsilon_1,epsilon_2,gamma_1,gamma_2):
    return x*(epsilon_1-gamma_1*y)

def g_ypunkt(x,y,epsilon_1,epsilon_2,gamma_1,gamma_2):
    return -y*(epsilon_2-gamma_2*x)


def Runge_Kutta(x,y,parameter,h):
    
    k_1_x=f_xpunkt(x,y,*parameter)
    l_1_y=g_ypunkt(x,y,*parameter)
    
    k_2_x=f_xpunkt(x+h/2*k_1_x,y+h/2*l_1_y,*parameter)
    l_2_y=g_ypunkt(x+h/2*k_1_x,y+h/2*l_1_y,*parameter)
    
    k_3_x=f_xpunkt(x+h/2*k_2_x,y+h/2*l_2_y,*parameter)
    l_3_y=g_ypunkt(x+h/2*k_2_x,y+h/2*l_2_y,*parameter)
    
    k_4_x=f_xpunkt(x+h*k_3_x,y+h*l_3_y,*parameter)
    l_4_y=g_ypunkt(x+h*k_3_x,y+h*l_3_y,*parameter)
    
    xneu=x+h/6*(k_1_x+2*k_2_x+2*k_3_x+k_4_x)
    yneu=y+h/6*(l_1_y+2*l_2_y+2*l_3_y+l_4_y)
    
    return xneu, yneu


for t in range(1,int(T/h)+1):
    
    x[t],y[t]=Runge_Kutta(x[t-1],y[t-1],[epsilon_1,epsilon_2,gamma_1,gamma_2],h)
    

plt.plot(ts, x, 'b', linewidth=1.5 ,label='p_1')
plt.plot(ts, y, 'r', linewidth=2, label='p_2')
plt.legend(loc='upper right')
plt.axis([0, T, 0, 30000])
plt.xlabel('$t$ [a.u.]')
plt.ylabel('$t$')
plt.show

 
"""
plt.plot(y, x, 'g', linewidth=1.5, label='Phasenraum')
plt.legend(loc='upper right')
plt.axis([0, 430, 0, 30000])
plt.xlabel('$p_2$')
plt.ylabel('$p_1$')
plt.show
"""