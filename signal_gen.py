import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def sin_wave(A=1, f=50, fs=5000, phi=0, t=1):
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s, 
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y

def sin_wave(A=1, f=50, fs=5000, phi=0, t=1):
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s, 
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y

def triangle_wave(A=1, f=50, fs=5000, phi=0, t=1):
    Ts = 1/fs
    n = t/Ts
    n = np.arange(n)
    y = A*signal.sawtooth(2*np.pi*f*n*Ts + phi*(np.pi/180), 0.5) # second parameter response to Width of the rising ramp
    return y

def square_wave(A=1, f=50, fs=5000, phi=0, t=1):
    Ts = 1/fs
    n = t/Ts
    n = np.arange(n)
    y = A*signal.square(2*np.pi*f*n*Ts + phi*(np.pi/180), 0.5) # second parameter response to Width of the rising ramp
    return y

if __name__== "__main__":
    # f=50 hz
    A = 1
    f = 5
    fs = 256
    phi = 50
    t = 1
    # phase=0
    sin = sin_wave(A=A, f=f, fs=fs, phi=phi, t=t)
    tri = triangle_wave(A=A, f=f, fs=fs, phi=phi, t=t)
    squ = square_wave(A=A, f=f, fs=fs, phi=phi, t=t)
    np.save('data/'+'sin_'+str(phi),sin)
    np.save('data/'+'tri_'+str(phi),tri)
    np.save('data/'+'squ_'+str(phi),squ)

    # # phase=30
    # sin = sin_wave(A=A, f=f, fs=fs, phi=30, t=t)
    # tri = triangle_wave(A=A, f=f, fs=fs, phi=30, t=t)
    # squ = square_wave(A=A, f=f, fs=fs, phi=30, t=t)
    # # print(squ.shape)
    # np.save('data/sin_30',sin)
    # np.save('data/tri_30',tri)
    # np.save('data/squ_30',squ)

    # # phase=31
    # sin = sin_wave(A=A, f=f, fs=fs, phi=31, t=t)
    # tri = triangle_wave(A=A, f=f, fs=fs, phi=31, t=t)
    # squ = square_wave(A=A, f=f, fs=fs, phi=31, t=t)
    # # print(squ.shape)
    # np.save('data/sin_31',sin)
    # np.save('data/tri_31',tri)
    # np.save('data/squ_31',squ)
    # # x = np.arange(0,t,1/fs)
    # # plt.xlabel('t/s')
    # # plt.ylabel('y')
    # # plt.grid()
    # # plt.plot(x, tri)
    # # plt.show()