import math
import numpy as np
import matplotlib.pylab as plt
import pylab

import pywt
import pylab
from statsmodels.compat import scipy

mode = pywt.MODES.sp1
DWT = 1

def plot(data, w, title):
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []

    if DWT:
        for i in xrange(5):
            (a, d) = pywt.dwt(a, w, mode)
            ca.append(a)
            cd.append(d)
    else:
        coeffs = pywt.swt(data, w)  # [(cA5, cD5), ..., (cA1, cD1)]
        for a, d in reversed(coeffs):
            ca.append(a)
            cd.append(d)

    pylab.figure()
    ax_main = pylab.subplot(len(ca) + 1, 1, 1)
    pylab.title(title)
    ax_main.plot(data)
    pylab.xlim(0, len(data) - 1)

    for i, x in enumerate(ca):
        ax = pylab.subplot(len(ca) + 1, 2, 3 + i * 2)
        ax.plot(x, 'r')
        if DWT:
            pylab.xlim(0, len(x) - 1)
        else:
            pylab.xlim(w.dec_len * i, len(x) - 1 - w.dec_len * i)
        pylab.ylabel("A%d" % (i + 1))

    for i, x in enumerate(cd):
        ax = pylab.subplot(len(cd) + 1, 2, 4 + i * 2)
        ax = pylab.subplot(len(cd) + 1, 2, 4 + i * 2)
        ax.plot(x, 'g')
        pylab.xlim(0, len(x) - 1)
        if DWT:
            pylab.ylim(min(0, 1.4 * min(x)), max(0, 1.4 * max(x)))
        else:  # SWT
            pylab.ylim(
                min(0, 2 * min(
                    x[w.dec_len * (1 + i):len(x) - w.dec_len * (1 + i)])),
                max(0, 2 * max(
                    x[w.dec_len * (1 + i):len(x) - w.dec_len * (1 + i)]))
            )
        pylab.ylabel("D%d" % (i + 1))

x = np.linspace(-math.pi, math.pi, 100)
y = np.cos(5*x) + np.cos(10*x)+ np.cos(25*x)+ np.cos(50*x)+ np.cos(100*x)
plt.plot(x, y)
plt.axis('tight')
plt.show()
y_idx = [5,10,25,50,100]
y2 = []
for i in range(0,5):
    x = np.linspace(i*math.pi*2, 2*(i+1)*math.pi, 20)
    y_segment = np.cos(y_idx[i]*x)
    y2.extend(y_segment)
    plt.plot(x,y_segment)
    # plt.axis('tight')
plt.show()
y2 =  np.asarray(y2)
DWT = 1


freq = np.fft.fftfreq(x.shape[-1])
data = np.fft.fft(y2)
# coeffs = pywt.swt(data, w, 5)
plt.plot(data)
# plt.plot(data2)
# plt.show()
plot(y2, 'db1', "DWT")
plot(y, 'db1', "DWT")
plt.show()
# plot(data3, 'sym5', "DWT: Ecg sample - Symmlets5")

# DWT = 0  # SWT
# plot(data1, 'db1', "SWT: Signal irregularity detection - Haar wavelet")
# plot(data2, 'sym5', "SWT: Frequency and phase change - Symmlets5")
# # plot(data3, 'sym5', "SWT: Ecg sample - simple QRS detection - Symmlets5")
