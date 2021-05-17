#2020/04/22
#National Chiao Tung University
#Digital Image Processing
#Mini project NO.2
#Created by Le Van Hung (0860831)
# import library
from numpy import asarray
from numpy import savetxt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# import image
image = Image.open('Bird 1.tif')
image.show()

# convert image to array version
f = np.array(image,dtype='float')
# ff = np.zeros((512,512))
#define value
N = f.shape[0]
for x in range(N):
    for y in range(N):
        # ff[x,y] = f[x,y]*pow(-1,(x+y))
        f[x,y] = f[x,y]*pow(-1,(x+y))

plt.imshow(f,cmap='gray')
plt.title("input image after multiplying by (-1)^(x+y)")
plt.show()
# DFT
F = np.fft.fft2(f)

# multi fp with (-1)^(x+y)

# plot magnitude spectral before after using log function
F_abs = np.abs(F)
F_log = np.log(1+F_abs)
plt.imshow(F_abs,cmap='gray')
plt.title("Fourier magnitude spectrum of F(u,v) before log sale")
plt.show()

plt.imshow(F_log,cmap='gray')
plt.title("Fourier magnitude spectrum of F(u,v) after log sale")
plt.show()

#plot phase spectral of F
phase = np.angle(F)
plt.imshow(phase,cmap='gray')
plt.title("plot phase spectrum of F(u,v)")
plt.show()


# sort 25 top of frequence [25 max abs()]
def find_max(array):
    len_array = array.shape[0]
    max = 0;
    row = 0;
    col = 0;
    for u in range(len_array):
        for v in range(len_array):
            if(array[u,v] >=max):
                max = array[u,v]
                row = u
                col = v
    return [row,col,max]
A_sort = []
F_log_temp = 1*F_log
for i in range(25):
    temp = find_max(F_log_temp)
    F_log_temp[temp[0],temp[1]] = 0
    A_sort.append(temp)

A_sort = np.reshape(A_sort,(25,3))





