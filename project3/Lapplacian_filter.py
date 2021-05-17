#2020/04/22
#National Chiao Tung University
#Digital Image Processing
#Mini project NO.3
#Created by Le Van Hung (0860831)

# import library
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# import image
image = Image.open('Bird 1.tif')
image.show()

# convert image to array version
f = np.array(image,dtype='float')

#plot original picture
plt.imshow(f,cmap='gray')
plt.title("original image")
plt.show()
#define value
N = f.shape[0]
P = 2*N

# zeros padding to get new image fp 2N*2N
fp = np.zeros((2*N,2*N),dtype='float')
fp = np.array(fp)
fp[0:N,0:N] = f

#plot zero padded image before centering
plt.imshow(fp,cmap='gray')
plt.title("zero padded image before centering")
plt.show()

# multi fp with (-1)^(x+y)
for x in range(P):
    for y in range(P):
        fp[x,y] = fp[x,y]*pow(-1,(x+y))

# DFT
Fp = np.fft.fft2(fp)

# set up H (laplacian filter)
H = []
K = -1/(2*(N)*(N))
for u in range(P):
    for v in range(P):
        temp = K*((u-N)*(u-N)+(v-N)*(v-N))
        H.append(temp)
H = np.reshape(H,[P,P])
H_abs = np.abs(H)

# element wise F with H to get G
G = np.multiply(Fp,H)
# calculate gp (IDFT)
gp = np.fft.ifft2(G)

# get real of gp
gp_real = gp.real
gp_real = np.array(gp_real,dtype='float')

# multi gp with (-1)^(x+y)
for x in range(N):
    for y in range(N):
        gp_real[x,y] = gp_real[x,y]*pow(-1,(x+y))

g = gp_real[0:N,0:N]

#plot zero padded image after centering
plt.imshow(fp,cmap='gray')
plt.title("zero padded image after centering")
plt.show()

# plot magnitude spectra of F
F_abs = np.abs(Fp)
plt.imshow(F_abs,cmap='gray')
plt.title("Fourier magnitude of bird before applying filtering")
plt.show()

# plot magnitude spectra of G = F*H
G_abs = np.abs(G)
plt.imshow(G_abs,cmap='gray')
plt.title("Fourier magnitude of bird after applying filtering (G = F*H)")
plt.show()

# plot Fourier magnitude of Laplacian filter H(u,v)
plt.imshow(H_abs,cmap='gray')
plt.title("Fourier magnitude of Laplacian filter H(u,v)")
plt.show()

#plot Laplacian image
plt.imshow(g,cmap='gray')
plt.title("ouput of Laplacian image")
plt.show()

# plot image after plus with Laplacian image
plt.imshow(f+g,cmap='gray')
plt.title("image after plus with laplacian image")
plt.show()

# plot magnitude of original image
F = np.fft.fft2(f)
F_abs_0 = np.abs(F)
plt.imshow(F_abs_0,cmap='gray')
plt.title("Fourier magnitude of original image")
plt.show()

# show 25 top DFT frequencies (u,v) after Laplacian filtering
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
G_temp = 1*G_abs
for i in range(25):
    temp = find_max(G_temp)
    G_temp[temp[0],temp[1]] = 0
    A_sort.append(temp)

A_sort = np.reshape(A_sort,(25,3))
a =1
