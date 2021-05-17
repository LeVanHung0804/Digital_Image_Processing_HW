#2020/03/24
#National Chiao Tung University
#Digital Image Processing
#Mini project NO.1
#Created by Le Van Hung (0860831)

# import library
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt

# import image
image = Image.open('camellia (mono) 512x512.tif')
image.show()

# convert image to array version
image_array = np.array(image)
image_shape = image_array.shape

#plot original picture
plt.imshow(image_array,cmap='gray')
plt.show()

#plot histogram of original picture
plot_pic = image_array
plot_pic = np.reshape(plot_pic,(-1,1))
plt.hist(plot_pic, bins=256)
plt.title('Histogram of original picture')
plt.ylabel('Number of pixel')
plt.xlabel('gray level')
plt.show()


# create nk vector including the number of rk respectively (rk is gray level)
nk = []
L = 256
d = image_shape[0]
for k in range(L):
    temp = 0
    for row in range(d):
        for col in range(d):
            if(image_array[row,col] == k):
                temp = temp+1
    nk.append(temp)
nk = np.array(nk)
nk = np.reshape(nk,(-1,1))

# calculate pr vector
pr = nk / (d*d)
pr = np.reshape(pr,(-1,1))

# calculate sk vector
sk = []
for i in range(L):
    temp = pr[0:i+1,0]
    sk.append(sum(temp))
sk = np.array(sk)
sk = np.reshape(sk,(-1,1))

# calculate pz
nz1 = 1248
nz2 = 800
seg_0_63 = nz1/(d*d)
seg_64_191 = nz2/(d*d)
seg_192_255 = nz1/(d*d)

pz = np.zeros_like(sk)
pz[0:64,0] = seg_0_63
pz[64:192,0] = seg_64_191
pz[192:256,0] = seg_192_255
sum_pz = sum(pz)

# calculate vn
vn = []
for i in range(L):
    temp = pz[0:i+1,0]
    vn.append(sum(temp))
vn = np.array(vn)
vn = np.reshape(vn,(-1,1))

# we already have sk and vn => compare sk,vn to get the mapping from rk to zk
map_vector = []
for i in range(L):  #i for sk i=0..255
    temp = sk[i,0]
    for j in range(L): #j for vn j=0..255
        if (vn[j,0] >= temp):
            map_vector.append(j)
            break

map_vector = np.array(map_vector)
map_vector = np.reshape(map_vector,(-1,1))

#convert intensity base on map_vector
for row in range(d):
    for col in range(d):
        temp = image_array[row,col]
        image_array[row,col] = map_vector[temp,0]

#plot histogram of picture after using histogram specification
plot_pic = image_array
plot_pic = np.reshape(plot_pic,(-1,1))
plt.hist(plot_pic, bins=256)
plt.title('Histogram of picture after using histogram specification')
plt.ylabel('Number of pixel')
plt.xlabel('gray level')
plt.show()
plt.show()

# plot picture after using histogram specification
plt.imshow(image_array,cmap='gray')
plt.show()

#plot transformation function T(r)
rk = np.linspace(0,255,num=256)
rk = np.array(rk)
rk = np.reshape(rk,(-1,1))
plt.plot(rk,map_vector,color='g', label='T(r)')
plt.title('Transformation function')
plt.ylabel('Output intensity')
plt.xlabel('Input intensity')
plt.show()

#save nk,pr,sk,vn,pz respectively to excel file
# save to csv file
file_save = np.zeros([256,5])
file_save[:,0] = nk[:,0]
file_save[:,1] = pr[:,0]
file_save[:,2] = sk[:,0]
file_save[:,3] = vn[:,0]
file_save[:,4] = pz[:,0]

data = asarray(file_save)
savetxt('file_data_detail.csv', data, delimiter=',')

#save table of transformation function mapping r to z

file_save = np.zeros([256,2])
file_save[:,0] = rk[:,0]
file_save[:,1] = map_vector[:,0]
data = asarray(file_save)
savetxt('file_mapping', data, delimiter=',')









