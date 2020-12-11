import numpy as np
import os
import glob

a = glob.glob("./data3/sketch/*/*/*")

sketch_idx = []
for i in range(len(a)):

    c = a[i].split("\\")[1:]
    c = c[0]+"/"+c[1]+"/"+c[2]

    sketch_idx.append(c)

print(len(sketch_idx)) #4164



#메모리를 적게 쓰기 위해 uint8로 
sketch = np.zeros((len(sketch_idx),256,256,3),dtype=np.uint8) 
photo = np.zeros((len(sketch_idx),256,256,3),dtype=np.uint8)

from tqdm import tqdm_notebook
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize #사이즈 조절

for i, idx in (enumerate(sketch_idx)):
    # print("i : ",i)
    # print("idx : ",idx)
    img = load_img("./data3/sketch/"+idx)
    img = img_to_array(img)
    sketch[i] = img
    

#sketch에 맞춰서 photo를 늘림 
for i, idx in (enumerate(sketch_idx)):
    # print("i : ",i)
    # print("idx : ",idx)
    idx = idx.split("/")[0]+"/"+idx.split("/")[2]
    idx = idx.split("-")[0]
    img = load_img("./data3/photo/"+ idx+"-removebg-preview.png")
    img = img_to_array(img)
    photo[i] = img


print(sketch.shape) #(4164, 256, 256, 3)
print(photo.shape) #(4164, 256, 256, 3)

# print("=============================")
# print(sketch)

# print("=============================")
# print(photo)

#*npz = 압축된 npy?
from numpy import savez_compressed

# sketch: X, photo: Y
# np.save('./data/sketch_camel_door.npy', arr=sketch)
# np.save('./data/photo_camel_door.npy', arr=photo)

filename = 'bear_berry_white.npz'
np.savez_compressed(filename,sketch,photo)
print("success")
