import sys
import os
import cv2
import numpy as np
import h5py

resource_path = 'D:/LZH/code/deep_learning/pytorch/'
pic_dir = './data/0_presented_images_800/'
root_dir = os.path.join(resource_path, pic_dir)

sys.path.append(resource_path)



resolution = 300


image_path = os.listdir(root_dir)
path_dict = {}
for j in image_path:
    key = int(j.split('_')[0])  # 刺激呈现的顺序是图像名称下划线前面的数字顺序。
    path_dict[key] = j

stim_arr = np.zeros((len(image_path), resolution, resolution, 3))
for i in range(len(image_path)):
    img_bgr = cv2.imread(os.path.join(root_dir, path_dict[i+1]))
    stim_arr[i] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
stim_arr = stim_arr.astype('float32')

f = h5py.File(os.path.join(root_dir, 'image.h5'), 'w')
f.create_dataset('image', data = stim_arr)

f.close()
