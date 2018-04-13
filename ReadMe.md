# 3D-Segmentation-Pytorch-version
3D Segmentation Pytorch version(unet basic model)

#This is the basic pytorch version of segmentation source code.(first version)

run main.py train the model

no test.py now.(will be updated soon) using ensemble method 
    1. split the original cube into small cubes 
    2. send each of small cubes to the network
    3. ensemble small cubes to a big cube according to the split info.

model
    basic Unet model
    voxresnet(will be updated soon)
    dense net for seg(will be updated soon)

split 
    the images path info and labels path info(all the 3d data in my cases ara .tif format(ImageJ open))

config
​    all the parameters~ 

#if the GPU memory is not big enough, change the batch size and crop_size when training your network.