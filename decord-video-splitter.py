from decord import VideoReader
from decord import cpu, gpu
import decord
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # This is a workaround


decord.bridge.set_bridge('torch')
video_path = "data/red.mp4"
vr = VideoReader(video_path)
print(f"Total frames: {len(vr)}")



print(vr[0].shape)
frame = vr[5]
print(type(frame))

# Here's an example of color editing, we will max out the blue channel for some area
#frame[:,750:1250,2] = 255

x = frame.shape[0]
y = frame.shape[1]
z = frame.shape[2]
#frame = frame.float()

#frame[0:200,:,2] = 255
means = []
cut = frame[0:100,:,:]
print(cut.shape)
print(cut[:,:,0])
cut = cut.float()
print(torch.mean(cut[:,:,0]).int())

counter = range(1,75)
#counter = counter * 100
for chunk in counter:
    start = (chunk-1)*10
    end = chunk*10
    cut = frame[start:end,:,:].float()
    frame[start:end,:,0] = torch.mean(cut[:,:,0]).int()
    frame[start:end,:,1] = torch.mean(cut[:,:,1]).int()
    frame[start:end,:,2] = torch.mean(cut[:,:,2]).int()




plt.imshow(frame)
plt.show()


