# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:39:52 2020

@author: lenovo
"""
import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing
import joblib
import collections
import psutil,datetime,time
from collections import Counter
from numpy import *
from utils import *
def build_image_generator(data_path,params,istrain=False):
    epoch_npasses = params["epoch_npasses"]
    batch_size = params["batch_size"]
    width = params["width"]
    height = params["height"]
    burst_length = params["burst_length"]
    
    burst_images=[]
    groundtruth_images=[]
    image_list = get_list_dir(data_path)
    for i, image_name in enumerate(image_list):
        print("Loading {} [{}/{}]".format(image_name, i + 1, len(image_list)))
        
        burst_path = os.path.join(data_path,image_name,"burst_image.npz")
        groundtruth_path = os.path.join(data_path,image_name,"groundtruth.npz")
        burst_image_data = np.load(burst_path)
        burst_image=np.zeros(burst_image_data["shape"],dtype=np.float32)
        burst_image=burst_image_data["values"]
        
        groundtruth_data = np.load(groundtruth_path)
        groundtruth_image = np.zeros(groundtruth_data["shape"],dtype=np.float32)
        groundtruth_image = groundtruth_data["values"]
        
        assert burst_image.shape[3] == burst_length
        
        burst_images.append(burst_image)
        groundtruth_images.append(groundtruth_image)
    idxs = np.arange(len(burst_images))
    print("-------------------------number of images is{}--------------------------------".format(len(image_list)))
    batch_burstimages = np.empty(
        (batch_size, height, width,  burst_length), dtype=np.float32)
    batch_groundtruthimages = np.empty(
        (batch_size, height, width), dtype=np.float32)

    npasses = 0
    
    while True:
        np.random.shuffle(idxs)
        for batch_start_index in range(0,len(idxs),batch_size):
            batch_end_index = min(batch_start_index+batch_size, len(idxs))
            batch_idxs = idxs[batch_start_index:batch_end_index]
            batch_burstimages[:] = 0
            batch_groundtruthimages[:] = 0
            
# =============================================================================
#             for i,index in enumerate(batch_idxs):
#                 batch_groundtruthimages[i] = groundtruth_images[index]
#                 batch_burstimages[i] = burst_images[index]
# =============================================================================
            batch_groundtruthimages = groundtruth_images[batch_idxs]
            batch_burstimages = burst_images[batch_idxs]
            
        yield (batch_burstimages[:len(batch_idxs)],
                   batch_groundtruthimages[:len(batch_idxs)])
        npasses += 1

        if epoch_npasses > 0 and npasses >= epoch_npasses:
            npasses = 0
            yield
    
