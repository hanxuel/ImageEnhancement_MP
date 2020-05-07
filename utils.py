# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:37:57 2020

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


def getMemCpu(): 
    data = psutil.virtual_memory() 
    total = data.total 
    free = data.available 
    memory =(int(round(data.percent))) 
    cpu = psutil.cpu_percent(interval=1) 
    return memory,cpu

def mkdir_if_not_exists(path):
    assert os.path.exists(os.path.dirname(path.rstrip("/")))
    if not os.path.exists(path):
        os.makedirs(path)

def stepsize_variable(name,shape,value=1.0):
    init=tf.constant_initializer(value)
    return tf.get_variable(name,shape,dtype=tf.float32,
                           initializer=init)
    
def conv_weight_variable(name, shape, stddev=1.0):
    #initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=stddev))


def bias_weight_variable(name, shape, cval=0.0):
    initializer = tf.constant_initializer(cval)
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=initializer)

def conv2d(x, weights, name=None, padding="SAME"):
    return tf.nn.conv2d(x, weights, name=name,
                        strides=[1, 1, 1, 1], padding)

def gradient(imgs):
    return tf.stack([.5*(imgs[...,1:,:-1]-imgs[...,:-1,:-1]), .5*(imgs[...,:-1,1:]-imgs[...,:-1,:-1])], axis=-1)

def gradient_loss(guess, truth):
    return tf.reduce_mean(tf.abs(gradient(guess)-gradient(truth)))

def basic_img_loss(img, truth):
    l2_pixel = tf.reduce_mean(tf.square(img - truth))
    l1_grad = gradient_loss(img, truth)
    return l2_pixel + l1_grad

    
def get_list_dir(scene_path):
    working_dir = os.getcwd()

    os.chdir(scene_path)
    file_list = [] 

    for file in glob.glob("*/"):
        file_list.append(file)

    os.chdir(working_dir)

    return file_list


def get_list_files(scene_path, ext):
    working_dir = os.getcwd()

    os.chdir(scene_path)
    file_list = [] 

    for file in glob.glob("*{}".format(ext)):
        file_list.append(file)

    file_list = sorted(file_list, key=lambda name: int(name[:6]))

    os.chdir(working_dir)

    return file_list

def down_block(Input, ch, N):
    inch = tf.shape(Input)[-1]
    with tf.variable_scope("down_block{}".format(N)):
        w0 = conv_weight_variable(
                "w0", [3,3,inch, ch], stddev=0.01)
        w1 = conv_weight_variable(
                "w1", [3,3,ch, ch], stddev=0.01)
        b0 = bias_weight_variable("b0", [ch])
        b1 = bias_weight_variable("b1", [ch])
        
        Input0_residual = conv2d(Input, w0)
        Input0 = tf.nn.relu(Input0_residual + b0)
        Input1_residual = conv2d(Input0, w1)
        Input1 = tf.nn.relu(Input1_residual + b1)
        Output = tf.nn.max_pool(Input1, 2, 2)
    return Input1, Output

def up_block(Input, skip, ch, N):
    Input = tf.image.resize_images(Input,tf.shape(Input)[1]*2,tf.shape(Input)[2]*2,method=tf.image.ResizeMethod.BILINEAR)
    inch = tf.shape(Input)[-1]
    addch = tf.shape(skip)[-1]
    with tf.variable_scope("up_block{}".format(N)):
        w0 = conv_weight_variable(
                "w0", [3,3,inch, ch], stddev=0.01)
        w1 = conv_weight_variable(
                "w1", [3,3,ch+addch, ch], stddev=0.01)
        w2 = conv_weight_variable(
                "w2", [3,3,ch, ch], stddev=0.01)
        b0 = bias_weight_variable("b0", [ch])
        b1 = bias_weight_variable("b1", [ch])
        b2 = bias_weight_variable("b2", [ch])
        
        Input0_residual = conv2d(Input, w0)
        Input0 = tf.nn.relu(Input0_residual + b0)
        Merge = tf.concat([Input0, skip],axis=-1)
        Input1_residual = conv2d(Merge, w1)
        Input1 = tf.nn.relu(Input1_residual + b1)
        Output_residual = conv2d(Input1, w2)
        Output = tf.nn.relu(Output_residual + b2)
    return Output
def pooled_skip(Skip,k):
    Average_col = tf.reduce_mean(Skip,axis=1, keepdims=True)
    print("tf.shape(Average_col)",tf.shape(Average_col))
    Average = tf.reduce_mean(Average_col,axis=2, keepdims=True)
    print("tf.shape(Average)",tf.shape(Average))
    poolskip = tf.tile(Average, [1,k,k,1])
    print("tf.shape(poolskip)",tf.shape(poolskip))
    return poolskip
def convolve(img_stack, filts, final_K):
    initial_W = img_stack.get_shape().as_list()[-1]
    fsh = tf.shape(filts)
    filts = tf.reshape(filts, [fsh[0], fsh[1], fsh[2], final_K ** 2 * initial_W])
    
    kpad = final_K//2
    imgs = tf.pad(img_stack, [[0,0],[kpad,kpad],[kpad,kpad],[0,0]])
    ish = tf.shape(img_stack)
    img_stack = []
    for i in range(final_K):
        for j in range(final_K):
            img_stack.append(imgs[:, i:tf.shape(imgs)[1]-2*kpad+i, j:tf.shape(imgs)[2]-2*kpad+j, :])
            img_stack = tf.stack(img_stack, axis=-2)
            img_stack = tf.reshape(img_stack, [ish[0], ish[1], ish[2], final_K**2 * initial_W])
            img_net = tf.reduce_sum(img_stack * filts, axis=-1) # removes the final_K**2*initial_W dimension but keeps final_W
     return img_net