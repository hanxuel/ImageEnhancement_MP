# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:43:03 2020

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
from utils import *
from collections import Counter
from numpy import *

def build_kpn_model(params):
    epoch_npasses = params["epoch_npasses"]
    batch_size = params["batch_size"]
    width = params["width"]
    height = params["height"]
    burst_length = params["burst_length"]
    B = params["num_basis"]
    K = params["kernel_size"]
    #with tf.variable_scope("reconstruction"):
    # Setup placeholders and variables.
# =============================================================================
#     with tf.device('/gpu:' + str(0)): 
# =============================================================================
    Input = tf.placeholder(tf.float32, [None, height, width,  burst_length], name="Input")
    with tf.variable_scope("weights0"):
        w0 = conv_weight_variable(
                "w0", [3,3,burst_length, 64], stddev=0.01)
        b0 = bias_weight_variable("b0", [64])
        Input0_residual = conv2d(Input, w0)
        Input0 = tf.nn.relu(Input0_residual + b0)
        
    Skip1, Output1 = down_block(Input0, 64, 1)
    Skip2, Output2 = down_block(Output1, 128, 2)
    Skip3, Output3 = down_block(Output2, 256, 3)
    Skip4, Output4 = down_block(Output3, 512, 4)
    Skip5, Output5 = down_block(Output4, 1024, 5)
    
    with tf.variable_scope("weights1"):
        w0 = conv_weight_variable(
                "w0", [3,3,1024, 1024], stddev=0.01)
        w1 = conv_weight_variable(
                "w1", [3,3,1024, 1024], stddev=0.01)
        b0 = bias_weight_variable("b0", [1024])
        b1 = bias_weight_variable("b1", [1024])
        Output6_1_residual = conv2d(Output5, w0)
        Output6_1 = tf.nn.relu(Output6_1_residual + b0)
        Output6_2_residual = conv2d(Output6_1, w1)
        Output6 = tf.nn.relu(Output6_2_residual + b1)
        
    with tf.variable_scope("CoefficientsBranch"):
        Up5 = up_block(Output6, Skip5, 512, 5)
        Up4 = up_block(Up5, Skip4, 256, 4)
        Up3 = up_block(Up4, Skip3, 128, 3)
        Up2 = up_block(Up3, Skip2, 64, 2)
        Up1 = up_block(Up2, Skip1, 64, 1)
        
    with tf.variable_scope("weights2"):
        w0 = conv_weight_variable(
                "w0", [3,3,64, 64], stddev=0.01)
        w1 = conv_weight_variable(
                "w1", [3,3,64, 64], stddev=0.01)
        w2 = conv_weight_variable(
                "w2", [3,3,64, B], stddev=0.01)
        b0 = bias_weight_variable("b0", [64])
        b1 = bias_weight_variable("b1", [64])
        b2 = bias_weight_variable("b2", [B])
        Output7_1_residual = conv2d(Up1, w0)
        Output7_1 = tf.nn.relu(Output7_1_residual + b0)
        Output7_2_residual = conv2d(Output7_1, w1)
        Output7 = tf.nn.relu(Output7_2_residual + b1)
        
        Coefficients = tf.nn.relu(conv2d(Output7,w2)+b2)
        Coefficients = tf.nn.softmax(Coefficients, axis=-1, name=None)
        #batch_size*H*W*B
    with tf.variable_scope("BasisBranch"):
        Globalaverage_col = tf.reduce_mean(Output6,axis=1, keepdims=True)
        Globalaverage = tf.reduce_mean(Globalaverage_col,axis=2, keepdims=True)
        
        poolskip5 = pooled_skip(Skip5,2)
        Upbasis5 = up_block(Globalaverage, poolskip5, 512, 5)
        poolskip4 = pooled_skip(Skip4,4)
        Upbasis4 = up_block(Upbasis5, poolskip4, 256, 4)
        poolskip3 = pooled_skip(Skip3,8)
        Upbasis3 = up_block(Upbasis4, poolskip3, 128,3)
        poolskip2 = pooled_skip(Skip2,16)
        Upbasis2 = up_block(Upbasis3, poolskip2, 64,2)
        
    with tf.variable_scope("weights3"):
        w0 = conv_weight_variable(
                "w0", [2,2,128, 128], stddev=0.01)
        w1 = conv_weight_variable(
                "w1", [3,3,128, 128], stddev=0.01)
        w2 = conv_weight_variable(
                "w2", [3,3,128, burst_length*B], stddev=0.01)
        b0 = bias_weight_variable("b0", [128])
        b1 = bias_weight_variable("b1", [128])
        b2 = bias_weight_variable("b2", [burst_length*B])
        Output8_1_residual = conv2d(Upbasis2, w0, padding="VALID")
        print("Output8_1_residual.shape",tf.shape(Output8_1_residual))
        Output8_1 = tf.nn.relu(Output8_1_residual + b0)
        Output8_2_residual = conv2d(Output8_1, w1)
        Output8 = tf.nn.relu(Output8_2_residual + b1)
        
        Basis = tf.nn.relu(conv2d(Output8,w2)+b2)
        print("Basis.shape",tf.shape(Basis))
    ish = tf.shape(Basis)
    Basis = tf.nn.softmax(tf.reshape(Basis,[ish[0],K**2,ish[3],ish[4]]),axis=1)
    Basis = tf.reshape(Basis,[ish[0],K,K,ish[3],ish[4]])
    
    Coefficients = tf.expand_dims(tf.expand_dims(tf.expand_dims(Coefficients,3),3), 3)
    print("Coefficients.shape",tf.shape(Coefficients))
    Coefficients = tf.tile(Coefficients,[1,1,1,K,K,burst_length,1])
    Basis = tf.expand_dims(tf.expand_dims(Basis,1),1)
    Basis = tf.tile(Basis,[1,height,width,1,1,1,1])
    filts = tf.reduce_sum(Basis*Coefficients,axis=-1)
    Deblur = convolve(Input, filts, K)
    return Input, Deblur
