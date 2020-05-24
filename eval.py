# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:17:10 2020

@author: lenovo
"""

import pydot
import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from data_utils import *
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from model_library import *
import argparse
from keras.utils import plot_model
from tensorflow.keras import regularizers
import time
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="G:\\master_thesis\Cengiz\BurstDenoising\Traindata")
    parser.add_argument("--val_path", default="G:\\master_thesis\Cengiz\BurstDenoising\Testdata")
    #parser.add_argument("--model_path", required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--BURST_LENGTH", type=int, default=2)
    parser.add_argument("--Kernel_size", type=int, default=15)
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--jitter", type=int, default=16)
    parser.add_argument("--smalljitter", type=int, default=2)
    parser.add_argument("--Basis_num", type=int, default=10)
    parser.add_argument("--color", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=4)
    parser.add_argument("--val_nbatches", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--degamma", type=float, default=2.2)
    parser.add_argument("--to_shift", type=float, default=1.0)
    parser.add_argument("--decay_step", type=int, default=90) 
    parser.add_argument("--loss_weight", type=float, default=2)
    parser.add_argument("--anneal",type=float, default=0.998)
    parser.add_argument("--regu",type=float, default=0.000)
    parser.add_argument("--checkpoint_path",default="./gray_tf_ckpts")
    return parser.parse_args()
def evaluate():
    args = parse_args()

    np.random.seed(0)
    tf.random.set_seed(0)

    #tf.logging.set_verbosity(tf.logging.INFO)
    params = {
        "train_path":args.train_path,
        "val_path":args.val_path,
        "batch_size":args.batch_size,
        "height":args.height,
        "width":args.width,
        "color":args.color,
        "degamma":args.degamma,
        "to_shift":args.to_shift,
        "upscale":args.upscale,
        "jitter":args.jitter,
        "smalljitter":args.smalljitter,
        "BURST_LENGTH":args.BURST_LENGTH,
        "Kernel_size":args.Kernel_size,
        "Basis_num":args.Basis_num,
        "nepochs":args.nepochs,
        "learning_rate":args.learning_rate,
        "regu":args.regu,
        "ckpt":args.checkpoint_path,
        "anneal":args.anneal
        }
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = '.\\logs\\gradient_tape\\' + current_time + '\\train'
    #log_dir = '.\\logs\\gradient_tape\\20200517-175525'+'\\train'
    model=Simplemodel(params)    
    Image_ds = DataLoader(params)
    #train_image_ds = Image_ds.get_ds()
    val_image_ds = Image_ds.get_val_ds()
    #epochs = params["nepochs"]
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    loss_metric = tf.keras.metrics.Mean()
    
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), iterate=tf.Variable(0,trainable=False), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, params["ckpt"], max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("ckpt.interate",int(ckpt.iterate))
    print("ckpt.step",int(ckpt.step))
    

    start = time.clock()
    val_psnr = []
    val_psnr_perlayer=[[0]*1 for i in range(params["BURST_LENGTH"])]
    val_psnr_noise0 = []
    val_psnr_average = []
    for step_val, x_batch_val in enumerate(val_image_ds):
        x_batch_burst, x_batch_truth=x_batch_val
        with tf.GradientTape() as tape:
            reconstructed = model(x_batch_burst)
            #print("tf.shape(reconstructed)",tf.shape(reconstructed))
            # Compute reconstruction loss
            loss = deblur_layer_loss(x_batch_truth, reconstructed,anneal_coef)
            #loss += sum(model.losses)  # Add KLD regularization loss
            
            white_noise=tf.expand_dims(x_batch_truth[...,1],axis=-1)
            white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
            gt = x_batch_truth[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
            invert_gt = invert_preproc(gt,white_noise)#.numpy()
            
            Deblur = reconstructed[...,0]#tf.expand_dims(y_pred[...,0],axis=-1)
            invert_deblur = invert_preproc(Deblur, white_noise)
            
            onestep_psnr = psnr_deblur(invert_deblur, invert_gt)
            #onestep_psnr =psnr_deblur(x_batch_truth, reconstructed)
            val_psnr.append(onestep_psnr.numpy())
            
            onestep_psnr_perlayer = psnr_each_layer(invert_gt, white_noise, reconstructed)
            
            onestep_psnr_noise0 = psnr_burst0(invert_gt, white_noise,x_batch_burst)
            val_psnr_noise0.append(onestep_psnr_noise0.numpy())
            
            onestep_psnr_average = psnr_average_f(invert_gt, white_noise,x_batch_burst)
            val_psnr_average.append(onestep_psnr_average.numpy())
            for i in range(params["BURST_LENGTH"]):
                val_psnr_perlayer[i].append(onestep_psnr_perlayer['da{}_noshow'.format(i)].numpy())
        loss_metric(loss)
    print("step_val",step_val)
    print('-----------------------------validation resule for %d------------------------------'%(Image_ds.val_count))
    print('epoch %s: mean loss = %s' % (int(ckpt.step), loss_metric.result()))
    print('epoch %s: val_psnr = %s' % (int(ckpt.step), np.mean(val_psnr)))
    print('epoch %s: val_psnrnoshow0 = %s' % (int(ckpt.step), np.mean(val_psnr_perlayer[0])))
    print('epoch %s: val_psnrburst0 = %s' % (int(ckpt.step), np.mean(val_psnr_noise0)))
    print('epoch %s: val_psnraverage = %s' % (int(ckpt.step), np.mean(val_psnr_average)))

if __name__ == "__main__":
    evaluate()