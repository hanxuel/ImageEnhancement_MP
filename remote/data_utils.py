# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:30:23 2020

@author: lenovo
"""
import os
import numpy as np
import tensorflow as tf
import random
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import glob
import matplotlib.pyplot as plt
import datetime
import psutil
def getMemCpu(): 
    data = psutil.virtual_memory() 
    total = data.total 
    free = data.available 
    memory =(int(round(data.percent))) 
    cpu = psutil.cpu_percent(interval=1) 
    return memory,cpu
def sRGBforward(x):
    b = .0031308
    gamma = 1./2.4
    # a = .055
    # k0 = 12.92
    a = 1./(1./(b**gamma*(1.-gamma))-1.)
    k0 = (1+a)*gamma*b**(gamma-1.)
    gammafn = lambda x : (1+a)*tf.pow(tf.maximum(x,b),gamma)-a
    # gammafn = lambda x : (1.-k0*b)/(1.-b)*(x-1.)+1.
    srgb = tf.where(x < b, k0*x, gammafn(x))
    k1 = (1+a)*gamma
    srgb = tf.where(x > 1, k1*x-k1+1, srgb)
    return srgb
def gradient(imgs):
    return tf.stack([.5*(imgs[...,1:,:-1]-imgs[...,:-1,:-1]), .5*(imgs[...,:-1,1:]-imgs[...,:-1,:-1])], axis=-1)

def gradient_loss(guess, truth):
    return tf.reduce_mean(tf.abs(gradient(guess)-gradient(truth)))
def invert_preproc(imgs,white_level):
    lbuff = 8
    wl = tf.reshape(white_level, [-1])
    return sRGBforward(tf.transpose(tf.transpose(imgs) / wl))[:, lbuff:-lbuff, lbuff:-lbuff, ...]
def basic_img_loss(img, truth):
    l2_pixel = tf.reduce_mean(tf.square(img - truth))
    #print(l2_pixel.numpy())
    l1_grad = gradient_loss(img, truth)
    #print(l1_grad.numpy())
    return l2_pixel + l1_grad
def deblur_layer_loss(y_true, y_pred,anneal_coef):
    batch = tf.shape(y_pred)[0]
    burst_size = tf.shape(y_pred)[-1]-1
    white_noise=tf.expand_dims(y_true[...,1],axis=-1)
    white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
    wlb = tf.reshape(white_noise, [batch, 1,1,1])
    gt = y_true[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
    invert_gt = invert_preproc(gt,white_noise)#.numpy()
    Deblur = y_pred[...,0]#tf.expand_dims(y_pred[...,0],axis=-1)
    invert_deblur = invert_preproc(Deblur, white_noise)
    #print(tf.shape(invert_deblur))
    #print(tf.shape(invert_gt))
    loss0 = basic_img_loss(invert_deblur,invert_gt)
    invert_perlayer={}
    for i in range(burst_size):
        invert_perlayer['da{}_noshow'.format(i)]=invert_preproc(y_pred[...,i+1],white_noise)
        loss0 = loss0 + anneal_coef * basic_img_loss(invert_perlayer['da{}_noshow'.format(i)],invert_gt) 
    return loss0
def deblur_loss(y_true, y_pred):
    batch = tf.shape(y_pred)[0]
    burst_size = tf.shape(y_pred)[-1]-1
    white_noise=tf.expand_dims(y_true[...,1],axis=-1)
    white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
    wlb = tf.reshape(white_noise, [batch, 1,1,1])
    gt = y_true[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
    invert_gt = invert_preproc(gt,white_noise)#.numpy()
    Deblur = y_pred[...,0]#tf.expand_dims(y_pred[...,0],axis=-1)
    invert_deblur = invert_preproc(Deblur, white_noise)
    #print(tf.shape(invert_deblur))
    #print(tf.shape(invert_gt))
    loss0 = basic_img_loss(invert_deblur,invert_gt)
    return loss0
def cost_volume(Basis):
    ish = tf.shape(Basis)
    #print("basis shape",ish)
    average = tf.reduce_mean(Basis,axis=-1)
    #print("tf.shape(average)",tf.shape(average))
    average_2 = tf.reduce_mean(tf.math.square(Basis),axis=-1)
    #print("tf.shape(average_2)",tf.shape(average_2))
    
    cost = average_2 - tf.math.square(average)
    #print("tf.shape(cost)",tf.shape(cost))
    variance = tf.reduce_mean(cost)
    #loss1 = tf.math.reciprocal(variance, name=None)
    #loss2 = tf.math.exp(-variance, name=None)
    return -variance#loss1+loss2
def psnr_tf(estimate, truth):
  return -10. * tf.log(tf.reduce_mean(tf.square(estimate - truth))) / tf.math.log(10.)


def psnr_tf_batch(estimate, truth):
    return tf.reduce_mean(-10. * tf.math.log(tf.reduce_mean(tf.reshape(tf.square(estimate - truth), [tf.shape(estimate)[0],-1]), axis=1)) / tf.math.log(10.))

def psnr_deblur(invert_deblur,invert_gt):
# =============================================================================
#     white_noise=tf.expand_dims(y_true[...,1],axis=-1)
#     white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
#     gt = y_true[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
#     invert_gt = invert_preproc(gt,white_noise)#.numpy()
#     Deblur = y_pred[...,0]#tf.expand_dims(y_pred[...,0],axis=-1)
#     invert_deblur = invert_preproc(Deblur, white_noise)
# =============================================================================
    return psnr_tf_batch(invert_deblur, invert_gt)
def psnr_each_layer(invert_gt, white_noise, y_pred):
    burst_size = tf.shape(y_pred)[-1]-1
# =============================================================================
#     white_noise=tf.expand_dims(y_true[...,1],axis=-1)
#     white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
#     gt = y_true[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
#     invert_gt = invert_preproc(gt,white_noise)#.numpy()
# =============================================================================
    invert_perlayer={}
    psnr = {}
    for i in range(burst_size):
        invert_perlayer['da{}_noshow'.format(i)]=invert_preproc(y_pred[...,i+1],white_noise)
        psnr['da{}_noshow'.format(i)] = psnr_tf_batch(invert_perlayer['da{}_noshow'.format(i)],invert_gt)
    return psnr
def psnr_burst0(invert_gt, white_noise, x_batch_burst):
# =============================================================================
#     white_noise=tf.expand_dims(y_true[...,1],axis=-1)
#     white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
#     gt = y_true[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
#     invert_gt = invert_preproc(gt,white_noise)#.numpy()
# =============================================================================
    noise0 = x_batch_burst[...,0]
    invert_noise0 = invert_preproc(noise0,white_noise)
    return psnr_tf_batch(invert_noise0, invert_gt)
def psnr_average_f(invert_gt, white_noise, x_batch_burst):
# =============================================================================
#     white_noise=tf.expand_dims(y_true[...,1],axis=-1)
#     white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
#     gt = y_true[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
#     invert_gt = invert_preproc(gt,white_noise)#.numpy()
# =============================================================================
    average =tf.reduce_mean(x_batch_burst, axis=-1)
    invert_average = invert_preproc(average,white_noise)
    return psnr_tf_batch(invert_average, invert_gt)

class DataLoader():
    def __init__(self, params):
        self.params=params
        self.batch_size = params["batch_size"]
        self.color = params["color"]
        if 'train_path' in params:
            self.train_path = params["train_path"]
            data_root = pathlib.Path(self.train_path)
            self.percent = params["percent"]
            if self.color:
                self.channels = 3
                self.all_img_paths = list(data_root.glob('*.jpg'))
            else:
                self.channels = 1
                self.all_img_paths = list(data_root.glob('*.png'))   
            self.all_img_paths = [str(path) for path in self.all_img_paths]
            valid = int(self.percent* len(self.all_img_paths))
            self.all_img_paths = self.all_img_paths[:valid]
            self.count = len(self.all_img_paths)
            print("number of train dataset--------------------",self.count)
        
        if 'val_path' in params:
            self.val_path = params["val_path"]
            val_data_root = pathlib.Path(self.val_path)
            if self.color:
                self.val_all_img_paths = list(val_data_root.glob('*.jpg'))
            else:
                self.val_all_img_paths = list(val_data_root.glob('*.png'))
            self.val_all_img_paths = [str(path) for path in self.val_all_img_paths]
            self.val_count = len(self.val_all_img_paths)
            print("number of val dataset--------------------",self.val_count)

    def preprocess_image(self,image,params):
        
        if 'height' in params:
            height=params["height"]
            width=params["width"]
        else:
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
        BURST_LENGTH=params["BURST_LENGTH"]
        degamma=params["degamma"]
        to_shift=params["to_shift"]
        upscale=params["upscale"]
        jitter=params["jitter"]
        smalljitter=params["smalljitter"]
        print("BURST_LENGTH",BURST_LENGTH)
        patches = self.make_first_truth((tf.cast(image, tf.float32) / 255.)**degamma,\
                                                 height, width, BURST_LENGTH, to_shift, upscale, jitter)
        print("patches size",tf.shape(patches))
        #PATCHES ================= [1, 256, 256, 3]
        demosaic_truth = self.make_truth_hqjitter(patches, BURST_LENGTH, height, width, to_shift, upscale, jitter, smalljitter)
        print("demosaic_truth size",tf.shape(demosaic_truth))

        demosaic_truth = tf.reduce_mean(demosaic_truth, axis=-2)
            
        truth_all = demosaic_truth

        degamma = 1.
        white_level = tf.pow(10., tf.compat.v1.random_uniform([1, 1, 1], np.log10(.1), np.log10(1.)))
# =============================================================================
#         if self.color:
#             white_level = tf.expand_dims(white_level, axis=-1)
# =============================================================================
        demosaic_truth = (white_level * demosaic_truth ** degamma)
        
        sig_read = tf.pow(10., tf.compat.v1.random_uniform([1, 1, 1], -3., -1.5))
        sig_shot = tf.pow(10., tf.compat.v1.random_uniform([1, 1, 1], -2., -1.))
        dec = demosaic_truth
        #dec = self.decimatergb(demosaic_truth, BURST_LENGTH) if self.color else demosaic_truth
#        print 'DEC',dec.get_shape().as_list()
        noisy_, _ = self.add_read_shot_tf(dec, sig_read, sig_shot)
        noisy = noisy_

        first2 = demosaic_truth[...,:2]
        #demosaic_truth = demosaic_truth[...,0,:] if self.color else demosaic_truth[...,0]
        demosaic_truth = demosaic_truth[...,0]
        
        demosaic_truth = tf.expand_dims(demosaic_truth,-1)
        #print(tf.shape(demosaic_truth))
        white_level = tf.tile(white_level,[height,width,1])
        return noisy,tf.concat([demosaic_truth,white_level],axis=-1)
    

    def load_and_preprocess_image(self,image,params):
        #image = tf.io.read_file(path)
        #DataLoader.preprocess_image(self,image,params)
        return self.preprocess_image(image,params)
    def load_image(self, path, params):
        image = tf.io.read_file(path)
        if self.color:
            image = tf.image.decode_jpeg(image, channels=3)
        else:
            image = tf.image.decode_png(image, channels=1)
        return image
    def parse_function(self,example_proto, params):
        burst_length = params["BURST_LENGTH"]
        height_2=2*params["height"]
        width_2=2*params["width"]
        height=2*params["height"]
        width=2*params["width"]
        features={
                'readvar': tf.FixedLenFeature([], tf.string),
                'shotfactor': tf.FixedLenFeature([], tf.string),
                'blacklevels': tf.FixedLenFeature([], tf.string),
                'channelgain': tf.FixedLenFeature([], tf.string),
                'burst_raw': tf.FixedLenFeature([], tf.string),
                'merge_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        burst, merged, readvar, shotfactor = self.decode(parsed_features)
        
        d = tf.shape(burst)[-1]
        burst = tf.cond(d > burst_length, lambda: burst[...,:burst_length], lambda: burst)
        burst = tf.cond(d < burst_length, lambda: tf.pad(burst, [[0,0],[0,0],[0,burst_length-d]]), lambda: burst)

        mosaic, demosaic = self.sample_patch(burst, merged, height_2, width_2, burst_length)
        burst = tf.reshape(mosaic, [height_2, width_2, burst_length])
        demosaic = tf.reshape(demosaic, [height_2, width_2, 3])
        readvar = tf.reshape(readvar, [1, 4])
        shotfactor = tf.reshape(shotfactor, [1, 4])
# =============================================================================
#         valid_mask = tf.ones([1,tf.minimum(burst_length,d)])
#         valid_mask = tf.cond(burst_length > d, lambda : tf.concat([valid_mask,tf.zeros([1,burst_length-d])], axis=-1), lambda : valid_mask)
#         valid_mask = tf.reshape(valid_mask, [burst_length])
# =============================================================================
        noisy = self.batch_down2(burst)
        demosaic_truth = self.batch_down2(tf.reduce_mean(demosaic, axis=-1,keepdims=True))
        #truth_all = tf.expand_dims(demosaic_truth, axis=-1)
        #shift = tf.zeros([batch, BURST_LENGTH-1])
        
        #noisiness = tf.reshape(tf.reduce_mean(readvar,axis=1),[1,1,1]) + tf.maximum(0.,noisy[...,0:1]) * tf.reshape(tf.reduce_mean(shotfactor,axis=1),[1,1,1])
        #sig_read = tf.reshape(tf.sqrt(tf.reduce_mean(noisiness, axis=[0,1,2])), [1,1,1])
        #sig_shot = sig_read
        #sig_read = tf.sqrt(noisiness)
        white_level = tf.reduce_max(tf.reshape(demosaic_truth, [-1]))
        white_level = tf.reshape(white_level, [1,1,1])
        white_level = tf.tile(white_level,[height,width,1])
        return noisy,tf.concat([demosaic_truth,white_level],axis=-1)
# =============================================================================
#         if color:
#           noisy = small_bayer_stack(burst)
#           demosaic_truth = demosaic
#           # dumb0 = dumb_tf_demosaic(burst[...,0])
#           # dumb_avg = dumb_tf_demosaic(tf.reduce_mean(burst, axis=-1))
#           dumb0 = dumb_tf_demosaic(tf22reshape(noisy[...,::BURST_LENGTH]))
#           dumb_avg = dumb_tf_demosaic(tf22reshape(tf.reduce_mean(tf.reshape(noisy, [tf.shape(noisy)[0],tf.shape(noisy)[1],tf.shape(noisy)[2],4,BURST_LENGTH]), axis=-2)))
#           white_level = tf.ones([batch, 1, 1, 1])
# =============================================================================
        
        
# =============================================================================
#         mosaic, demosaic = burst2patches(burst, merged, height, width, depth, burst_length)
#         mosaic = tf.reshape(mosaic, [depth, height, width, burst_length])
#         demosaic = tf.reshape(demosaic, [depth, height, width, 3])
#         readvar = tf.tile(tf.reshape(readvar, [1, 4]), [depth, 1])
#         shotfactor = tf.tile(tf.reshape(shotfactor, [1, 4]), [depth, 1])
# =============================================================================
    
# =============================================================================
#         valid_mask = tf.ones([1,tf.minimum(burst_length,d)])
#         valid_mask = tf.cond(burst_length > d, lambda : tf.concat([valid_mask,tf.zeros([1,burst_length-d])], axis=-1), lambda : valid_mask)
#         valid_mask = tf.tile(valid_mask, [depth, 1])
#         valid_mask = tf.reshape(valid_mask, [depth, burst_length])
# =============================================================================

    def get_ds(self):
# =============================================================================
#         current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         path_ds = tf.data.Dataset.from_tensor_slices(self.all_img_paths)
#         image_ds = path_ds.map(lambda x: self.load_image(x,self.params)).cache(filename='traindataset'+current_time)
#         image_ds = image_ds.shuffle(self.count).map(lambda x:self.preprocess_image(x, self.params)).repeat() 
#         image_ds = image_ds.batch(self.batch_size,drop_remainder=True)
#         image_ds = image_ds.prefetch(buffer_size=AUTOTUNE)
# =============================================================================
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path_ds = tf.data.Dataset.from_tensor_slices(self.all_img_paths).shuffle(self.count)
        image_ds = path_ds.map(lambda x: self.load_image(x,self.params), num_parallel_calls=tf.data.experimental.AUTOTUNE)#.cache(filename='traindataset'+current_time)
        image_ds = image_ds.map(lambda x:self.preprocess_image(x, self.params)).repeat() 
        image_ds = image_ds.batch(self.batch_size,drop_remainder=True)
        image_ds = image_ds.prefetch(buffer_size=AUTOTUNE)
        return image_ds
    def get_val_ds(self):
# =============================================================================
#         current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         val_path_ds = tf.data.Dataset.from_tensor_slices(self.val_all_img_paths)
#         image_ds = val_path_ds.map(lambda x: self.load_image(x,self.params)).cache(filename='valdataset'+current_time)
#         image_ds = image_ds.shuffle(self.val_count).map(lambda x:self.preprocess_image(x, self.params))#.repeat() #
#         image_ds = image_ds.batch(self.batch_size,drop_remainder=True)
#         image_ds = image_ds.prefetch(buffer_size=AUTOTUNE)
# =============================================================================
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        val_path_ds = tf.data.Dataset.from_tensor_slices(self.val_all_img_paths).shuffle(self.val_count)
        image_ds = val_path_ds.map(lambda x: self.load_image(x,self.params), num_parallel_calls=tf.data.experimental.AUTOTUNE)#.cache(filename='traindataset'+current_time)
        image_ds = image_ds.map(lambda x:self.preprocess_image(x, self.params))#.repeat() 
        image_ds = image_ds.batch(self.batch_size,drop_remainder=True)
        image_ds = image_ds.prefetch(buffer_size=AUTOTUNE)
        #suggest
# =============================================================================
#         image_ds = image_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.val_count)).map(lambda x:self.preprocess_image(x, self.params))
#         image_ds = image_ds.batch(self.batch_size)
#         image_ds = image_ds.prefetch(buffer_size=AUTOTUNE)
# =============================================================================
        return image_ds
    def get_real_ds(self):
        path_ds = tf.data.Dataset.from_tensor_slices(self.all_img_paths)
        dataset = path_ds.interleave(map_func=tf.data.TFRecordDataset, cycle_length=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x:self.parse_function(self, x, self.params), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    def get_real_val_ds(self):
        path_ds = tf.data.Dataset.from_tensor_slices(self.val_all_img_paths)
        dataset = path_ds.interleave(map_func=tf.data.TFRecordDataset, cycle_length=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x:self.parse_function(self, x, self.params), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    #@staticmethod
# =============================================================================
#     def burst2patches(self,burst, merged, height, width, depth, burst_length):
#         mosaic, demosaic = self.sample_patch(burst, merged, height, width, burst_length)
#         mosaic = tf.expand_dims(mosaic, axis=0)
#         demosaic = tf.expand_dims(demosaic, axis=0)
#         for i in range(depth-1):
#             m, d = sample_patch(burst, merged, height, width, burst_length)
#             m = tf.expand_dims(m, axis=0)
#             d = tf.expand_dims(d, axis=0)
#             mosaic = tf.concat((mosaic, m), axis=0)
#             demosaic = tf.concat((demosaic, d), axis=0)
#         return mosaic, demosaic
# =============================================================================
    def sample_patch(self,burst, merged, height, width, burst_length):
        y = tf.random_uniform([1], 0, tf.shape(burst)[0]-height, tf.int32)
        x = tf.random_uniform([1], 0, tf.shape(burst)[1]-width, tf.int32)
        y, x = (y[0]//2)*2, (x[0]//2)*2
        mosaic = burst[y:y+height, x:x+width,:burst_length]
        demosaic = merged[y:y+height, x:x+width,:]
        return mosaic, demosaic
    def make_first_truth(self,image, height, width, BURST_LENGTH, to_shift, upscale, jitter):
        j_up = jitter * upscale
        h_up = height * upscale + 2 * j_up
        w_up = width * upscale + 2 * j_up
        v_error = tf.maximum((h_up - tf.shape(image)[0] + 1) // 2, 0)
        h_error = tf.maximum((w_up - tf.shape(image)[1] + 1) // 2, 0)
        image = tf.pad(image, [[v_error, v_error],[h_error,h_error],[0,0]])
        return tf.image.random_crop(image, [h_up, w_up, self.channels])
        
    #@staticmethod
    def make_truth_hqjitter(self,patches, BURST_LENGTH, height, width,\
                            to_shift, upscale, jitter, smalljitter):
        # patches is [h_up, w_up, 3]
        j_up = jitter * upscale
        h_up = height * upscale # + 2 * j_up
        w_up = width * upscale # + 2 * j_up
        bigj_patches = patches
        delta_up = (jitter - smalljitter) * upscale
        smallj_patches = patches[delta_up:-delta_up, delta_up:-delta_up, ...]
        
        curr = [patches[j_up:-j_up, j_up:-j_up, :]]
        prob = tf.minimum(tf.cast(tf.compat.v1.random_poisson(1.5, []), tf.float32)/BURST_LENGTH, 1.)
        for k in range(BURST_LENGTH - 1):
            flip = tf.compat.v1.random_uniform([])
            p2use = tf.cond(flip < prob, lambda : bigj_patches, lambda : smallj_patches)
            curr.append(tf.compat.v1.random_crop(p2use, [h_up, w_up, self.channels]))
        curr = tf.stack(curr, axis=0)
        curr = tf.image.resize(curr, [height, width], method=tf.image.ResizeMethod.AREA)
        curr = tf.transpose(curr, [1,2,3,0])
        return curr
    def add_read_shot_tf(self,truth, sig_read, sig_shot):
        read = sig_read * tf.compat.v1.random_normal(tf.shape(truth))
        shot = tf.sqrt(truth) * sig_shot * tf.compat.v1.random_normal(tf.shape(truth))
        noisy = truth + shot + read
        return noisy, self.batch_down2(tf.sqrt(noisy * sig_shot ** 2 + sig_read ** 2))
    def decimatergb(self,rgb,bl):
        r = rgb[:,::2,::2,...,0]
        g1 = rgb[:,1::2,::2,...,1]
        g2 = rgb[:,::2,1::2,...,1]
        b = rgb[:,1::2,1::2,...,2]
        return self.tf22reshape2(tf.stack([r,g1,g2,b],axis=-1), bl)
    def tf22reshape2(self,t, BURST_LENGTH):
        sh = tf.shape(t)
        t = tf.reshape(t, (sh[0], sh[1], sh[2], BURST_LENGTH, 2, 2))
        t = tf.transpose(t, (0, 1, 4, 2, 5, 3))
        t = tf.reshape(t, (sh[0], sh[1]*2, sh[2]*2, BURST_LENGTH))
        return t
    def batch_down2(self,img):
        return (img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])/4
   




# =============================================================================
#     def show(self, ds, num=2):
#         for data, label in ds.take(num):
#             print(self.label_names[label.numpy()[0]])
#             plt.imshow(data.numpy()[0, :, :, :])
#             plt.show()
# =============================================================================
# =============================================================================
#         for i in range(unique):
#             curr = [patches[i, j_up:-j_up, j_up:-j_up, :]]
#             prob = tf.minimum(tf.cast(tf.random_poisson(1.5, []), tf.float32)/BURST_LENGTH, 1.)
#             for k in range(BURST_LENGTH - 1):
#                 flip = tf.random_uniform([])
#                 p2use = tf.cond(flip < prob, lambda : bigj_patches, lambda : smallj_patches)
#                 curr.append(tf.compat.v1.random_crop(p2use[i, ...], [h_up, w_up, 3]))
#             curr = tf.stack(curr, axis=0)
#             curr = tf.image.resize_images(curr, [height, width], method=tf.image.ResizeMethod.AREA)
#             curr = tf.transpose(curr, [1,2,3,0])
#             batch.append(curr)
#         batch = tf.stack(batch, axis=0)
#         return batch
# =============================================================================
# =============================================================================
#     def show(self, ds, num=2):
#         for data, label in ds.take(num):
#             print(self.label_names[label.numpy()[0]])
#             plt.imshow(data.numpy()[0, :, :, :])
#             plt.show()
# 
#     def write_record(self, record_name):
#         ds_image = tf.data.Dataset.from_tensor_slices(
#             self.all_img_paths).map(tf.io.read_file)
#         record = tf.data.experimental.TFRecordWriter(record_name)
#         record.write(ds_image)
#         print('record saved in {}'.format(record_name))
# 
#     def read_record(self, record_name):
#         image_ds = tf.data.TFRecordDataset(
#             record_name).map(self.preprocess_image)
#         label_ds = tf.data.Dataset.from_tensor_slices(self.all_img_labels)
#         ds = tf.data.Dataset.zip((image_ds, label_ds))
#         ds = ds.repeat()
#         # ds = ds.apply(tf.data.experimental.shuffle_and_repeat(
#         #     buffer_size=10))
#         ds = ds.batch(self.batch_size)
#         return ds
# =============================================================================


# =============================================================================
# params = {
#         "train_path":'G:\\master_thesis\Cengiz\BurstDenoising\Testdata',
#         "batch_size":1,
#         "color":False,
#         "height":64,
#         "width":64,
#         "degamma":2.2,
#         "to_shift":1.,
#         "upscale":4,
#         "jitter":16,
#         "smalljitter":2,
#         "BURST_LENGTH":8
#         }
# image_ds = DataLoader(params).get_ds()
# plt.figure(figsize=(8,8))
# for epoch in range(2):
#     for n, image in enumerate(image_ds):
#         print(n)
#         noise,truth=image
#         gt = truth[...,0]
#         plt.subplot(4,4,epoch*8+n+1)
#         plt.imshow(gt[0,...])
# =============================================================================
# =============================================================================
#     for i in range(8):
#         plt.subplot(9,1,i+2)
#         plt.imshow(noise[2,...,i])
#     print(tf.shape(truth))
#     print(tf.shape(noise))
# =============================================================================
    
 