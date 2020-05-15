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
def custom_loss(y_true, y_pred):
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
        loss0 = loss0+ 0.5 * basic_img_loss(invert_perlayer['da{}_noshow'.format(i)],invert_gt) 
    return loss0

def psnr_tf(estimate, truth):
  return -10. * tf.log(tf.reduce_mean(tf.square(estimate - truth))) / tf.math.log(10.)


def psnr_tf_batch(estimate, truth):
    return tf.reduce_mean(-10. * tf.math.log(tf.reduce_mean(tf.reshape(tf.square(estimate - truth), [tf.shape(estimate)[0],-1]), axis=1)) / tf.math.log(10.))

def psnr_deblur(y_true, y_pred):
    white_noise=tf.expand_dims(y_true[...,1],axis=-1)
    white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
    gt = y_true[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
    invert_gt = invert_preproc(gt,white_noise)#.numpy()
    Deblur = y_pred[...,0]#tf.expand_dims(y_pred[...,0],axis=-1)
    invert_deblur = invert_preproc(Deblur, white_noise)
    return psnr_tf_batch(invert_deblur, invert_gt)
def psnr_each_layer(y_true, y_pred):
    burst_size = tf.shape(y_pred)[-1]-1
    white_noise=tf.expand_dims(y_true[...,1],axis=-1)
    white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
    gt = y_true[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
    invert_gt = invert_preproc(gt,white_noise)#.numpy()
    invert_perlayer={}
    psnr = []
    for i in range(burst_size):
        invert_perlayer['da{}_noshow'.format(i)]=invert_preproc(y_pred[...,i+1],white_noise)
        psnr.append(psnr_tf_batch(invert_perlayer['da{}_noshow'.format(i)],invert_gt)) 
    return psnr

class DataLoader():
    def __init__(self, params):
        self.params=params
        self.root = params["root"]
        self.batch_size = params["batch_size"]
        data_root = pathlib.Path(self.root)

        self.all_img_paths = list(data_root.glob('*.jpg'))
        self.all_img_paths = [str(path) for path in self.all_img_paths]
        self.count = len(self.all_img_paths)
        random.shuffle(self.all_img_paths)
        self.color = params["color"]
        #BURST_LENGTH=8

    @staticmethod
    def preprocess_image(self,image,params):
        BURST_LENGTH=params["BURST_LENGTH"]
        height=params["height"]
        width=params["width"]
        degamma=params["degamma"]
        to_shift=params["to_shift"]
        upscale=params["upscale"]
        jitter=params["jitter"]
        smalljitter=params["smalljitter"]
        image = tf.image.decode_jpeg(image, channels=3)
        patches = self.make_stack_hqjitter((tf.cast(image, tf.float32) / 255.)**degamma,\
                                                 height, width, BURST_LENGTH, to_shift, upscale, jitter)
        #PATCHES ================= [1, 256, 256, 3]
        demosaic_truth = self.make_batch_hqjitter(patches, BURST_LENGTH, height, width, to_shift, upscale, jitter, smalljitter)
        
        if not self.color:
            demosaic_truth = tf.reduce_mean(demosaic_truth, axis=-2)
            
        truth_all = demosaic_truth

        degamma = 1.
        white_level = tf.pow(10., tf.compat.v1.random_uniform([1, 1, 1], np.log10(.1), np.log10(1.)))
        if self.color:
          white_level = tf.expand_dims(white_level, axis=-1)
        demosaic_truth = (white_level * demosaic_truth ** degamma)
        
        sig_read = tf.pow(10., tf.compat.v1.random_uniform([1, 1, 1], -3., -1.5))
        sig_shot = tf.pow(10., tf.compat.v1.random_uniform([1, 1, 1], -2., -1.))
        dec = demosaic_truth
        #dec = self.decimatergb(demosaic_truth, BURST_LENGTH) if self.color else demosaic_truth
#        print 'DEC',dec.get_shape().as_list()
        noisy_, _ = self.add_read_shot_tf(dec, sig_read, sig_shot)
        noisy = noisy_

        first2 = demosaic_truth[...,:2]
        demosaic_truth = demosaic_truth[...,0,:] if self.color else demosaic_truth[...,0]
        demosaic_truth = tf.expand_dims(demosaic_truth,2)
        white_level = tf.tile(white_level,[height,width,1])
        return noisy,tf.concat([demosaic_truth,white_level],axis=-1)
    
    @staticmethod
    def load_and_preprocess_image(self,path,params):
        image = tf.io.read_file(path)
        #DataLoader.preprocess_image(self,image,params)
        return DataLoader.preprocess_image(self,image,params)
    def get_ds(self):
        path_ds = tf.data.Dataset.from_tensor_slices(self.all_img_paths)
        #label_ds = tf.data.Dataset.from_tensor_slices(self.all_img_labels)
        image_ds = path_ds.map(lambda x:self.load_and_preprocess_image(self, x, self.params))
        image_ds = image_ds.shuffle(self.count).repeat(3).batch(self.batch_size)
        
        #suggest
        #image_ds = image_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.count))
        #image_ds = image_ds.batch(self.batch_size)
        return image_ds

    #@staticmethod
    def make_stack_hqjitter(self,image, height, width, BURST_LENGTH, to_shift, upscale, jitter):
        j_up = jitter * upscale
        h_up = height * upscale + 2 * j_up
        w_up = width * upscale + 2 * j_up
        v_error = tf.maximum((h_up - tf.shape(image)[0] + 1) // 2, 0)
        h_error = tf.maximum((w_up - tf.shape(image)[1] + 1) // 2, 0)
        image = tf.pad(image, [[v_error, v_error],[h_error,h_error],[0,0]])
        return tf.image.random_crop(image, [h_up, w_up, 3])
        
    #@staticmethod
    def make_batch_hqjitter(self,patches, BURST_LENGTH, height, width,\
                            to_shift, upscale, jitter, smalljitter):
        # patches is [BURST_LENGTH, h_up, w_up, 3]
        #jiang every patches repeat two times in each batch
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
            curr.append(tf.compat.v1.random_crop(p2use, [h_up, w_up, 3]))
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
# 
# params = {
#         "root":'G:\\master_thesis\Cengiz\BurstDenoising\SmallValidationdata',
#         "batch_size":1,
#         "color":False,
#         "height":128,
#         "width":128,
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
    
    
    
# =============================================================================
# class DataLoader():
#     BURST_LENGTH=8
#     def __init__(self, params):
#         self.root = params["root"]
#         self.batch_size = params["batch_size"]
#         data_root = pathlib.Path(self.root)
# 
#         self.all_img_paths = list(data_root.glob('*.jpg'))
#         self.all_img_paths = [str(path) for path in self.all_img_paths]
#         self.count = len(self.all_img_paths)
#         random.shuffle(self.all_img_paths)
#         self.color = params["color"]
#         #BURST_LENGTH=8
# 
#     @staticmethod
#     def preprocess_image(image):
#         #BURST_LENGTH=8
#         height=128
#         width=128
#         degamma=2.2
#         to_shift=1.
#         upscale=4
#         jitter=16
#         smalljitter=2
#         image = tf.image.decode_jpeg(image, channels=3)
#         patches = DataLoader.make_stack_hqjitter((tf.cast(image, tf.float32) / 255.)**degamma,\
#                                                  height, width, DataLoader.BURST_LENGTH, to_shift, upscale, jitter)
#         #PATCHES ================= [1, 256, 256, 3]
#         patches = DataLoader.make_batch_hqjitter(patches, DataLoader.BURST_LENGTH, height, width, to_shift, upscale, jitter, smalljitter)
#   
#         return patches
#     
#     @staticmethod
#     def load_and_preprocess_image(path):
#         image = tf.io.read_file(path)
#         return DataLoader.preprocess_image(image)
#     @staticmethod
#     def make_stack_hqjitter(image, height, width, BURST_LENGTH, to_shift, upscale, jitter):
#         j_up = jitter * upscale
#         h_up = height * upscale + 2 * j_up
#         w_up = width * upscale + 2 * j_up
#         v_error = tf.maximum((h_up - tf.shape(image)[0] + 1) // 2, 0)
#         h_error = tf.maximum((w_up - tf.shape(image)[1] + 1) // 2, 0)
#         image = tf.pad(image, [[v_error, v_error],[h_error,h_error],[0,0]])
#         return tf.image.random_crop(image, [h_up, w_up, 3])
#         
#     @staticmethod
#     def make_batch_hqjitter(patches, BURST_LENGTH, height, width,\
#                             to_shift, upscale, jitter, smalljitter):
#         # patches is [BURST_LENGTH, h_up, w_up, 3]
#         #jiang every patches repeat two times in each batch
#         j_up = jitter * upscale
#         h_up = height * upscale # + 2 * j_up
#         w_up = width * upscale # + 2 * j_up
#         bigj_patches = patches
#         delta_up = (jitter - smalljitter) * upscale
#         smallj_patches = patches[delta_up:-delta_up, delta_up:-delta_up, ...]
#         
#         curr = [patches[j_up:-j_up, j_up:-j_up, :]]
#         prob = tf.minimum(tf.cast(tf.compat.v1.random_poisson(1.5, []), tf.float32)/BURST_LENGTH, 1.)
#         for k in range(BURST_LENGTH - 1):
#             flip = tf.compat.v1.random_uniform([])
#             p2use = tf.cond(flip < prob, lambda : bigj_patches, lambda : smallj_patches)
#             curr.append(tf.compat.v1.random_crop(p2use, [h_up, w_up, 3]))
#         curr = tf.stack(curr, axis=0)
#         curr = tf.image.resize(curr, [height, width], method=tf.image.ResizeMethod.AREA)
#         curr = tf.transpose(curr, [1,2,3,0])
#         return curr
#     def get_ds(self):
#         path_ds = tf.data.Dataset.from_tensor_slices(self.all_img_paths)
#         #label_ds = tf.data.Dataset.from_tensor_slices(self.all_img_labels)
#         image_ds = path_ds.map(self.load_and_preprocess_image)
#         image_ds = image_ds.shuffle(self.count).batch(self.batch_size)
#         
# # =============================================================================
# #         #suggest
# #         image_ds = image_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.count))
# #         image_ds = image_ds.batch(self.batch_size)
# # =============================================================================
#         return image_ds
# =============================================================================