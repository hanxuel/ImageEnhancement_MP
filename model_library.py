# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:26:01 2020

@author: lenovo
"""
import pydot
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from data_utils import *
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

  def __init__(self,
               latent_dim=32,
               intermediate_dim=64,
               name='encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_mean = layers.Dense(latent_dim)
    self.dense_log_var = layers.Dense(latent_dim)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.dense_proj(inputs)
    z_mean = self.dense_mean(x)
    z_log_var = self.dense_log_var(x)
    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var, z


class Decoder(layers.Layer):
  """Converts z, the encoded digit vector, back into a readable digit."""

  def __init__(self,
               original_dim,
               intermediate_dim=64,
               name='decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_output = layers.Dense(original_dim, activation='sigmoid')

  def call(self, inputs):
    x = self.dense_proj(inputs)
    return self.dense_output(x)

class Downblock(layers.Layer):
    def __init__(self,
                 intermediate_dim=64,
                 regu=0.001,
                 name='downblock',
                 **kwargs):
        super(Downblock, self).__init__(name=name, **kwargs)
        self.conv2d1 = layers.Conv2D(intermediate_dim,3, padding="same", activation='relu',kernel_regularizer=regularizers.l2(regu))
        self.conv2d2 = layers.Conv2D(intermediate_dim,3, padding="same", activation='relu',kernel_regularizer=regularizers.l2(regu))
        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='valid')
    def call(self, inputs):
        inputs = self.conv2d1(inputs)
        output1 = self.conv2d2(inputs)
        output2 = self.maxpool(output1)
        return output1,output2

class Upblock(layers.Layer):
    def __init__(self,
                 intermediate_dim=512,
                 name='upblock',
                 s=2,
                 regu=0.001,
                 **kwargs):
        super(Upblock, self).__init__(name=name, **kwargs)
        self.conv2d1 = layers.Conv2D(intermediate_dim,3, padding="same",kernel_regularizer=regularizers.l2(regu), activation='relu')
        self.conv2d2 = layers.Conv2D(intermediate_dim,3, padding="same",kernel_regularizer=regularizers.l2(regu), activation='relu')
        self.conv2d3 = layers.Conv2D(intermediate_dim,3, padding="same", kernel_regularizer=regularizers.l2(regu),activation='relu')
        self.upsampling = layers.UpSampling2D(size=(s, s),interpolation='bilinear')
    def call(self, inputs,skip):
        inputs = self.upsampling(inputs)
        #inputs = self.conv2d1(inputs)
        output1 = layers.concatenate([inputs, skip],axis=-1)
        output2 = self.conv2d1(output1)
        #print(tf.shape(output))
        output3 = self.conv2d2(output2)
        output = self.conv2d3(output3)
        return output
class Poolskip(layers.Layer):
    def __init__(self,
                 k=2,
                 name='poolskip',
                 **kwargs):
        super(Poolskip, self).__init__(name=name, **kwargs)
        self.k=k
    def call(self, inputs):
        inputs = layers.GlobalAveragePooling2D()(inputs)
        inputs = tf.expand_dims(tf.expand_dims(inputs,1),1)
        poolskip = tf.tile(inputs,[1,self.k,self.k,1])
        return poolskip
class Convolve(layers.Layer):
    def __init__(self,final_K,
                 name='convolve',
                 **kwargs):
        super(Convolve, self).__init__(name=name, **kwargs)
        self.final_K = final_K
    def call(self, img_stack, filts):
        initial_W = img_stack.get_shape().as_list()[-1]
        fsh = tf.shape(filts)
        filts = tf.reshape(filts, [fsh[0], fsh[1], fsh[2], self.final_K ** 2 * initial_W])
    
        kpad = self.final_K//2
        imgs = tf.pad(img_stack, [[0,0],[kpad,kpad],[kpad,kpad],[0,0]])
        ish = tf.shape(img_stack)
        img_stack = []
        for i in range(self.final_K):
            for j in range(self.final_K):
                img_stack.append(imgs[:, i:tf.shape(imgs)[1]-2*kpad+i, j:tf.shape(imgs)[2]-2*kpad+j, :])
        img_stack = tf.stack(img_stack, axis=-2)
        img_stack = tf.reshape(img_stack, [ish[0], ish[1], ish[2], self.final_K**2 * initial_W])
        img_net = tf.reduce_sum(img_stack * filts, axis=-1)
        return img_net
def cus_convolve(img_stack, filts, final_K):
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
    #batch_size*H*w
    return img_net
class Convolve_perlayer(layers.Layer):
    def __init__(self,final_K,burst_length,
                 name='convolve_perlayer',
                 **kwargs):
        super(Convolve_perlayer, self).__init__(name=name, **kwargs)
        self.final_K = final_K
        self.burst_length = burst_length
    def call(self, conv_stack, filts):
        initial_W = conv_stack.get_shape().as_list()[-1]
        img_net = []
        for i in range(initial_W):
            onepiece = cus_convolve(conv_stack[...,i:i+1], filts[...,i:i+1], self.final_K)*initial_W 
            img_net.append(tf.expand_dims(onepiece,-1))
        img_net = tf.concat(img_net, axis=-1)
        #batch_size*H*w*burst_length
        return img_net
# =============================================================================
# def convolve_per_layer(conv_stack, filts, final_K, final_W):
#   initial_W = conv_stack.get_shape().as_list()[-1]
#   img_net = []
#   for i in range(initial_W):
#     img_net.append(convolve(conv_stack[...,i:i+1], filts[...,i:i+1,:], final_K, final_W))
#   img_net = tf.concat(img_net, axis=-1)
#   return img_net
# =============================================================================

class Basis_kpn(tf.keras.Model):
    def __init__(self,
                 params,
                 name='basis_kpn',
                 **kwargs):
        super(Basis_kpn, self).__init__(name=name, **kwargs)
        self.burst_length=params["BURST_LENGTH"]
        self.K = params['Kernel_size']
        self.height = params['height']
        self.width = params['width']
        self.B=params['Basis_num']
        self.layer0 = layers.Conv2D(64,3, padding="same", activation='relu',input_shape=(self.height, self.width, self.burst_length),name="weight0")
        self.down1 = Downblock(intermediate_dim=64, name='downblock1')
        self.down2 = Downblock(intermediate_dim=128, name='downblock2')
        self.down3 = Downblock(intermediate_dim=256, name='downblock3')
        self.down4 = Downblock(intermediate_dim=512, name='downblock4')
        self.down5 = Downblock(intermediate_dim=1024, name='downblock5')
        
        self.layer1_1 = layers.Conv2D(1024,3, padding="same", activation='relu',name="weight1_1")
        self.layer1_2 = layers.Conv2D(1024,3, padding="same", activation='relu',name="weight1_2")
        
        self.Coef_up1 = Upblock(intermediate_dim=512, name='Coef_upblock1')
        self.Coef_up2 = Upblock(intermediate_dim=256, name='Coef_upblock2')
        self.Coef_up3 = Upblock(intermediate_dim=128, name='Coef_upblock3')
        self.Coef_up4 = Upblock(intermediate_dim=64, name='Coef_upblock4')
        self.Coef_up5 = Upblock(intermediate_dim=64, name='Coef_upblock5')
        
        self.layer2_1 = layers.Conv2D(64,3, padding="same", activation='relu',name="weight2_1")
        self.layer2_2 = layers.Conv2D(64,3, padding="same", activation='relu',name="weight2_2")
        self.coef = layers.Conv2D(self.B,3, padding="same", activation='relu',name="coef")
        
        self.pool_skip1 = Poolskip(k=2, name="pool_skip1")
        self.pool_skip2 = Poolskip(k=4, name="pool_skip2")
        self.pool_skip3 = Poolskip(k=8, name="pool_skip3")
        self.pool_skip4 = Poolskip(k=16, name="pool_skip4")
        self.Basis_up1 = Upblock(intermediate_dim=512, name='Basis_up1')
        self.Basis_up2 = Upblock(intermediate_dim=256, name='Basis_up2')
        self.Basis_up3 = Upblock(intermediate_dim=256, name='Basis_up3')
        self.Basis_up4 = Upblock(intermediate_dim=128, name='Basis_up4')
        
        self.layer3_1 = layers.Conv2D(128,2, padding="valid", activation='relu',name="weight3_1")
        self.layer3_2 = layers.Conv2D(128,3, padding="same", activation='relu',name="weight3_2")
        self.layer3_3 = layers.Conv2D(self.burst_length*self.B,3, padding="same", activation='relu',name="weight3_3")
        
        self.convolve = Convolve(self.K,name ="convolve")
        self.convolve_perlayer = Convolve_perlayer(self.K,self.burst_length,name ="convolve_perlayer")
    def call(self, inputs):
        inputs0 = self.layer0(inputs)
        #encoder part
        skip1, output1 = self.down1(inputs0)
        skip2, output2 = self.down2(output1)
        skip3, output3 = self.down3(output2)
        skip4, output4 = self.down4(output3)
        skip5, output5 = self.down5(output4)
        
        output6_1 = self.layer1_1(output5)
        output6 = self.layer1_2(output6_1)
        #coefficent branch
        Up5 = self.Coef_up1(output6,skip5)
        Up4 = self.Coef_up2(Up5,skip4)
        Up3 = self.Coef_up3(Up4,skip3)
        Up2 = self.Coef_up4(Up3,skip2)
        Up1 = self.Coef_up5(Up2,skip1)
        
        Output7_1 = self.layer2_1(Up1)
        Output7 = self.layer2_2(Output7_1)
        Coef = self.coef(Output7)
        Coef = tf.nn.softmax(Coef, axis=-1, name="Coef_softmax")
        print("Coef.shape",tf.shape(Coef))
        #basis branch
        Global_average_col = tf.reduce_mean(output6, axis=1, keepdims=True, name="Global_average_col")
        Global_average = tf.reduce_mean(Global_average_col, axis=2, keepdims=True, name="gloval_average")
        poolskip5 = self.pool_skip1(skip5)
        #print("gloval_average.shape",tf.shape(Global_average))
        #print("tf.shape(poolskip5)",tf.shape(poolskip5))
        Upbasis5 = self.Basis_up1(Global_average, poolskip5)
        poolskip4 = self.pool_skip2(skip4)
        Upbasis4 = self.Basis_up2(Upbasis5, poolskip4)
        poolskip3 = self.pool_skip3(skip3)
        Upbasis3 = self.Basis_up3(Upbasis4, poolskip3)
        poolskip2 = self.pool_skip4(skip2)
        Upbasis2 = self.Basis_up4(Upbasis3, poolskip2)
        
        Output8_1 = self.layer3_1(Upbasis2)
        Output8_2 = self.layer3_2(Output8_1)
        Basis = self.layer3_3(Output8_2)
        
        
        ish = tf.shape(Basis)
        #print("basis.shape",ish)
        Basis = tf.nn.softmax(tf.reshape(Basis,[ish[0],self.K**2*self.burst_length,self.B]),axis=1)
        Basis = tf.reshape(Basis,[ish[0],self.K,self.K,self.burst_length,self.B])
        
        Coefficients = tf.expand_dims(tf.expand_dims(tf.expand_dims(Coef,3),3), 3)
        Coefficients = tf.tile(Coefficients,[1,1,1,self.K,self.K,self.burst_length,1])
        #print("Coefficients.shape",tf.shape(Coefficients))
        Basis = tf.expand_dims(tf.expand_dims(Basis,1),1)
        Basis = tf.tile(Basis,[1,self.height,self.width,1,1,1,1])
        filts = tf.reduce_sum(Basis*Coefficients,axis=-1)
        #print("input.shape",tf.shape(inputs))
        #print("filts.shape",tf.shape(filts))
        Deblur = tf.expand_dims(self.convolve(inputs, filts),-1)
        #batch_size*H*W*1
        Deblur_perburst = self.convolve_perlayer(inputs, filts)
        #batch_size*H*W*burst_length
        output = tf.concat([Deblur,Deblur_perburst], axis=-1)
        return output 



# =============================================================================
# height=128,
# width=128,
# burst_length=8,
# B=90,
# K=15,
# =============================================================================
class Simplemodel(tf.keras.Model):
    def __init__(self,
                 params,
                 name='simple_kpn',
                 **kwargs):
        super(Simplemodel, self).__init__(name=name, **kwargs)
        self.burst_length=params["BURST_LENGTH"]
        self.K = params['Kernel_size']
        self.height = params['height']
        self.width = params['width']
        self.B=params['Basis_num']
        self.regu=params['regu']
        self.layer0 = layers.Conv2D(64,3, padding="same", activation='relu',kernel_regularizer=regularizers.l2(self.regu),input_shape=(self.height, self.width, self.burst_length),name="weight0")
        self.down1 = Downblock(intermediate_dim=64, name='downblock1')
        self.down2 = Downblock(intermediate_dim=128, name='downblock2')
# =============================================================================
#         self.down3 = Downblock(intermediate_dim=256, name='downblock3')
#         self.down4 = Downblock(intermediate_dim=512, name='downblock4')
# =============================================================================
        self.down5 = Downblock(intermediate_dim=1024, name='downblock5')
        
        self.layer1_1 = layers.Conv2D(1024,3, padding="same", activation='relu',kernel_regularizer=regularizers.l2(self.regu),name="weight1_1")
# =============================================================================
#         self.layer1_2 = layers.Conv2D(1024,3, padding="same", activation='relu',name="weight1_2")
# =============================================================================
        
        self.Coef_up1 = Upblock(intermediate_dim=512, name='Coef_upblock1')
# =============================================================================
#         self.Coef_up2 = Upblock(intermediate_dim=256, name='Coef_upblock2')
#         self.Coef_up3 = Upblock(intermediate_dim=128, name='Coef_upblock3')
# =============================================================================
        self.Coef_up4 = Upblock(intermediate_dim=64, name='Coef_upblock4')
        self.Coef_up5 = Upblock(intermediate_dim=64, name='Coef_upblock5')
        
        self.layer2_1 = layers.Conv2D(64,3, padding="same", activation='relu',kernel_regularizer=regularizers.l2(self.regu),name="weight2_1")
# =============================================================================
#         self.layer2_2 = layers.Conv2D(64,3, padding="same", activation='relu',name="weight2_2")
# =============================================================================
        self.coef = layers.Conv2D(self.B,3, padding="same", activation='relu',kernel_regularizer=regularizers.l2(self.regu),name="coef")
        
        self.pool_skip1 = Poolskip(k=2, name="pool_skip1")
# =============================================================================
#         self.pool_skip2 = Poolskip(k=4, name="pool_skip2")
#         self.pool_skip3 = Poolskip(k=8, name="pool_skip3")
# =============================================================================
        self.pool_skip4 = Poolskip(k=16, name="pool_skip4")
        self.Basis_up1 = Upblock(intermediate_dim=512, name='Basis_up1')
# =============================================================================
#         self.Basis_up2 = Upblock(intermediate_dim=256, name='Basis_up2')
#         self.Basis_up3 = Upblock(intermediate_dim=256, name='Basis_up3')
# =============================================================================
        self.Basis_up4 = Upblock(intermediate_dim=128, s=8, name='Basis_up4')
        
        self.layer3_1 = layers.Conv2D(128,2, padding="valid", activation='relu',kernel_regularizer=regularizers.l2(self.regu),name="weight3_1")
# =============================================================================
#         self.layer3_2 = layers.Conv2D(128,3, padding="same", activation='relu',name="weight3_2")
# =============================================================================
        self.layer3_3 = layers.Conv2D(self.burst_length*self.B,3, padding="same", activation='relu',kernel_regularizer=regularizers.l2(self.regu),name="weight3_3")
        
        self.convolve = Convolve(self.K,name ="convolve")
        self.convolve_perlayer = Convolve_perlayer(self.K,self.burst_length, name ="convolve_perlayer")
    def call(self, inputs):
        inputs0 = self.layer0(inputs)
        #encoder part
        skip1, output1 = self.down1(inputs0)
        skip2, output2 = self.down2(output1)
# =============================================================================
#         skip3, output3 = self.down3(output2)
#         skip4, output4 = self.down4(output3)
# =============================================================================
        skip5, output5 = self.down5(output2)
        
        output6_1 = self.layer1_1(output5)
# =============================================================================
#         output6 = self.layer1_2(output6_1)
# =============================================================================
        #coefficent branch
        Up5 = self.Coef_up1(output6_1,skip5)
# =============================================================================
#         Up4 = self.Coef_up2(Up5,skip4)
#         Up3 = self.Coef_up3(Up4,skip3)
# =============================================================================
        #print(tf.shape(Up5))
        #print(tf.shape(skip2))
        Up2 = self.Coef_up4(Up5,skip2)
        Up1 = self.Coef_up5(Up2,skip1)
        
        Output7_1 = self.layer2_1(Up1)
# =============================================================================
#         Output7 = self.layer2_2(Output7_1)
# =============================================================================
        Coef = self.coef(Output7_1)
        Coef = tf.nn.softmax(Coef, axis=-1, name="Coef_softmax")
        #print("Coef.shape",tf.shape(Coef))
        #basis branch
        Global_average_col = tf.reduce_mean(output6_1, axis=1, keepdims=True, name="Global_average_col")
        Global_average = tf.reduce_mean(Global_average_col, axis=2, keepdims=True, name="gloval_average")
        poolskip5 = self.pool_skip1(skip5)
        #print("gloval_average.shape",tf.shape(Global_average))
        #print("tf.shape(poolskip5)",tf.shape(poolskip5))
        Upbasis5 = self.Basis_up1(Global_average, poolskip5)
# =============================================================================
#         poolskip4 = self.pool_skip2(skip4)
#         Upbasis4 = self.Basis_up2(Upbasis5, poolskip4)
#         poolskip3 = self.pool_skip3(skip3)
#         Upbasis3 = self.Basis_up3(Upbasis4, poolskip3)
# =============================================================================
        poolskip2 = self.pool_skip4(skip2)
        Upbasis2 = self.Basis_up4(Upbasis5, poolskip2)
        
        Output8_1 = self.layer3_1(Upbasis2)
# =============================================================================
#         Output8_2 = self.layer3_2(Output8_1)
# =============================================================================
        Basis = self.layer3_3(Output8_1)
# =============================================================================
#         print(tf.shape(Basis))
#         print(tf.shape(Coef))
# =============================================================================
        
        ish = tf.shape(Basis)
        #print("basis.shape",ish)
        Basis = tf.nn.softmax(tf.reshape(Basis,[ish[0],self.K**2*self.burst_length,self.B]),axis=1)
        Basis = tf.reshape(Basis,[ish[0],self.K,self.K,self.burst_length,self.B])
        
        Coefficients = tf.expand_dims(tf.expand_dims(tf.expand_dims(Coef,3),3), 3)
        Coefficients = tf.tile(Coefficients,[1,1,1,self.K,self.K,self.burst_length,1])
        #print("Coefficients.shape",tf.shape(Coefficients))
        Basis = tf.expand_dims(tf.expand_dims(Basis,1),1)
        Basis = tf.tile(Basis,[1,self.height,self.width,1,1,1,1])
        filts = tf.reduce_sum(Basis*Coefficients,axis=-1)
        #print("input.shape",tf.shape(inputs))
        #print("filts.shape",tf.shape(filts))
        Deblur = tf.expand_dims(self.convolve(inputs, filts),-1)
        #batch_size*H*W*1
        Deblur_perburst = self.convolve_perlayer(inputs, filts)
        #batch_size*H*W*burst_length
        output = tf.concat([Deblur,Deblur_perburst], axis=-1)
        return output 

# =============================================================================
# params = {
#         "root":'C:\\Users\lenovo\Desktop\Validation',
#         "batch_size":3,
#         "color":False,
#         "height":128,
#         "width":128,
#         "degamma":2.2,
#         "to_shift":1.,
#         "upscale":4,
#         "jitter":16,
#         "smalljitter":2,
#         "BURST_LENGTH":2
#         }
# model=Testmodel(height=16,width=16,burst_length=2,B=30,K=15)
# 
# image_ds = DataLoader(params).get_ds()
# for n, image in enumerate(image_ds.take(1)):
#     print(n)
#     noise, truth=image
#     print(tf.shape(truth))
#     plt.figure(figsize=(16,16))
#     for i in range(3):
#         plt.subplot(2,3,i+1)
#         plt.imshow(truth[i,...,0])
#     #truth = tf.convert_to_tensor(truth)
#     #print(truth[...,1])
#     #print(np.mean(truth[...,1]))
#     white_noise=tf.expand_dims(truth[...,1],axis=-1)
#     white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
#     print(tf.shape(white_noise))
#     gt = truth[...,0] #tf.expand_dims(truth[...,0],axis=-1)
#     print(tf.shape(gt))
#     invert = invert_preproc(gt,white_noise)#.numpy()
#     print("invert",tf.shape(invert))
#     for i in range(3):
#         plt.subplot(2,3,i+4)
#         plt.imshow(invert[i,...])
# =============================================================================




    