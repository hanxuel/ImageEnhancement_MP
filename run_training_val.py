# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:29:32 2020

@author: lenovo
"""
import pydot
import datetime,time
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
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="G:\\master_thesis\Cengiz\BurstDenoising\SmallTraindatagray")
    parser.add_argument("--val_path", default="G:\\master_thesis\Cengiz\BurstDenoising\SmallValidationgray")
    #parser.add_argument("--model_path", required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--BURST_LENGTH", type=int, default=2)
    parser.add_argument("--Kernel_size", type=int, default=15)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--jitter", type=int, default=16)
    parser.add_argument("--smalljitter", type=int, default=2)
    parser.add_argument("--Basis_num", type=int, default=10)
    parser.add_argument("--color", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--nepochs", type=int, default=50)
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
def main():
    args = parse_args()

    np.random.seed(0)
    tf.random.set_seed(0)

    #tf.logging.set_verbosity(tf.logging.INFO)
    params = {
        "train_path":args.train_path,
        "val_path":args.val_path,
        "batch_size":args.batch_size,
        "color":args.color,
        "height":args.height,
        "width":args.width,
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
    #log_dir = '.\\logs\\gradient_tape\\' + current_time + '\\train'
    log_dir = '.\\logs\\gradient_tape\\' + '20200523-220530' + '\\train'
    anneal = params["anneal"]
    summary_writer = tf.summary.create_file_writer(log_dir)
    model=Simplemodel(params)    
    train_image_ds = DataLoader(params).get_ds()
    val_image_ds = DataLoader(params).get_val_ds()
    
    epochs = params["nepochs"]
# =============================================================================
#     STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
#     lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001,\
#                                                                  decay_steps=STEPS_PER_EPOCH*1000,
#                                                                  decay_rate=1,
#                                                                  staircase=False)
#     def get_optimizer():
#         return tf.keras.optimizers.Adam(lr_schedule)
# =============================================================================
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    loss_metric = tf.keras.metrics.Mean()
    
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), iterate=tf.Variable(0,trainable=False), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, params["ckpt"], max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    
# =============================================================================
#     TensorBoardcallback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
#                                                          histogram_freq=1,
#                                                          write_graph=True, write_grads=False, write_images=True,
#                                                          embeddings_freq=0, embeddings_layer_names=None,
#                                                          embeddings_metadata=None, embeddings_data=None, update_freq=500
#                                                          )
#     model.compile(optimizer=optimizer,loss=deblur_loss,\
#                   metrics=[psnr_deblur])
#     model.fit(image_ds, epochs=params["nepochs"],\
#               callbacks=[TensorBoardcallback])
# =============================================================================
    # Iterate over epochs.
    
    for epoch in range(epochs):
        start = time.clock()
        print('Start of epoch %d' %(epoch))
        # Iterate over the batches of the dataset.
        psnr = []
        psnr_perlayer=[[0]*1 for i in range(params["BURST_LENGTH"])]
        psnr_noise0 = []
        psnr_average = []
        for step, x_batch_train in enumerate(train_image_ds):
            x_batch_burst, x_batch_truth=x_batch_train
            with tf.GradientTape() as tape:
                reconstructed = model(x_batch_burst)
                #print("tf.shape(reconstructed)",tf.shape(reconstructed))
                # Compute reconstruction loss
                anneal_coef = tf.pow(anneal, tf.cast(ckpt.iterate,tf.float32))*(10**2)
                loss = deblur_layer_loss(x_batch_truth, reconstructed,anneal_coef)
                #print("model.losses------------------",model.losses)
                #loss = loss+sum(model.losses)  # Add KLD regularization loss
                
                white_noise=tf.expand_dims(x_batch_truth[...,1],axis=-1)
                white_noise = tf.reduce_mean(tf.reduce_mean(white_noise,axis=1,keepdims=True),axis=2,keepdims=True)
                gt = x_batch_truth[...,0] #tf.expand_dims(y_true[...,0],axis=-1)
                invert_gt = invert_preproc(gt,white_noise)#.numpy()
                
                Deblur = reconstructed[...,0]#tf.expand_dims(y_pred[...,0],axis=-1)
                invert_deblur = invert_preproc(Deblur, white_noise)
                
                onestep_psnr = psnr_deblur(invert_deblur, invert_gt)
                psnr.append(onestep_psnr.numpy())
                
                onestep_psnr_perlayer = psnr_each_layer(invert_gt, white_noise, reconstructed)
                
                onestep_psnr_noise0 = psnr_burst0(invert_gt, white_noise, x_batch_burst)
                psnr_noise0.append(onestep_psnr_noise0.numpy())
                
                onestep_psnr_average = psnr_average_f(invert_gt, white_noise, x_batch_burst)
                psnr_average.append(onestep_psnr_average.numpy())
                for i in range(params["BURST_LENGTH"]):
                    psnr_perlayer[i].append(onestep_psnr_perlayer['da{}_noshow'.format(i)].numpy())
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #print(len(model.trainable_weights))
            #model.summury()
            loss_metric(loss)
            if step % 4==0:
                ckpt.iterate.assign_add(1)
            if step % 100 == 0:
                print('ckpt.iterate=%d with anneal_coef=%f'%(ckpt.iterate.numpy(),anneal_coef.numpy()))
                print('step %s: mean loss = %s' % (step, loss_metric.result()))
                print('step %s: psnr = %s' % (step, np.mean(psnr)))
                print('step %s: psnrnoshow0 = %s' % (step, np.mean(psnr_perlayer[0])))
                print('step %s: psnrburst0 = %s' % (step, np.mean(psnr_noise0)))
                print('step %s: psnraverage = %s' % (step, np.mean(psnr_average)))
        if int(ckpt.step) % 2 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        ckpt.step.assign_add(1)
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss_metric.result(), step=epoch)
            tf.summary.scalar('psnr_deblur', tf.convert_to_tensor(np.mean(psnr)), step=epoch)
            tf.summary.scalar('psnr_noise0', tf.convert_to_tensor(np.mean(psnr_noise0)), step=epoch)
            tf.summary.scalar('psnr_average', tf.convert_to_tensor(np.mean(psnr_average)), step=epoch)
            tf.summary.scalar('anneal_coef', anneal_coef, step=epoch)
            for i in range(params["BURST_LENGTH"]): 
                tf.summary.scalar('da{}_noshow'.format(i), tf.convert_to_tensor(np.mean(psnr_perlayer[i])), step=epoch)
        loss_metric.reset_states()
        
        
        
        val_psnr = []
        val_psnr_perlayer=[[0]*1 for i in range(params["BURST_LENGTH"])]
        val_psnr_noise0 = []
        val_psnr_average = []
        for step, x_batch_val in enumerate(val_image_ds):
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
        print('-----------------------------validation resule------------------------------')
        print('step %s: mean loss = %s' % (step, loss_metric.result()))
        print('step %s: val_psnr = %s' % (step, np.mean(val_psnr)))
        print('step %s: val_psnrnoshow0 = %s' % (step, np.mean(val_psnr_perlayer[0])))
        print('step %s: val_psnrburst0 = %s' % (step, np.mean(val_psnr_noise0)))
        print('step %s: val_psnraverage = %s' % (step, np.mean(val_psnr_average)))
        with summary_writer.as_default():
            tf.summary.scalar('val_loss', loss_metric.result(), step=epoch)
            tf.summary.scalar('val_psnr_deblur', tf.convert_to_tensor(np.mean(val_psnr)), step=epoch)
            tf.summary.scalar('val_psnr_noise0', tf.convert_to_tensor(np.mean(val_psnr_noise0)), step=epoch)
            tf.summary.scalar('val_psnr_average', tf.convert_to_tensor(np.mean(val_psnr_average)), step=epoch)
            for i in range(params["BURST_LENGTH"]): 
                tf.summary.scalar('val_da{}_noshow'.format(i), tf.convert_to_tensor(np.mean(val_psnr_perlayer[i])), step=epoch)
            
        loss_metric.reset_states()
        end = time.clock()
        print("epoch %d takes %f"%(epoch, end-start))

if __name__ == "__main__":
    main()
