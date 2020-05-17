# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:29:32 2020

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
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default="G:\\master_thesis\Cengiz\BurstDenoising\SmallTraindata")
    parser.add_argument("--val_path", default="/cluster/scratch/haliang/")
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
    parser.add_argument("--nepochs", type=int, default=200)
    parser.add_argument("--val_nbatches", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--degamma", type=float, default=2.2)
    parser.add_argument("--to_shift", type=float, default=1.0)
    parser.add_argument("--decay_step", type=int, default=90) 
    parser.add_argument("--loss_weight", type=float, default=2)
    parser.add_argument("--anneal_rate",type=float, default=0.998)
    return parser.parse_args()
def main():
    args = parse_args()

    np.random.seed(0)
    tf.random.set_seed(0)

    #tf.logging.set_verbosity(tf.logging.INFO)
    params = {
        "root":args.root,
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
        "learning_rate":args.learning_rate
        }
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '.\\logs\\gradient_tape\\' + current_time + '\\train'
    #test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    #test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    #tf.summary.trace_on(graph=True, profiler=True)
    model=Simplemodel(params)    
    image_ds = DataLoader(params).get_ds()
# =============================================================================
#     ###try to visualize with keras api
#     for step, x_batch_train in enumerate(image_ds.take(1)):
#         x_batch_burst, x_batch_truth=x_batch_train
#         with tf.GradientTape() as tape:
#             reconstructed = model(x_batch_burst)
#     print(model.summary())
#     plot_model(model, to_file='model.png')
# =============================================================================
    epochs = params["nepochs"]
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    loss_metric = tf.keras.metrics.Mean()
    
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    
# =============================================================================
#     TensorBoardcallback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir,
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
        print('Start of epoch %d' % (epoch,))
        # Iterate over the batches of the dataset.
        psnr=[]
        for step, x_batch_train in enumerate(image_ds):
            x_batch_burst, x_batch_truth=x_batch_train
            with tf.GradientTape() as tape:
                reconstructed = model(x_batch_burst)
                #print("tf.shape(reconstructed)",tf.shape(reconstructed))
                # Compute reconstruction loss
                loss = deblur_layer_loss(x_batch_truth, reconstructed)
                loss += sum(model.losses)  # Add KLD regularization loss
                onestep_psnr =psnr_deblur(x_batch_truth, reconstructed)
                psnr.append(onestep_psnr.numpy())
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #print(len(model.trainable_weights))
            #model.summury()
            loss_metric(loss)
            if step % 10 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))
                print('step %s: psnr = %s' % (step, np.mean(psnr)))
        ckpt.step.assign_add(1)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_metric.result(), step=epoch)
            tf.summary.scalar('accuracy', tf.convert_to_tensor(np.mean(psnr)), step=epoch)
            
        loss_metric.reset_states()
        if int(ckpt.step) % 1 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

if __name__ == "__main__":
    main()
# =============================================================================
# 
# params = {
#         "root":'G:\\master_thesis\Cengiz\BurstDenoising\SmallValidationdata',
#         "batch_size":2,
#         "color":False,
#         "height":32,
#         "width":32,
#         "degamma":2.2,
#         "to_shift":1.,
#         "upscale":4,
#         "jitter":16,
#         "smalljitter":2,
#         "BURST_LENGTH":2,
#         "Kernel_size":15,
#         "Basis_num":10
#         }
# model=Simplemodel(params)
# # =============================================================================
# # print("to show the model")
# # tf.keras.utils.plot_model(model,to_file='model.png', show_shapes=True, dpi=64)
# # =============================================================================
# image_ds = DataLoader(params).get_ds()
# epochs = 50
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# loss_metric = tf.keras.metrics.Mean()
# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
# manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=1)
# 
# ckpt.restore(manager.latest_checkpoint)
# if manager.latest_checkpoint:
#     print("Restored from {}".format(manager.latest_checkpoint))
# else:
#     print("Initializing from scratch.")
# # Iterate over epochs.
# for epoch in range(epochs):
#     print('Start of epoch %d' % (epoch,))
#     # Iterate over the batches of the dataset.
#     for step, x_batch_train in enumerate(image_ds):
#         x_batch_burst, x_batch_truth=x_batch_train
#         with tf.GradientTape() as tape:
#             reconstructed = model(x_batch_burst)
#             #print("tf.shape(reconstructed)",tf.shape(reconstructed))
#             # Compute reconstruction loss
#             loss = custom_loss(x_batch_truth, reconstructed)
#             loss += sum(model.losses)  # Add KLD regularization loss
#             psnr = psnr_deblur(x_batch_truth, reconstructed)
#         grads = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         #print(len(model.trainable_weights))
#         #model.summury()
#         loss_metric(loss)
#         if step % 3 == 0:
#             print('step %s: mean loss = %s' % (step, loss_metric.result()))
#             print('step %s: psnr = %s' % (step, psnr.numpy()))
#     ckpt.step.assign_add(1)
#     if int(ckpt.step) % 1 == 0:
#         save_path = manager.save()
#         print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
# 
# =============================================================================
