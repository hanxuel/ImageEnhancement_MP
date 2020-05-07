# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:45:29 2020

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

def train_model(data_path,val_path, model_path, params):
    log_path = os.path.join(model_path, "logs")
    checkpoint_path = os.path.join(model_path, "checkpoint")

    mkdir_if_not_exists(log_path)
    mkdir_if_not_exists(checkpoint_path)

    epoch_npasses = params["epoch_npasses"]
    batch_size = params["batch_size"]
    width = params["width"]
    height = params["height"]
    burst_length = params["burst_length"]
    B = params["num_basis"]
    K = params["kernel_size"]
    anneal = params['anneal_rate']

    train_data_generator = \
        build_data_generator(data_path, params)

    val_params = dict(params)
    val_params["epoch_npasses"] = -1
    val_data_generator = \
        build_data_generator(val_path, val_params)
    memory,cpu=getMemCpu()
    time.sleep(0.2)
    print("memory used percent:",memory)
    print("CPU used:",cpu)
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    #sess = tf.Session(config=tf_config)
    with tf.Session(config=config) as sess:
        Input, Deblur = build_kpn_model(params)
        memory,cpu=getMemCpu()
        time.sleep(0.2)
        print("after build model, memory used percent:",memory)
        print("CPU used:",cpu)
        batch_i = tf.Variable(0, name="batch_i")
        for variable in tf.trainable_varaibles():
            print(variable.name + '-' +str(variable.get_shape()) + ' - ' + str(np.prod(variable.get_shape().as_list())))
        groundtruth_image = tf.placeholder(tf.float32, Deblur.shape, name="groundtruth_image")
        
        ###for loss operation
        annel_coeff = []
        if anneal > 0:
            for ii in range(burst_length):
                anneal_coeff.append(tf.pow(anneal, batch_i) * (10. ** (2)))
        losses = []
        basic_loss = basic_img_loss(groundtruth_image, Deblur)
        losses.append(basic_loss)
        for i in range(burst_length):
            if anneal is not None:
            a = anneal_coeff[i]
            print 'includes anneal'
            losses.append(basic_img_loss(groundtruth_image, Input[...,i]) * a)
        loss_op = tf.reduce_sum(tf.stack(losses)) 
        
        ###learning rate
        learning_rate_op=get_learning_rate(
                batch_i, params['batch_size'], params['initial_learning_rate'], params['decay_rate'], params['decay_step'])
        tf.summary.scalar('learning_rate', learning_rate_op)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_op)
        train_op = optimizer.minimize(loss_op,global_step=batch_i)
        
        train_loss_summary = \
            tf.placeholder(tf.float32, name="train_loss_summary")
        tf.summary.scalar("train_loss",train_loss_summary)
        
        val_loss_summary = \
            tf.placeholder(tf.float32, name="val_loss_summary")
        tf.summary.scalar("val_loss", val_loss_summary)
        summary_op = tf.summary.merge_all()
    
        model_saver = tf.train.Saver(save_relative_paths=True)
        train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2,
                                     save_relative_paths=True, pad_step_number=True)
        log_writer = tf.summary.FileWriter(log_path)
        log_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        model_saver.save(sess, os.path.join(checkpoint_path, "initial"),
                         write_meta_graph=True)
        
        for epoch in range(params["nepochs"]):
            train_loss_values = []
            batch = 0
            while True:
                train_data = next(train_data_generator) #for one epoche, if finish, yield none
                memory,cpu=getMemCpu()
                time.sleep(0.2)
                print("after load data, memory used percent:",memory)
                print("CPU used:",cpu)
                # Check if epoch finished.
                if train_data is None:
                    break

                burstimages_batch, groundtruthimage_batch = train_data

                num_batch_samples = burstimages_batch.shape[0]

                feed_dict = {}

                feed_dict[Input] = burstimages_batch
                feed_dict[groundtruth_image] = groundtruthimage_batch
                (_,
                 loss) = sess.run(
                    [train_op,
                     loss_op],
                     feed_dict=feed_dict
                )
                print('batch_i:{}'.format(sess.run(batch_i)))
                print('Learning rate: %f' % (sess.run(optimizer._lr_t)))

                train_loss_values.append(loss)
                print("Epoch: {}, "
                      "Batch: {}\n"
                      "  Loss:                  {}\n".format(
                      epoch + 1,
                      batch + 1,
                      loss))
                batch += 1
                memory,cpu=getMemCpu()
                time.sleep(0.2)
                print("after training, memory used percent:",memory)
                print("CPU used:",cpu)
            train_loss_value = np.nanmean(train_loss_values)
            val_loss_values = []
            
            for _ in range(val_nbatches):
                burstimages_batch, groundtruthimage_batch = next(val_data_generator)

                num_batch_samples = burstimages_batch.shape[0]

                feed_dict = {}

                feed_dict[Input] = burstimages_batch
                feed_dict[groundtruth_image] = groundtruthimage_batch
                
                loss = sess.run([loss_op],feed_dict=feed_dict)

                val_loss_values.append(loss)
            val_loss_value = np.nanmean(val_loss_values)

            print("Validation\n"
                  "  Loss:                  {}\n".format(
                  val_loss_value))

            summary = sess.run(
                summary_op,
                feed_dict={
                    train_loss_summary:
                        train_loss_value,
                    val_loss_summary:
                        val_loss_value
                }
            )

            log_writer.add_summary(summary, epoch)
            train_saver.save(sess, os.path.join(checkpoint_path, "checkpoint"),
                             global_step=epoch, write_meta_graph=False)

        model_saver.save(sess, os.path.join(checkpoint_path, "final"),
                         write_meta_graph=True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="/cluster/scratch/haliang/")
    parser.add_argument("--val_path", default="/cluster/scratch/haliang/")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--burst_length", type=int, default=8)
    parser.add_argument("--num_basis", type=int, default=90)
    parser.add_argument("--kernel_size", type=int, default=15)
    
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--val_nbatches", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--initial_learning_rate", type=float, default=0.0005)
    parser.add_argument("--decay_rate", type=float, default=0.99)
    parser.add_argument("--decay_step", type=int, default=90) 
    parser.add_argument("--loss_weight", type=float, default=2)
    parser.add_argument("--anneal_rate",type=float, default=0.998)
    return parser.parse_args()

def main():
    args = parse_args()

    np.random.seed(0)
    tf.set_random_seed(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    params = {
        "width":args.width,
        "height":args.height,
        "burst_length":args.burst_length,
        "num_basis":args.num_basis,
        "kernel_size":args.kernel_size,
        "nepochs": args.nepochs,
        "epoch_npasses": args.epoch_npasses,
        "val_nbatches": args.val_nbatches,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "initial_learning_rate":args.initial_learning_rate,
        "decay_rate":args.decay_rate,
        "decay_step":args.decay_step,
        "loss_weight":args.loss_weight
    }

    train_model(args.train_path, args.val_path, args.model_path, params)


if __name__ == "__main__":
    main()
