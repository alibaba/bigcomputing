#coding:utf-8
import numpy as np
from data_iterator import DataIterator
import tensorflow as tf
from model_taobao_allfea import *
import time
import random
import sys
import json
from utils import *
import multiprocessing
from multiprocessing import Process, Value, Array
from wrap_time import time_it
from data_loader import DataLoader
import threading
from collections import deque

import os
if "BRIDGE_ENABLE_TAO" in os.environ and os.environ["BRIDGE_ENABLE_TAO"] == "true":
    tf.load_op_library("/home/admin/.tao_compiler/libtao_ops.so")

def file_num(x):
    if x < 10:
        return '0' + str(x)
    else:
        return str(x)


EMBEDDING_DIM = 4
HIDDEN_SIZE = EMBEDDING_DIM * 6 
MEMORY_SIZE = 4
 

def eval(sess, test_file, model, model_path, batch_size, maxlen,  best_auc = [1.0]):
    print("Testing starts------------")
    data_load_test= DataLoader(test_file, 'test_sample', batch_size, 4 )
    producer1 = threading.Thread(target=data_load_test.data_read, args=(0,2))
    producer2 = threading.Thread(target=data_load_test.data_read, args=(1,2))
    producer1.start()
    producer2.start()
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    iterations = 0
    stored_arr = []
 
    for data in data_load_test.next():
        iterations +=1
        user_id, item_id, cate_id,shop_id, node_id, product_id, brand_id, \
          label, hist_item, hist_cate, hist_shop, hist_node, hist_product, \
          hist_brand, hist_mask, neg_hist_item, neg_hist_cate, \
          neg_hist_shop, neg_hist_node, neg_hist_product, neg_hist_brand = data 
        target = label
        prob, loss, acc, aux_loss = model.calculate(sess, [user_id, item_id, \
            cate_id, shop_id, node_id, product_id, brand_id, \
            hist_item, hist_cate, hist_shop, hist_node, hist_product, hist_brand, \
        neg_hist_item, neg_hist_cate, neg_hist_shop, neg_hist_node, neg_hist_product, neg_hist_brand, hist_mask, label])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        # user_l = user_id.tolist()
        for p ,t in zip(prob_1, target_1):
    	    stored_arr.append([p, t])
    
    #test_auc = calc_gauc(stored_arr, user_l)
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / iterations
    loss_sum = loss_sum / iterations
    aux_loss_sum = aux_loss_sum / iterations
    if best_auc[0] < test_auc:
        best_auc[0] = test_auc
        model.save(sess, model_path)
    producer1.join()
    producer2.join()
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum, best_auc[0]

def train(
        train_file,
        test_file,
        batch_size = 256,
        maxlen = 100,
        test_iter = 500,
        save_iter = 5000,
        model_type = 'DNN',
        Memory_Size = 4,
):
    TEM_MEMORY_SIZE = Memory_Size
    model_path = "dnn_save_path/taobao_ckpt_noshuff" + model_type
    best_model_path = "dnn_best_model/taobao_ckpt_noshuff" + model_type
    gpu_options = tf.GPUOptions(allow_growth=True) 
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        # Obtained in the data preprocess stage. To save time, json files are not needed.
        uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n = [7956430, 34196611, 5596, 4377722, 2975349, 65624, 584181] 
        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen

        if model_type == 'DNN': 
            model = Model_DNN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'PNN': 
            model = Model_PNN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'GRU4REC': 
            model = Model_GRU4REC(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIN': 
            model = Model_DIN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'ARNN': 
            model = Model_ARNN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIEN': 
            model = Model_DIEN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIEN_with_neg': 
            model = Model_DIEN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=True)
        else:
            print ("Invalid model_type : %s", model_type)
            return
 
        #参数初始化
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sys.stdout.flush()

        start_time = time.time()
        last_time = start_time
        iter = 0
        lr = 0.001
        best_auc= [0.0]
        loss_sum = 0.0
        accuracy_sum = 0.
        left_loss_sum = 0.
        aux_loss_sum = 0.
        mem_loss_sum = 0.
        # set 1 epoch only
        epoch = 1
        for itr in range(epoch):
            print("epoch"+str(itr))
            #load data 
            data_load= DataLoader(train_file, 'train_sample', 256, 15)
            producer1 = threading.Thread(target=data_load.data_read, args=(0,2))
            producer2 = threading.Thread(target=data_load.data_read, args=(1,2))
            producer1.start()
            producer2.start()
            #for iteration in range(number_samples):
            for  data in data_load.next():
                user_id, item_id, cate_id,shop_id, node_id, product_id, brand_id, \
                    label, hist_item, hist_cate, hist_shop, hist_node, hist_product, \
                    hist_brand, hist_mask, neg_hist_item, neg_hist_cate, neg_hist_shop, \
                    neg_hist_node, neg_hist_product, neg_hist_brand = data 
                # preprocess the -1 index in batch data
                item_id[item_id == -1] = item_n
                shop_id[shop_id == -1] = shop_n
                node_id[node_id == -1] = node_n
                product_id[product_id == -1] = product_n
                brand_id[brand_id == -1] = brand_n

                loss, acc, aux_loss, mem_loss, left_loss = model.train(sess, [user_id, item_id,\
                    cate_id, shop_id, node_id, product_id, brand_id, \
                    hist_item, hist_cate, hist_shop, hist_node, hist_product, hist_brand, \
                    neg_hist_item, neg_hist_cate, neg_hist_shop, neg_hist_node, neg_hist_product, \
                    neg_hist_brand, hist_mask, label, lr])
                    # cate_id, hist_item, hist_cate, neg_hist_item, neg_hist_cate, hist_mask, label, lr])
                loss_sum += loss
                accuracy_sum += acc
                left_loss_sum += left_loss
                aux_loss_sum += aux_loss
                mem_loss_sum += mem_loss
                iter += 1
                sys.stdout.flush()
                
                if (iter % test_iter) == 0:
                    test_time = time.time()
                    print('[Iteration]=%d, train_loss=%.4f, train_accuracy=%.4f, train_aux_loss=%.4f, train_left_loss=%.4f, throughput=%.4f, total time=%.4f' %(iter, loss_sum / test_iter, accuracy_sum / test_iter, \
                      aux_loss_sum /test_iter ,  left_loss_sum / test_iter, batch_size*test_iter/(test_time-last_time),\
                      test_time-start_time))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    left_loss_sum = 0.0
                    aux_loss_sum = 0.
                    mem_loss_sum = 0.
                    if (iter % save_iter) == 0:
                        print('save model iter: %d' %(iter))
                        model.save(sess, model_path+"--"+str(iter))
                        print('Testing finishes-------test_auc=%.4f, test_loss=%.4f, test_accuracy=%.4f, test_aux_loss=%.4f, best_auc=%.4f ' % eval(sess, test_file, model, best_model_path, batch_size, maxlen, best_auc))
                    last_time = test_time

            producer1.join()
            producer2.join()

def test(
        train_file,
        test_file,
        batch_size = 256,
        maxlen = 100,
        test_iter = 100,
        save_iter = 400,
        model_type = 'DNN',
        Memory_Size = 4,
):
    
    TEM_MEMORY_SIZE = Memory_Size
    Ntm_Flag = "base"
    if Ntm_Flag == "base":
        model_path = "dnn_save_path/taobao_ckpt_noshuff" + model_type
    else:
        model_path = "dnn_save_path/taobao_ckpt_noshuff" + model_type+str(Memory_Size)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n = [7956430, 34196611, 5596, 4377722, 2975349, 65624, 584181] 
        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen

        if model_type == 'DNN': 
            model = Model_DNN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'PNN': 
            model = Model_PNN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'GRU4REC': 
            model = Model_GRU4REC(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIN': 
            model = Model_DIN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'ARNN': 
            model = Model_ARNN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIEN': 
            model = Model_DIEN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIEN_with_neg': 
            model = Model_DIEN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=True)
        else:
            print ("Invalid model_type : %s", model_type)
            return

        model.restore(sess, model_path+'--50000')
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- best_auc=%.4f ' % eval(sess, test_file, model, model_path, batch_size, maxlen))


if __name__ == '__main__':
    SEED = int(sys.argv[3])
    if len(sys.argv) > 5:
        Memory_Size = int(sys.argv[4])      
    else:
        Memory_Size = 4
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    train_file = './process_data_maxlen100_0225/'
    test_file = train_file
    if sys.argv[1] == 'train':
        train(train_file=train_file, test_file=test_file, model_type=sys.argv[2], Memory_Size=Memory_Size)
    elif sys.argv[1] == 'test':
        test(train_file=train_file, test_file=test_file, model_type=sys.argv[2], Memory_Size=Memory_Size)
    else:
        print('do nothing...')
