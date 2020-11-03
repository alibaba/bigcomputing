# Copyright (c) Alibaba, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: preprocess input data for DIEN benchmark
#
# Utility function(s) to download and pre-process public data sets
#   - DIEN public data
#      ..........
#

 
import sys
# import os
from os import path
# import io
# from io import StringIO
# import collections as coll
import random
import numpy as np
import json
import time

class preprocessDataset:

    def __init__(
        self,
        data_path,
        maxlen,
        ):
        self.data_path = data_path
        #self.user_dict, self.item_dict, self.cate_dict, self.shop_dict, self.node_dict \
        #, self.product_dict, self.brand_dict, self.item_info = dict_list
        self.neg_sample = 'LastInstance' # 'Random' or 'LastInstance'
    
        
        self.source_buffer = []
        self.neg_hist_catch = {}
        self.maxlen = maxlen
        self.max_catch_num = 20
	#self.end_of_data = False
        st = time.time()       
        print("Start loading dict...") 
        #hack here
        data_path = '/disk3/w.wei/dien-new/'
        self.item_info = json.load(open(data_path + '/item_info.json', 'r'))
        print ('Finish loading item_info.json,length=',len(self.item_info.keys()))
        self.user_dict = json.load(open(data_path + '/user_voc.json', 'r'))
        print ('Finish loading user_voc.json,length=',len(self.user_dict.keys()))
        self.item_dict = json.load(open(data_path + '/item_voc.json', 'r'))
        print ('Finish loading item_voc.json,length=',len(self.item_dict.keys()))
        self.cate_dict = json.load(open(data_path + '/cate_voc.json', 'r'))
        print ('Finish loading cate_voc.json,length=',len(self.cate_dict.keys()))
        self.shop_dict = json.load(open(data_path + '/shop_voc.json', 'r'))
        print ('Finish loading shop_voc.json,length=',len(self.shop_dict.keys()))
        self.node_dict = json.load(open(data_path + '/node_voc.json', 'r'))
        print ('Finish loading node_voc.json,length=',len(self.node_dict.keys()))
        self.product_dict = json.load(open(data_path + '/product_voc.json', 'r'))
        print ('Finish loading product_voc.json,length=',len(self.product_dict.keys()))
        self.brand_dict = json.load(open(data_path + '/brand_voc.json', 'r'))
        print ('Finish loading brand_voc.json,length=',len(self.brand_dict.keys()))
        print("Time for load dict=", time.time()-st)
        #import pdb; pdb.set_trace()
        self.all_items = self.item_info.keys()
        self.num_items = len(self.all_items)
        #generate random neg item to initialize last_item information
        item_idx = int(random.random()*self.num_items)
        neg_item = self.all_items[item_idx]
        self.last_cate = self.map_cate(self.item_info[neg_item][0])
        self.last_shop = self.map_shop(self.item_info[neg_item][1])
        self.last_node = self.map_node(self.item_info[neg_item][2])
        self.last_product = self.map_product(self.item_info[neg_item][3])
        self.last_brand = self.map_brand(self.item_info[neg_item][4])
        self.last_item = self.map_item(self.all_items[item_idx])#map origin item to item_id
    
    def get_id_nums(self):
        uid_n = len(self.user_dict.keys())
        item_n = len(self.item_dict.keys())
        cate_n = len(self.cate_dict.keys())
        shop_n = len(self.shop_dict.keys())
        node_n = len(self.node_dict.keys())
        product_n = len(self.product_dict.keys())
        brand_n = len(self.brand_dict.keys())
        return uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n
     
    def map_item(self, x):
        return int(self.item_dict.get(x, -1))

    def map_user(self, x):
        return int(self.user_dict.get(x, -1))

    def map_cate(self, x):
        return int(self.cate_dict.get(x, -1))

    def map_shop(self, x):
        return int(self.shop_dict.get(x, -1))

    def map_node(self, x):
        return int(self.node_dict.get(x, -1))

    def map_product(self, x):
        return int(self.product_dict.get(x, -1))

    def map_brand(self, x):
        return int(self.brand_dict.get(x, -1))
     
    def gen_neg_hist(self, length):
        if len(self.neg_hist_catch.get(length, [1])) == self.max_catch_num:
            index = int(random.random()*self.max_catch_num)
            return self.neg_hist_catch[length][index]
        else:
            #generate a new neg hist
            neg_item_hist = []
            neg_cate_hist = []
            neg_shop_hist = []
            neg_node_hist = []
            neg_product_hist = []
            neg_brand_hist = []
            for i in range(length):
                item_idx = int(random.random()*self.num_items)
                neg_item = self.all_items[item_idx]
                neg_cate = self.map_cate(self.item_info[neg_item][0])
                neg_cate_hist.append(neg_cate)
                neg_shop = self.map_shop(self.item_info[neg_item][1])
                neg_shop_hist.append(neg_shop)
                neg_node = self.map_node(self.item_info[neg_item][2])
                neg_node_hist.append(neg_node)
                neg_product = self.map_product(self.item_info[neg_item][3])
                neg_product_hist.append(neg_product)
                neg_brand = self.map_brand(self.item_info[neg_item][4])
                neg_brand_hist.append(neg_brand)
                neg_item = self.map_item(self.all_items[item_idx])#map origin item to item_id
                neg_item_hist.append(neg_item)
            self.neg_hist_catch[length] = self.neg_hist_catch.get(length, []) # do not understand TODO
            self.neg_hist_catch[length].append([neg_item_hist, neg_cate_hist, neg_shop_hist,\
		neg_node_hist, neg_product_hist, neg_brand_hist])
            return [neg_item_hist, neg_cate_hist, neg_shop_hist, neg_node_hist, neg_product_hist, neg_brand_hist]
    
    def gen_neg_hist3(self, length):
        if len(self.neg_hist_catch.get(length, [1])) == self.max_catch_num:
            index = int(random.random()*self.max_catch_num)
            return self.neg_hist_catch[length][index]
        else:
            #generate a new neg hist
            neg_item_hist = []
            neg_cate_hist = []
            neg_shop_hist = []
            neg_node_hist = []
            neg_product_hist = []
            neg_brand_hist = []
            for i in range(length):
                item_idx = int(random.random()*self.num_items)
                neg_item = self.all_items[item_idx]
               
                tmp = self.item_info[neg_item]
                neg_cate_hist.append(self.map_cate(tmp[0]))
                neg_shop_hist.append(self.map_shop(tmp[1]))
                neg_node_hist.append(self.map_node(tmp[2]))
                neg_product_hist.append(self.map_product(tmp[3]))
                neg_brand_hist.append(self.map_brand(tmp[4]))
                neg_item_hist.append(self.map_item(neg_item))#map origin item to item_id
            self.neg_hist_catch[length] = self.neg_hist_catch.get(length, []) # do not understand TODO
            self.neg_hist_catch[length].append([neg_item_hist, neg_cate_hist, neg_shop_hist,\
		neg_node_hist, neg_product_hist, neg_brand_hist])
            return [neg_item_hist, neg_cate_hist, neg_shop_hist, neg_node_hist, neg_product_hist, neg_brand_hist]
    
    def gen_neg_hist2(self, length):
        if len(self.neg_hist_catch.get(length, [1])) == self.max_catch_num:
            index = int(random.random()*self.max_catch_num)
            return self.neg_hist_catch[length][index]
        else:
            #generate a new neg hist
            neg_item_hist = []
            neg_cate_hist = []
            neg_shop_hist = []
            neg_node_hist = []
            neg_product_hist = []
            neg_brand_hist = []
            item_idx = [int(random.random()*self.num_items) for i in range(length)]
            neg_item = self.all_items[item_idx]
            neg_info = self.item_info[neg_item]
            neg_cate = self.map_cate(neg_info[0])
            neg_cate_hist.append(neg_cate)
            neg_shop = self.map_shop(neg_info[1])
            neg_shop_hist.append(neg_shop)
            neg_node = self.map_node(neg_info[2])
            neg_node_hist.append(neg_node)
            neg_product = self.map_product(neg_info[3])
            neg_product_hist.append(neg_product)
            neg_brand = self.map_brand(neg_info[4])
            neg_brand_hist.append(neg_brand)
            neg_item_id = self.map_item(neg_item)#map origin item to item_id
            neg_item_hist.append(neg_item_id)
            self.neg_hist_catch[length] = self.neg_hist_catch.get(length, []) # do not understand TODO
            self.neg_hist_catch[length].append([neg_item_hist, neg_cate_hist, neg_shop_hist,\
		neg_node_hist, neg_product_hist, neg_brand_hist])
            return [neg_item_hist, neg_cate_hist, neg_shop_hist, neg_node_hist, neg_product_hist, neg_brand_hist]
    
    def fill_ndarray(self, hist):
        nd_his = np.ones(self.maxlen) * -1
        nd_his[:len(hist)] = hist
        return nd_his

    # 2nd method by using previous user's behavior to replace the current user's negative behavior data
    def process2(self, data_files, file_name):
        #try:
        #    ss = self.source_buffer.pop()
        #except IndexError:
            

        source = []
        target = []
        hist_item_list = []
        hist_cate_list = []
        hist_shop_list = []
        hist_node_list = []
        hist_product_list = []
        hist_brand_list = []
     
        neg_hist_item_list = []
        neg_hist_cate_list = []
        neg_hist_shop_list = []
        neg_hist_node_list = []
        neg_hist_product_list = []
        neg_hist_brand_list = []

        last_neg_hist = self.gen_neg_hist3(self.maxlen)
        
        st = time.time()
        #data_size = 20000
        #data_size = 500000 #around 9GB data for maxlength=100
        data_size = 50000 #around 9GB data for maxlength=1000
        line_count = 0
        file_count = 0
        for file_read in data_files:
            data_file = self.data_path + file_read
            print('Open file ', data_file)
            f = open(data_file, 'r')
            tmp_s = f.readline()
            while(tmp_s):
                line_count = line_count + 1
                
                ss = tmp_s.strip().split("^H")
                uid = self.map_user(ss[0])
                hist_item = map(self.map_item, ss[1].split('\003'))
                hist_cate = map(self.map_cate, ss[2].split('\003'))
                hist_shop = map(self.map_shop, ss[3].split('\003'))
                hist_node = map(self.map_node, ss[4].split('\003'))
                hist_product = map(self.map_product, ss[5].split('\003'))
                hist_brand = map(self.map_brand, ss[6].split('\003'))
                       
                pos_item = hist_item[-1] 
                pos_cate = hist_cate[-1] 
                pos_shop = hist_shop[-1] 
                pos_node = hist_node[-1] 
                pos_product = hist_product[-1] 
                pos_brand = hist_brand[-1]  
                if self.neg_sample == 'LastInstance':
                    #set item of last instance as neg sample
                    neg_item, neg_cate, neg_shop, neg_node, neg_product, neg_brand \
                    = self.last_item, self.last_cate, self.last_shop, self.last_node, \
                        self.last_product, self.last_brand
                    # neg is the last user's behavior which is not random
                    random_neg_hist = last_neg_hist
                elif self.neg_sample == 'Random': 
                    #generate random neg_item information for neg sample
                    item_idx = int(random.random()*self.num_items)
                    while self.map_item(self.all_items[item_idx]) in hist_item:
                        item_idx = int(random.random()*self.num_items)
                    neg_item = self.all_items[item_idx]
                    neg_cate = self.map_cate(self.item_info[neg_item][0])
                    neg_shop = self.map_shop(self.item_info[neg_item][1])
                    neg_node = self.map_node(self.item_info[neg_item][2])
                    neg_product = self.map_product(self.item_info[neg_item][3])
                    neg_brand = self.map_brand(self.item_info[neg_item][4])
                    neg_item = self.map_item(self.all_items[item_idx])#map origin item to item_id     
                #gen neg hist
                #lengthx = len(hist_item[-self.maxlen:])
                #random_neg_hist = self.gen_neg_hist3(lengthx)
                

                #add positive sample
                source.append([uid, pos_item, pos_cate, pos_shop, pos_node, pos_product, pos_brand])
                target.append([1, 0])
                hist_item_list.append(hist_item[-(self.maxlen+1):-1])
                hist_cate_list.append(hist_cate[-(self.maxlen+1):-1])
                hist_shop_list.append(hist_shop[-(self.maxlen+1):-1])
                hist_node_list.append(hist_node[-(self.maxlen+1):-1])
                hist_product_list.append(hist_product[-(self.maxlen+1):-1])
                hist_brand_list.append(hist_brand[-(self.maxlen+1):-1])
                
                neg_hist_item_list.append(random_neg_hist[0])
                neg_hist_cate_list.append(random_neg_hist[1])
                neg_hist_shop_list.append(random_neg_hist[2])
                neg_hist_node_list.append(random_neg_hist[3])
                neg_hist_product_list.append(random_neg_hist[4])
                neg_hist_brand_list.append(random_neg_hist[5])       
    	    
                #add negative sample, histogram are the same!
                source.append([uid, neg_item, neg_cate, neg_shop, neg_node, neg_product, neg_brand])
                target.append([0, 1])
                hist_item_list.append(hist_item[-(self.maxlen+1):-1])
                hist_cate_list.append(hist_cate[-(self.maxlen+1):-1])
                hist_shop_list.append(hist_shop[-(self.maxlen+1):-1])
                hist_node_list.append(hist_node[-(self.maxlen+1):-1])
                hist_product_list.append(hist_product[-(self.maxlen+1):-1])
                hist_brand_list.append(hist_brand[-(self.maxlen+1):-1])
                
                neg_hist_item_list.append(random_neg_hist[0])
                neg_hist_cate_list.append(random_neg_hist[1])
                neg_hist_shop_list.append(random_neg_hist[2])
                neg_hist_node_list.append(random_neg_hist[3])
                neg_hist_product_list.append(random_neg_hist[4])
                neg_hist_brand_list.append(random_neg_hist[5])       
                if self.neg_sample == 'LastInstance':
                    self.last_item, self.last_cate, self.last_shop, self.last_node, self.last_product, self.last_brand = pos_item, pos_cate, pos_shop, pos_node, pos_product, pos_brand 
                    last_neg_hist= [hist_item[-(self.maxlen+1):-1],
                                    hist_cate[-(self.maxlen+1):-1], 
                                    hist_shop[-(self.maxlen+1):-1], 
                                    hist_node[-(self.maxlen+1):-1], 
                                    hist_product[-(self.maxlen+1):-1], 
                                    hist_brand[-(self.maxlen+1):-1] ]
                
                # read in next line 
    	        tmp_s = f.readline()
               
                if(line_count %  10000 == 0):
                    print("Total processed lines = ", line_count,",spent time = ",time.time() -st)
                    #print("source =",source_array)
                    #print("neg history iteam =",neg_history_item_array)
                    #break
                if(line_count == data_size ):     
                    print("Start to save: total processed lines = ", line_count,",spent time = ",time.time() -st)
                    line_count = 0
                    source_array = np.array(source)
                    target_array = np.array(target)
                    history_item_array = np.array(hist_item_list)        
                    history_cate_array = np.array(hist_cate_list)
                    history_shop_array = np.array(hist_shop_list)
                    history_node_array = np.array(hist_node_list)
                    history_product_array = np.array(hist_product_list)
                    history_brand_array = np.array(hist_brand_list)
                    # 
                    print("test1", len(hist_item_list))
                    neg_history_item_array = np.array(neg_hist_item_list)        
                    neg_history_cate_array = np.array(neg_hist_cate_list)
                    neg_history_shop_array = np.array(neg_hist_shop_list)
                    neg_history_node_array = np.array(neg_hist_node_list)
                    neg_history_product_array = np.array(neg_hist_product_list)
                    neg_history_brand_array = np.array(neg_hist_brand_list)
                    print("test2", len(neg_hist_item_list))
                    print("Finished array transform")
                    np.savez('./' + file_name + '_' + str(file_count) + '_processed.npz', source_array=source_array,target_array=target_array,
    	                 history_item_array=history_item_array,history_cate_array=history_cate_array, 
                         history_shop_array=history_shop_array,history_node_array=history_node_array,
                         history_product_array=history_product_array,history_brand_array=history_brand_array,
                         neg_history_item_array=neg_history_item_array,neg_history_cate_array=neg_history_cate_array,
                         neg_history_shop_array=neg_history_shop_array,neg_history_node_array=neg_history_node_array,
                         neg_history_product_array=neg_history_product_array,neg_history_brand_array=neg_history_brand_array)
                    file_count = file_count + 1
                    #===== empty the var
                    source[:] = []
                    target[:] = []
                    hist_item_list[:] = []
                    hist_cate_list[:] = []
                    hist_shop_list[:] = []
                    hist_node_list[:] = []
                    hist_product_list[:] = []
                    hist_brand_list[:] = []
                 
                    neg_hist_item_list[:] = []
                    neg_hist_cate_list[:] = []
                    neg_hist_shop_list[:] = []
                    neg_hist_node_list[:] = []
                    neg_hist_product_list[:] = []
                    neg_hist_brand_list[:] = []
           # for residual lines
     
            if(line_count != 0 ):    
                print("Residue lines:total processed lines = ", line_count,",spent time = ",time.time() -st)
                line_count = 0
                source_array = np.array(source)
                target_array = np.array(target)
                history_item_array = np.array(hist_item_list)        
                history_cate_array = np.array(hist_cate_list)
                history_shop_array = np.array(hist_shop_list)
                history_node_array = np.array(hist_node_list)
                history_product_array = np.array(hist_product_list)
                history_brand_array = np.array(hist_brand_list)
                neg_history_item_array = np.array(neg_hist_item_list)        
                neg_history_cate_array = np.array(neg_hist_cate_list)
                neg_history_shop_array = np.array(neg_hist_shop_list)
                neg_history_node_array = np.array(neg_hist_node_list)
                neg_history_product_array = np.array(neg_hist_product_list)
                neg_history_brand_array = np.array(neg_hist_brand_list)
                print("Finished array transform")
                np.savez('./' + file_name + '_' + str(file_count) + '_processed.npz', source_array=source_array,target_array=target_array,
    	                 history_item_array=history_item_array,history_cate_array=history_cate_array, 
                         history_shop_array=history_shop_array,history_node_array=history_node_array,
                         history_product_array=history_product_array,history_brand_array=history_brand_array,
                         neg_history_item_array=neg_history_item_array,neg_history_cate_array=neg_history_cate_array,
                         neg_history_shop_array=neg_history_shop_array,neg_history_node_array=neg_history_node_array,
                         neg_history_product_array=neg_history_product_array,neg_history_brand_array=neg_history_brand_array)
                file_count = file_count + 1
                #===== empty the var
                source[:] = []
                target[:] = []
                hist_item_list[:] = []
                hist_cate_list[:] = []
                hist_shop_list[:] = []
                hist_node_list[:] = []
                hist_product_list[:] = []
                hist_brand_list[:] = []
                 
                neg_hist_item_list[:] = []
                neg_hist_cate_list[:] = []
                neg_hist_shop_list[:] = []
                neg_hist_node_list[:] = []
                neg_hist_product_list[:] = []
                neg_hist_brand_list[:] = []

    # Process the data by read in the user information from raw_path file
    def process(self, data_files, file_name):
        #try:
        #    ss = self.source_buffer.pop()
        #except IndexError:
            

        source = []
        target = []
        hist_item_list = []
        hist_cate_list = []
        hist_shop_list = []
        hist_node_list = []
        hist_product_list = []
        hist_brand_list = []
     
        neg_hist_item_list = []
        neg_hist_cate_list = []
        neg_hist_shop_list = []
        neg_hist_node_list = []
        neg_hist_product_list = []
        neg_hist_brand_list = []

        st = time.time()
        data_size = 500000 #around 9GB data for maxlength=100
        #data_size = 250000 #around 9GB data for maxlength=1500
        line_count = 0
        file_count = 0
        for file_read in data_files:
            data_file = self.data_path + file_read
            print('Open file ', data_file)
            f = open(data_file, 'r')
            tmp_s = f.readline()
            while(tmp_s):
                line_count = line_count + 1
                
                ss = tmp_s.strip().split("^H")
                uid = self.map_user(ss[0])
                hist_item = map(self.map_item, ss[1].split('\003'))
                hist_cate = map(self.map_cate, ss[2].split('\003'))
                hist_shop = map(self.map_shop, ss[3].split('\003'))
                hist_node = map(self.map_node, ss[4].split('\003'))
                hist_product = map(self.map_product, ss[5].split('\003'))
                hist_brand = map(self.map_brand, ss[6].split('\003'))
                       
                pos_item = hist_item[-1] 
                pos_cate = hist_cate[-1] 
                pos_shop = hist_shop[-1] 
                pos_node = hist_node[-1] 
                pos_product = hist_product[-1] 
                pos_brand = hist_brand[-1]  
                if self.neg_sample == 'LastInstance':
                    #set item of last instance as neg sample
                    neg_item, neg_cate, neg_shop, neg_node, neg_product, neg_brand \
                    = self.last_item, self.last_cate, self.last_shop, self.last_node, \
                        self.last_product, self.last_brand
                elif self.neg_sample == 'Random': 
                    #generate random neg_item information for neg sample
                    item_idx = int(random.random()*self.num_items)
                    while self.map_item(self.all_items[item_idx]) in hist_item:
                        item_idx = int(random.random()*self.num_items)
                    neg_item = self.all_items[item_idx]
                    neg_cate = self.map_cate(self.item_info[neg_item][0])
                    neg_shop = self.map_shop(self.item_info[neg_item][1])
                    neg_node = self.map_node(self.item_info[neg_item][2])
                    neg_product = self.map_product(self.item_info[neg_item][3])
                    neg_brand = self.map_brand(self.item_info[neg_item][4])
                    neg_item = self.map_item(self.all_items[item_idx])#map origin item to item_id     
                 #gen neg hist
                lengthx = len(hist_item[-self.maxlen:])
                random_neg_hist = self.gen_neg_hist3(lengthx)
        
                #add positive sample
                source.append([uid, pos_item, pos_cate, pos_shop, pos_node, pos_product, pos_brand])
                target.append([1, 0])
                hist_item_list.append(hist_item[-(self.maxlen+1):-1])
                hist_cate_list.append(hist_cate[-(self.maxlen+1):-1])
                hist_shop_list.append(hist_shop[-(self.maxlen+1):-1])
                hist_node_list.append(hist_node[-(self.maxlen+1):-1])
                hist_product_list.append(hist_product[-(self.maxlen+1):-1])
                hist_brand_list.append(hist_brand[-(self.maxlen+1):-1])
                
                neg_hist_item_list.append(random_neg_hist[0])
                neg_hist_cate_list.append(random_neg_hist[1])
                neg_hist_shop_list.append(random_neg_hist[2])
                neg_hist_node_list.append(random_neg_hist[3])
                neg_hist_product_list.append(random_neg_hist[4])
                neg_hist_brand_list.append(random_neg_hist[5])       
    	    
                #add negative sample, histogram are the same!
                source.append([uid, neg_item, neg_cate, neg_shop, neg_node, neg_product, neg_brand])
                target.append([0, 1])
                hist_item_list.append(hist_item[-(self.maxlen+1):-1])
                hist_cate_list.append(hist_cate[-(self.maxlen+1):-1])
                hist_shop_list.append(hist_shop[-(self.maxlen+1):-1])
                hist_node_list.append(hist_node[-(self.maxlen+1):-1])
                hist_product_list.append(hist_product[-(self.maxlen+1):-1])
                hist_brand_list.append(hist_brand[-(self.maxlen+1):-1])
                
                neg_hist_item_list.append(random_neg_hist[0])
                neg_hist_cate_list.append(random_neg_hist[1])
                neg_hist_shop_list.append(random_neg_hist[2])
                neg_hist_node_list.append(random_neg_hist[3])
                neg_hist_product_list.append(random_neg_hist[4])
                neg_hist_brand_list.append(random_neg_hist[5])       
                if self.neg_sample == 'LastInstance':
                    self.last_item, self.last_cate, self.last_shop, self.last_node, self.last_product, self.last_brand = pos_item, pos_cate, pos_shop, pos_node, pos_product, pos_brand 
                # seperate the source and target into array
                #uid_array = np.array(source)[:,0]
                #item_array = np.arrayoelf(source)[:,1]
                #cate_array = np.array(source)[:,2]
                #shop_array = np.array(source)[:,3]
                #node_array = np.array(source)[:,4]
                #product_array = np.array(source)[:,5]
                #brand_array = np.array(source)[:,6]
        	    
                
                # read in next line 
    	        tmp_s = f.readline()
               
                if(line_count %  10000 == 0):
                    print("Total processed lines = ", line_count,",spent time = ",time.time() -st)
                if(line_count == data_size):     
                    line_count = 0
                    #print("Total processed lines = ", line_count,",spent time = ",time.time() -st)
                    source_array = np.array(source)
                    target_array = np.array(target)
                    history_item_array = np.array(hist_item_list)        
                    history_cate_array = np.array(hist_cate_list)
                    history_shop_array = np.array(hist_shop_list)
                    history_node_array = np.array(hist_node_list)
                    history_product_array = np.array(hist_product_list)
                    history_brand_array = np.array(hist_brand_list)
                    neg_history_item_array = np.array(neg_hist_item_list)        
                    neg_history_cate_array = np.array(neg_hist_cate_list)
                    neg_history_shop_array = np.array(neg_hist_shop_list)
                    neg_history_node_array = np.array(neg_hist_node_list)
                    neg_history_product_array = np.array(neg_hist_product_list)
                    neg_history_brand_array = np.array(neg_hist_brand_list)
                    print("Finished array transform")
                    np.savez('./' + file_name + '_' + str(file_count) + '_processed.npz', source_array=source_array,target_array=target_array,
    	                 history_item_array=history_item_array,history_cate_array=history_cate_array, 
                         history_shop_array=history_shop_array,history_node_array=history_node_array,
                         history_product_array=history_product_array,history_brand_array=history_brand_array,
                         neg_history_item_array=neg_history_item_array,neg_history_cate_array=neg_history_cate_array,
                         neg_history_shop_array=neg_history_shop_array,neg_history_node_array=neg_history_node_array,
                         neg_history_product_array=neg_history_product_array,neg_history_brand_array=neg_history_brand_array)
                    file_count = file_count + 1
                    #===== empty the var
                    source[:] = []
                    target[:] = []
                    hist_item_list[:] = []
                    hist_cate_list[:] = []
                    hist_shop_list[:] = []
                    hist_node_list[:] = []
                    hist_product_list[:] = []
                    hist_brand_list[:] = []
                 
                    neg_hist_item_list[:] = []
                    neg_hist_cate_list[:] = []
                    neg_hist_shop_list[:] = []
                    neg_hist_node_list[:] = []
                    neg_hist_product_list[:] = []
                    neg_hist_brand_list[:] = []
           # for residual lines
     
            if(line_count != 0 ):     
                line_count = 0
                #print("Total processed lines = ", line_count,",spent time = ",time.time() -st)
                source_array = np.array(source)
                target_array = np.array(target)
                history_item_array = np.array(hist_item_list)        
                history_cate_array = np.array(hist_cate_list)
                history_shop_array = np.array(hist_shop_list)
                history_node_array = np.array(hist_node_list)
                history_product_array = np.array(hist_product_list)
                history_brand_array = np.array(hist_brand_list)
                neg_history_item_array = np.array(neg_hist_item_list)        
                neg_history_cate_array = np.array(neg_hist_cate_list)
                neg_history_shop_array = np.array(neg_hist_shop_list)
                neg_history_node_array = np.array(neg_hist_node_list)
                neg_history_product_array = np.array(neg_hist_product_list)
                neg_history_brand_array = np.array(neg_hist_brand_list)
                print("Finished array transform")
                np.savez('./' + file_name + '_' + str(file_count) + '_processed.npz', source_array=source_array,target_array=target_array,
    	                 history_item_array=history_item_array,history_cate_array=history_cate_array, 
                         history_shop_array=history_shop_array,history_node_array=history_node_array,
                         history_product_array=history_product_array,history_brand_array=history_brand_array,
                         neg_history_item_array=neg_history_item_array,neg_history_cate_array=neg_history_cate_array,
                         neg_history_shop_array=neg_history_shop_array,neg_history_node_array=neg_history_node_array,
                         neg_history_product_array=neg_history_product_array,neg_history_brand_array=neg_history_brand_array)
                file_count = file_count + 1
                #===== empty the var
                source[:] = []
                target[:] = []
                hist_item_list[:] = []
                hist_cate_list[:] = []
                hist_shop_list[:] = []
                hist_node_list[:] = []
                hist_product_list[:] = []
                hist_brand_list[:] = []
                 
                neg_hist_item_list[:] = []
                neg_hist_cate_list[:] = []
                neg_hist_shop_list[:] = []
                neg_hist_node_list[:] = []
                neg_hist_product_list[:] = []
                neg_hist_brand_list[:] = []


def loadDataset(
    raw_path 
):
    # dataset
    #output_filename = "dienDataset_processed"

    # read in the dataset
    #pos = raw_path.rfind('/')
    #data_path = raw_path[:pos]
    #data_file = raw_path[pos:]
    #npzfile = "." + data_file + "_processed.npz"
   
    data_path = raw_path
    max_length = 1000
    # If already processed, just load it
    file = preprocessDataset(data_path, max_length)
    
    #train_data_file = [ 'sample_00','sample_01', 'sample_02', 'sample_03']
    #train_data_file = [ 'sample_03']
    #file.process2( train_data_file, 'train_sample')

    test_data_file = ['test_sample_00', 'test_sample_01', 'test_sample_02', 'test_sample_03']
    file.process2( test_data_file, 'test_sample')


if __name__ == "__main__":
    ### import packages ###
    import argparse
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Preprocess Alibaba dataset"
    )
    # model related parameters
    parser.add_argument("--raw-data-file", type=str, default="")
    args = parser.parse_args()
    print("Start process ", args.raw_data_file) 
    # control randomness
    random.seed(0)
    
    loadDataset(
        args.raw_data_file
    )
