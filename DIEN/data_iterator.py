import numpy
import json
#import pickle as pkl
import random
import numpy as np
import sys
from functools import wraps
import time
from wrap_time import time_it
import copy

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)


class DataIterator:
    
    @time_it(freq=10)
    def __init__(self, source, dict_list,
                 batch_size=128,
                 maxlen=2000,
                 skip_empty=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None,
                 parall=False
                ):
        self.source = open(source, 'r')
        self.batch_shuffle = 1
        self.neg_sample = 'LastInstance' # 'Random' or 'LastInstance'
        #self.user_dict = copy.copy(dict_list[0])
        #self.item_dict = copy.copy(dict_list[1])
        #self.cate_dict = copy.copy(dict_list[2])
        #self.shop_dict = copy.copy(dict_list[3])
        #self.node_dict = copy.copy(dict_list[4])
        #self.product_dict = copy.copy(dict_list[5])
        #self.brand_dict = copy.copy(dict_list[6])
        #self.item_info = copy.copy(dict_list[7])
        self.user_dict, self.item_dict, self.cate_dict, self.shop_dict, self.node_dict, self.product_dict, self.brand_dict, self.item_info = dict_list
        #  self.user_dict = json.load(open('user_voc.json', 'r'))
        #  self.item_dict = json.load(open('item_voc.json', 'r'))
        #  self.cate_dict = json.load(open('cate_voc.json', 'r'))
        #  self.shop_dict = json.load(open('shop_voc.json', 'r'))
        #  self.node_dict = json.load(open('node_voc.json', 'r'))
        #  self.product_dict = json.load(open('product_voc.json', 'r'))
        #  self.brand_dict = json.load(open('brand_voc.json', 'r'))
        #  self.item_info = json.load(open('item_info.json', 'r'))
        self.all_items = self.item_info.keys()
        self.num_items = len(self.all_items)
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty
        self.sort_by_length = sort_by_length
        self.max_catch_num = 20

        self.source_buffer = []
        #self.batch_size = batch_size * max_batch_size
        self.batch_size = batch_size
        self.neg_hist_catch = {}
        self.end_of_data = False
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

    def gen_item_block(self, item_idx):
        neg_item = self.all_items[item_idx]
        neg_cate = self.map_cate(self.item_info[neg_item][0])
        neg_shop = self.map_shop(self.item_info[neg_item][1])
        neg_node = self.map_node(self.item_info[neg_item][2])
        neg_product = self.map_product(self.item_info[neg_item][3])
        neg_brand = self.map_brand(self.item_info[neg_item][4])
        neg_item = self.map_item(self.all_items[item_idx])#map origin item to item_id
        return neg_item, neg_cate, neg_shop, neg_node, neg_product, neg_brand 

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
            self.neg_hist_catch[length] = self.neg_hist_catch.get(length, [])
            self.neg_hist_catch[length].append([neg_item_hist, neg_cate_hist, neg_shop_hist, neg_node_hist, neg_product_hist, neg_brand_hist])
            return [neg_item_hist, neg_cate_hist, neg_shop_hist, neg_node_hist, neg_product_hist, neg_brand_hist]
    
    def fill_ndarray(self, hist):
        nd_his = numpy.ones(self.maxlen) * -1
        nd_his[:len(hist)] = hist
        return nd_his
    

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
    

    @time_it(freq=20)
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

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
        count = 0
        if len(self.source_buffer) == 0:
            for k_ in xrange(self.batch_size):
                ss = self.source.readline()
                if ss == "" and count < self.batch_size:
                    self.end_of_data = True
                    self.source.seek(0)
                    ss = self.source.readline()
                self.source_buffer.append(ss.strip().split("^H"))
                count += 1

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
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
                    neg_item, neg_cate, neg_shop, neg_node, neg_product, neg_brand = self.last_item, self.last_cate, self.last_shop, self.last_node, self.last_product, self.last_brand
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
                random_neg_hist = self.gen_neg_hist(lengthx)
                #add positive sample
                source.append([uid, pos_item, pos_cate, pos_shop, pos_node, pos_product, pos_brand])
                target.append([1, 0])
                # drop the last item
                hist_item_list.append(self.fill_ndarray(hist_item[-(self.maxlen+1):-1]))
                hist_cate_list.append(self.fill_ndarray(hist_cate[-(self.maxlen+1):-1]))
                hist_shop_list.append(self.fill_ndarray(hist_shop[-(self.maxlen+1):-1]))
                hist_node_list.append(self.fill_ndarray(hist_node[-(self.maxlen+1):-1]))
                hist_product_list.append(self.fill_ndarray(hist_product[-(self.maxlen+1):-1]))
                hist_brand_list.append(self.fill_ndarray(hist_brand[-(self.maxlen+1):-1]))
                neg_hist_item_list.append(self.fill_ndarray(random_neg_hist[0]))
                neg_hist_cate_list.append(self.fill_ndarray(random_neg_hist[1]))
                neg_hist_shop_list.append(self.fill_ndarray(random_neg_hist[2]))
                neg_hist_node_list.append(self.fill_ndarray(random_neg_hist[3]))
                neg_hist_product_list.append(self.fill_ndarray(random_neg_hist[4]))
                neg_hist_brand_list.append(self.fill_ndarray(random_neg_hist[5]))
                #add negative sample
                source.append([uid, neg_item, neg_cate, neg_shop, neg_node, neg_product, neg_brand])
                target.append([0, 1])
                hist_item_list.append(self.fill_ndarray(hist_item[-(self.maxlen+1):-1]))
                hist_cate_list.append(self.fill_ndarray(hist_cate[-(self.maxlen+1):-1]))
                hist_shop_list.append(self.fill_ndarray(hist_shop[-(self.maxlen+1):-1]))
                hist_node_list.append(self.fill_ndarray(hist_node[-(self.maxlen+1):-1]))
                hist_product_list.append(self.fill_ndarray(hist_product[-(self.maxlen+1):-1]))
                hist_brand_list.append(self.fill_ndarray(hist_brand[-(self.maxlen+1):-1]))
                neg_hist_item_list.append(self.fill_ndarray(random_neg_hist[0]))
                neg_hist_cate_list.append(self.fill_ndarray(random_neg_hist[1]))
                neg_hist_shop_list.append(self.fill_ndarray(random_neg_hist[2]))
                neg_hist_node_list.append(self.fill_ndarray(random_neg_hist[3]))
                neg_hist_product_list.append(self.fill_ndarray(random_neg_hist[4]))
                neg_hist_brand_list.append(self.fill_ndarray(random_neg_hist[5]))
                if self.neg_sample == 'LastInstance':
                    self.last_item, self.last_cate, self.last_shop, self.last_node, self.last_product, self.last_brand = pos_item, pos_cate, pos_shop, pos_node, pos_product, pos_brand 
                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()
            
         
        uid_array = np.array(source)[:,0]
        item_array = np.array(source)[:,1]
        cate_array = np.array(source)[:,2]
        shop_array = np.array(source)[:,3]
        node_array = np.array(source)[:,4]
        product_array = np.array(source)[:,5]
        brand_array = np.array(source)[:,6]

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
        #history_neg_item_array = np.array(6eg_item_list)        
        #history_neg_cate_array = np.array(neg_cate_list)        
        
        history_mask_array = np.greater(history_item_array, 0)*1.0      
        if self.batch_shuffle:
            per = np.random.permutation(uid_array.shape[0])
            uid_array = uid_array[per]
            item_array = item_array[per]
            cate_array = cate_array[per]
            shop_array = shop_array[per]
            node_array = node_array[per]
            product_array = product_array[per]
            brand_array = brand_array[per]
            target_array = target_array[per]
            history_item_array = history_item_array[per, :]        
            history_cate_array = history_cate_array[per, :]
            history_shop_array = history_shop_array[per, :]
            history_node_array = history_node_array[per, :]
            history_product_array = history_product_array[per, :]
            history_brand_array = history_brand_array[per, :]
            
            neg_history_item_array = neg_history_item_array[per, :]        
            neg_history_cate_array = neg_history_cate_array[per, :]
            neg_history_shop_array = neg_history_shop_array[per, :]
            neg_history_node_array = neg_history_node_array[per, :]
            neg_history_product_array = neg_history_product_array[per, :]
            neg_history_brand_array = neg_history_brand_array[per, :]
            

        
        
        return (uid_array, item_array, cate_array, shop_array, node_array, product_array, brand_array), \
(target_array, history_item_array, history_cate_array, history_shop_array, history_node_array, history_product_array, history_brand_array, history_mask_array, neg_history_item_array, neg_history_cate_array, neg_history_shop_array, neg_history_node_array, neg_history_product_array, neg_history_brand_array)


