import sys
from os import path
import random
import numpy as np
import json
import time
import threading
from collections import deque

class DataLoader:

    def __init__(
        self,
        data_path,
        data_file,
        batch_size,
        data_file_num,
        sleep_time=1,
        max_queue_size = 2
    ):
        # load data 
        self.queue = deque() #multiprocessing.Queue(maxsize=max_queue_size) # it may change in future if we decide to split data into many small chunks instead of 4
        self.batch_size = batch_size
        self.data_path = data_path
        self.data_file = data_file
        self.data_file_num = data_file_num
        self.sleep_time = sleep_time
        self.max_queue_size = max_queue_size 
        self.help_count = 0
 
    def __iter__(self):
        return self

    def data_read(self, start_id, total_thread):
        sample_id = start_id
        while sample_id < self.data_file_num:
                if len(self.queue) >= self.max_queue_size:
                    time.sleep(1)
                    continue 
                processed_data_path = self.data_path + self.data_file + "_" + str(sample_id) + '_processed.npz'
                print('Start loading processed data...' + processed_data_path)
                st = time.time()
                data = np.load(processed_data_path)
                source = data['source_array']
                uid_array = np.array(source)[:,0]
                item_array = np.array(source)[:,1]
                cate_array = np.array(source)[:,2]
                shop_array = np.array(source)[:,3]
                node_array = np.array(source)[:,4]
                product_array = np.array(source)[:,5]
                brand_array = np.array(source)[:,6]
    
                target = data['target_array']
                history_item = data['history_item_array']        
                history_cate = data['history_cate_array']         
                history_shop = data['history_shop_array']         
                history_node = data['history_node_array']         
                history_product = data['history_product_array']         
                history_brand = data['history_brand_array']         
                
                neg_history_item = data['neg_history_item_array']         
                neg_history_cate = data['neg_history_cate_array']         
                neg_history_shop = data['neg_history_shop_array']         
                neg_history_node = data['neg_history_node_array']         
                neg_history_product = data['neg_history_product_array']         
                neg_history_brand = data['neg_history_brand_array']  
                print('Finish loading processed data id '+ str(sample_id) + ',Time cost = %.4f' % (time.time()-st))
                data_file  = (uid_array,item_array,cate_array,shop_array,node_array,product_array,brand_array,\
                    target, history_item,history_cate,history_shop, history_node,history_product,history_brand,\
                    neg_history_item,neg_history_cate,neg_history_shop, neg_history_node,neg_history_product,neg_history_brand)  
                while self.help_count % total_thread != start_id:
                    time.sleep(1)
                print('help_count=', self.help_count)
                self.queue.append(data_file)
                self.help_count += 1
                sample_id = sample_id + total_thread


    def _batch_data(self, data, data_slice):
        uid_array,item_array,cate_array,shop_array,node_array,product_array,brand_array,\
            target, history_item,history_cate,history_shop, history_node,history_product,history_brand,\
                neg_history_item,neg_history_cate,neg_history_shop, neg_history_node,neg_history_product,neg_history_brand = data
        #print("in _batch_data func")
        user_id = uid_array[data_slice]
        item_id = item_array[data_slice]
        cate_id = cate_array[data_slice]
        shop_id = shop_array[data_slice]
        node_id = node_array[data_slice]
        product_id = product_array[data_slice]
        brand_id = brand_array[data_slice]
        label = target[data_slice, :]
        hist_item = history_item[data_slice, :]
        hist_cate = history_cate[data_slice, :]
        hist_shop = history_shop[data_slice, :]
        hist_node = history_node[data_slice, :]
        hist_product = history_product[data_slice, :]
        hist_brand = history_brand[data_slice, :]
        
        hist_mask = np.greater( hist_item, 0) * 1.0
        
        neg_hist_item = neg_history_item[data_slice, :]
        neg_hist_cate = neg_history_cate[data_slice, :]
        neg_hist_shop = neg_history_shop[data_slice, :]
        neg_hist_node = neg_history_node[data_slice, :]
        neg_hist_product = neg_history_product[data_slice, :]
        neg_hist_brand = neg_history_brand[data_slice, :]

        return [user_id, item_id, cate_id,shop_id, node_id, product_id, brand_id, 
            label, hist_item, hist_cate, hist_shop, hist_node, hist_product, hist_brand, 
            hist_mask, neg_hist_item, neg_hist_cate, neg_hist_shop, neg_hist_node, 
            neg_hist_product, neg_hist_brand ]

    def next(self):
        previous_data_out = []
        data_file_read = 0
        batch_id = 0
        #print('in next func')
        #import pdb; pdb.set_trace()
        previous_line = 0
        while len(self.queue) < 2:
            time.sleep(1)
        print ('Now the queue has two data file loaded in!')
        while data_file_read < self.data_file_num:
            if len(self.queue) == 0:
                time.sleep(1)
                continue
            data = self.queue.popleft()
            file_line_num = data[0].shape[0]
            start_ind = 0
            data_file_read = data_file_read + 1
            stime = time.time()
            #print('start one file,time=', stime)
            while start_ind <= file_line_num - self.batch_size:

                if previous_line != 0:
                    batch_left = self.batch_size - previous_line
                else:
                    batch_left = self.batch_size
                data_slice = slice(start_ind, start_ind + batch_left)
                # slice the data from the list
                data_out = self._batch_data(data, data_slice) #data_out is tuple
                
                if previous_line != 0:
                    #attach the data
                    for i in range(len(data_out)):
                        data_out[i] = np.concatenate(
                            [previous_data_out[i], data_out[i]],
                            axis=0
                        )
                if self.batch_size != len(data_out[0]):
                    raise ValueError('batch fetched wrong!')
                
                start_ind = start_ind + batch_left
             
                previous_line = 0
                #print("start_ind ", start_ind)
                yield data_out
            if start_ind != file_line_num:
                data_slice = slice(start_ind, file_line_num)
                previous_data_out = self._batch_data(data, data_slice)
                previous_line = file_line_num - start_ind
                print("Left batch of size %d" %( previous_line))
            etime = time.time()
            print('Consume one file takes time= %.4f' %(etime-stime))
        print ('drop last batch since it is not full batch size')

   
def test():
    data_load = DataLoader('/disk3/w.wei/dien-new/process_data_maxlen100_0225/', 'train_sample', 256, 15)
    producer1 = threading.Thread(target=data_load.data_read, args=(0, 3))  
    producer2 = threading.Thread(target=data_load.data_read, args=(1, 3))  
    producer3 = threading.Thread(target=data_load.data_read, args=(2, 3))  
    producer1.start()
    producer2.start()
    producer3.start()
    #data_i = iter(data_load)
    #data_o = next(data_i)
    #print('print=====',len(data_o))
    num = 0
    for data in data_load.next():
        num = num+1
        cnt = 1
        for i in range(10000):
            cnt = cnt * 1.0
        if num%1000 == 0:
            print('i=',num,',cnt=',cnt)

    producer1.join()
    producer2.join()
    producer3.join()

if __name__ == '__main__':
    test()
