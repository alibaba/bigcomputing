#coding:utf-8
import tensorflow as tf
from utils import *
from tensorflow.python.ops.rnn_cell import GRUCell
from rnn import dynamic_rnn 
# import mann_simple_cell as mann_cell
class Model(object):
    def __init__(self, uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=False, Flag="DNN"):
        self.model_flag = Flag
        self.use_negsample= use_negsample
        def get_embeddings_variable(var_name, embedding_shape):
            # workaround to return vector of 0
            embeddings = tf.get_variable(var_name, embedding_shape, trainable=True)
            embeddings = tf.concat([ embeddings, [tf.constant([0.] * embedding_shape[1])] ], axis = 0)
            return embeddings 

        with tf.name_scope('Inputs'):
            self.item_id_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='item_id_his_batch_ph')
            self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph')
            self.shop_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='shop_his_batch_ph')
            self.node_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='node_his_batch_ph')
            self.product_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='product_his_batch_ph')
            self.brand_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='brand_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.item_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='item_id_batch_ph')
            self.cate_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_id_batch_ph')
            self.shop_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='shop_id_batch_ph')
            self.node_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='node_id_batch_ph')
            self.product_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='product_id_batch_ph')
            self.brand_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='brand_id_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        # Embedding layer
        with tf.name_scope('Embedding_layer'):

            #self.item_id_embeddings_var = tf.get_variable("item_id_embedding_var", [item_n, EMBEDDING_DIM], trainable=True)
            self.item_id_embeddings_var = get_embeddings_variable("item_id_embedding_var", [item_n, EMBEDDING_DIM])
            self.item_id_batch_embedded = tf.nn.embedding_lookup(self.item_id_embeddings_var, self.item_id_batch_ph)
            self.item_id_his_batch_embedded = tf.nn.embedding_lookup(self.item_id_embeddings_var, self.item_id_his_batch_ph)
           
            self.cate_id_embeddings_var = tf.get_variable("cate_id_embedding_var", [cate_n, EMBEDDING_DIM], trainable=True)
            self.cate_id_batch_embedded = tf.nn.embedding_lookup(self.cate_id_embeddings_var, self.cate_id_batch_ph)
            self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.cate_id_embeddings_var, self.cate_his_batch_ph)

            #self.shop_id_embeddings_var = tf.get_variable("shop_id_embedding_var", [shop_n, EMBEDDING_DIM], trainable=True)
            self.shop_id_embeddings_var = get_embeddings_variable("shop_id_embedding_var", [shop_n, EMBEDDING_DIM])
            self.shop_id_batch_embedded = tf.nn.embedding_lookup(self.shop_id_embeddings_var, self.shop_id_batch_ph)
            self.shop_his_batch_embedded = tf.nn.embedding_lookup(self.shop_id_embeddings_var, self.shop_his_batch_ph)

            self.node_id_embeddings_var = tf.get_variable("node_id_embedding_var", [node_n, EMBEDDING_DIM], trainable=True)
            self.node_id_batch_embedded = tf.nn.embedding_lookup(self.node_id_embeddings_var, self.node_id_batch_ph)
            self.node_his_batch_embedded = tf.nn.embedding_lookup(self.node_id_embeddings_var, self.node_his_batch_ph)
            
            self.product_id_embeddings_var = tf.get_variable("product_id_embedding_var", [product_n, EMBEDDING_DIM], trainable=True)
            self.product_id_batch_embedded = tf.nn.embedding_lookup(self.product_id_embeddings_var, self.product_id_batch_ph)
            self.product_his_batch_embedded = tf.nn.embedding_lookup(self.product_id_embeddings_var, self.product_his_batch_ph)
            
            #self.brand_id_embeddings_var = tf.get_variable("brand_id_embedding_var", [brand_n, EMBEDDING_DIM], trainable=True)
            self.brand_id_embeddings_var = get_embeddings_variable("brand_id_embedding_var", [brand_n, EMBEDDING_DIM])
            self.brand_id_batch_embedded = tf.nn.embedding_lookup(self.brand_id_embeddings_var, self.brand_id_batch_ph)
            self.brand_his_batch_embedded = tf.nn.embedding_lookup(self.brand_id_embeddings_var, self.brand_his_batch_ph)
       
        if self.use_negsample:
            self.item_id_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_his_batch_ph')
            self.cate_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_cate_his_batch_ph')
            self.shop_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_shop_his_batch_ph')
            self.node_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_node_his_batch_ph')
            self.product_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_product_his_batch_ph')
            self.brand_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_brand_his_batch_ph')
            self.neg_item_his_eb = tf.nn.embedding_lookup(self.item_id_embeddings_var, self.item_id_neg_batch_ph)
            self.neg_cate_his_eb = tf.nn.embedding_lookup(self.cate_id_embeddings_var, self.cate_neg_batch_ph)
            self.neg_shop_his_eb = tf.nn.embedding_lookup(self.shop_id_embeddings_var, self.shop_neg_batch_ph)
            self.neg_node_his_eb = tf.nn.embedding_lookup(self.node_id_embeddings_var, self.node_neg_batch_ph)
            self.neg_product_his_eb = tf.nn.embedding_lookup(self.product_id_embeddings_var, self.product_neg_batch_ph)
            self.neg_brand_his_eb = tf.nn.embedding_lookup(self.brand_id_embeddings_var, self.brand_neg_batch_ph)
            self.neg_his_eb = tf.concat([self.neg_item_his_eb,self.neg_cate_his_eb, self.neg_shop_his_eb, self.neg_node_his_eb, self.neg_product_his_eb, self.neg_brand_his_eb], axis=2) * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1))   
            
        self.item_eb = tf.concat([self.item_id_batch_embedded, self.cate_id_batch_embedded, self.shop_id_batch_embedded, self.node_id_batch_embedded, self.product_id_batch_embedded, self.brand_id_batch_embedded], axis=1)
        self.item_his_eb = tf.concat([self.item_id_his_batch_embedded,self.cate_his_batch_embedded, self.shop_his_batch_embedded, self.node_his_batch_embedded, self.product_his_batch_embedded, self.brand_his_batch_embedded], axis=2) * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1))
        #debug if last item of history is leaked
        #self.item_his_eb = self.item_his_eb[:,:-1,:]
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, scope='prelu_1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, scope='prelu_2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsample:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask = None, stag = None):
        #mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]

        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask

        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.000001
        return y_hat
                            
    
    def train(self, sess, inps):
        if self.use_negsample:
            loss, aux_loss, accuracy, _ = sess.run([self.loss, self.aux_loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.item_id_batch_ph: inps[1],
                self.cate_id_batch_ph: inps[2],
                self.shop_id_batch_ph: inps[3],
                self.node_id_batch_ph: inps[4],
                self.product_id_batch_ph: inps[5],
                self.brand_id_batch_ph: inps[6],
                self.item_id_his_batch_ph: inps[7],
                self.cate_his_batch_ph: inps[8],
                self.shop_his_batch_ph: inps[9],
                self.node_his_batch_ph: inps[10],
                self.product_his_batch_ph: inps[11],
                self.brand_his_batch_ph: inps[12],
                self.item_id_neg_batch_ph: inps[13],
                self.cate_neg_batch_ph: inps[14],
                self.shop_neg_batch_ph: inps[15],
                self.node_neg_batch_ph: inps[16],
                self.product_neg_batch_ph: inps[17],
                self.brand_neg_batch_ph: inps[18],
                self.mask: inps[19],
                self.target_ph: inps[20],
                self.lr: inps[21]
            })
        else:
            loss, aux_loss, accuracy, _ = sess.run([self.loss, self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.item_id_batch_ph: inps[1],
                self.cate_id_batch_ph: inps[2],
                self.shop_id_batch_ph: inps[3],
                self.node_id_batch_ph: inps[4],
                self.product_id_batch_ph: inps[5],
                self.brand_id_batch_ph: inps[6],
                self.item_id_his_batch_ph: inps[7],
                self.cate_his_batch_ph: inps[8],
                self.shop_his_batch_ph: inps[9],
                self.node_his_batch_ph: inps[10],
                self.product_his_batch_ph: inps[11],
                self.brand_his_batch_ph: inps[12],
                # self.item_id_neg_batch_ph: inps[13],
                # self.cate_neg_batch_ph: inps[14],
                # self.shop_neg_batch_ph: inps[15],
                # self.node_neg_batch_ph: inps[16],
                # self.product_neg_batch_ph: inps[17],
                # self.brand_neg_batch_ph: inps[18],
                self.mask: inps[19],
                self.target_ph: inps[20],
                self.lr: inps[21]
            })
        
        return loss, accuracy, aux_loss, 0, 0 

    def calculate(self, sess, inps):
        if self.use_negsample:
            probs, loss, aux_loss, accuracy = sess.run([self.y_hat, self.loss, self.aux_loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.item_id_batch_ph: inps[1],
                self.cate_id_batch_ph: inps[2],
                self.shop_id_batch_ph: inps[3],
                self.node_id_batch_ph: inps[4],
                self.product_id_batch_ph: inps[5],
                self.brand_id_batch_ph: inps[6],
                self.item_id_his_batch_ph: inps[7],
                self.cate_his_batch_ph: inps[8],
                self.shop_his_batch_ph: inps[9],
                self.node_his_batch_ph: inps[10],
                self.product_his_batch_ph: inps[11],
                self.brand_his_batch_ph: inps[12],
                self.item_id_neg_batch_ph: inps[13],
                self.cate_neg_batch_ph: inps[14],
                self.shop_neg_batch_ph: inps[15],
                self.node_neg_batch_ph: inps[16],
                self.product_neg_batch_ph: inps[17],
                self.brand_neg_batch_ph: inps[18],
                self.mask: inps[19],
                self.target_ph: inps[20],
            })
        else:
            probs, loss, aux_loss, accuracy = sess.run([self.y_hat, self.loss, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.item_id_batch_ph: inps[1],
                self.cate_id_batch_ph: inps[2],
                self.shop_id_batch_ph: inps[3],
                self.node_id_batch_ph: inps[4],
                self.product_id_batch_ph: inps[5],
                self.brand_id_batch_ph: inps[6],
                self.item_id_his_batch_ph: inps[7],
                self.cate_his_batch_ph: inps[8],
                self.shop_his_batch_ph: inps[9],
                self.node_his_batch_ph: inps[10],
                self.product_his_batch_ph: inps[11],
                self.brand_his_batch_ph: inps[12],
                # self.item_id_neg_batch_ph: inps[13],
                # self.cate_neg_batch_ph: inps[14],
                # self.shop_neg_batch_ph: inps[15],
                # self.node_neg_batch_ph: inps[16],
                # self.product_neg_batch_ph: inps[17],
                # self.brand_neg_batch_ph: inps[18],
                self.mask: inps[19],
                self.target_ph: inps[20],
            })
        return probs, loss, accuracy, aux_loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_DNN(Model):
    def __init__(self,uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_DNN, self).__init__(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DNN")
        
        inp = tf.concat([self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)
        

class Model_PNN(Model):
    def __init__(self,uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_PNN, self).__init__(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="PNN")
        
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)


class Model_GRU4REC(Model):
    def __init__(self,uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_GRU4REC, self).__init__(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="GRU4REC")
        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, final_state1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
                    
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, final_state1], 1)
        self.build_fcn_net(inp, use_dice=False)
        

class Model_DIN(Model):
    def __init__(self,uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_DIN, self).__init__(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DIN")
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, HIDDEN_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, att_fea], -1)
        self.build_fcn_net(inp, use_dice=False)


class Model_ARNN(Model):
    def __init__(self,uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_ARNN, self).__init__(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="ARNN")
        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, final_state1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_gru = din_attention(self.item_eb, rnn_outputs, HIDDEN_SIZE, self.mask)
            #att_gru = din_attention(self.item_eb, rnn_outputs, HIDDEN_SIZE, None)
            att_gru = tf.reduce_sum(att_gru, 1)
            #att_hist = din_attention(self.item_eb, self.item_his_eb, HIDDEN_SIZE, None, stag="att")
            #att_hist = tf.reduce_sum(att_hist, 1)

        inp = tf.concat([self.item_eb, self.item_his_eb_sum, final_state1, att_gru], -1)
        self.build_fcn_net(inp, use_dice=False)        

class Model_DIEN(Model):
    def __init__(self, uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, BATCH_SIZE, SEQ_LEN=400, use_negsample=False):
        super(Model_DIEN, self).__init__(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE, MEMORY_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, use_negsample, Flag="DIEN")

        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)        
        
        if use_negsample:
            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.neg_his_eb[:, 1:, :], self.mask[:, 1:], stag = "bigru_0")
            self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            #att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, HIDDEN_SIZE, self.mask, softmax_stag=1, stag='1_1', mode="LIST", return_alphas=True)
            att_outputs, alphas = din_attention(self.item_eb, rnn_outputs, HIDDEN_SIZE, mask=self.mask, mode="LIST", return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.sequence_length, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)
