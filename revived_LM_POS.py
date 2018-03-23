import sys
import getopt
import pickle
import os
import time, datetime
import tensorflow as tf
import numpy as np
import copy


weight_path = "./print_files/20180322_060642_50_32_0.009_320/weights"

class Config:
    max_length = 50  # should be factor of 600 ;longest sequence to parse
    n_classes = 672 #size of vocabulary (since we're predicting at every sequence)
    embed_size = 100
    hidden_size = 1000
    batch_split = int(128*(600/max_length))
    batch_size = 32 ## should be factor of 128
    cell = "lstm"
    end_token = 328
    glove_matrix = "./POSGlove/100d_glove_matrix.npy"
    path_to_aashiq = "./aashiqdict"
    max_num_batches = 2  #end result is 10 * 2
    good_directory = "./FinalData/batches/ML600BS128/good_train"
    bad_directory = "./FinalData/batches/ML600BS128/bad_train"

    def __init__(self):
        pass

class Revived_Language_Model():
    def __init__(self,pretrained, config):

        self.config = config
        self.pretrained_embeddings = tf.get_variable(initializer=pretrained, name="pretrained_embeddings")   ###pretrained embeddings (vocab x embedding_size)

        #placeholders
        self.input_placeholder = tf.placeholder(shape=[None, self.config.max_length], dtype=tf.int32)   ### currently working with fixed length
        self.labels_placeholder = tf.placeholder(shape=[None, self.config.max_length], dtype=tf.int32)  ### shape = (max_length x m)

        #regularization
        #self.regularizer = tf.contrib.layers.l2_regularizer(scale = 0.03)
        #self.l1 = tf.contrib.layers.l1_regularizer(scale = 0.03)
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)

        self.num_good_batches, self.good_batches = self.doc_load(self.config.good_directory)
        print(self.good_batches[0].shape, "shape of batch")
        self.num_bad_batches, self.bad_batches = self.doc_load(self.config.bad_directory)

        #self.shuffled = np.random.shuffle(np.arange(self.num_batches)) #self.shuffle_batches()

        # self.print_train_acc_list = []
        # self.print_good_dev_loss_list = []
        # self.print_good_dev_acc_list = []
        # self.print_bad_dev_loss_list = []
        # self.print_bad_dev_acc_list = []
        # self.print_train_loss_list = []
        # self.print_sample = []

        ##restore model
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def add_embedding(self):
        flattened = tf.reshape(self.input_placeholder, [-1])
        look_up = tf.nn.embedding_lookup(params=self.pretrained_embeddings, ids=flattened)
        #look_up = tf.Print(look_up, [self.pretrained_embeddings[0,:]], "embedding")
        embeddings = tf.reshape(look_up, [-1, self.config.max_length, self.config.embed_size])
        #### batch x 600 x 50 ** THIS DEF WORKS; WE CHECKED ON COLAB --ES; Shank
        #### batch x max_len x embed_size
        return embeddings

    def add_prediction_op(self):
        x = self.add_embedding()    ### batch x max_size(time_step) x embedding_size(num_features)
#        with tf.variable_scope("RNN"):
        print("after embedding")
        print(tf.shape(x))
        print(x.shape)
        xavier = tf.contrib.layers.xavier_initializer()
        lstm_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size, initializer=xavier)
#        lstm_cell  = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.config.dropout, output_keep_prob=self.config.dropout)
        W_y = tf.get_variable("W_y", initializer=xavier, shape=[self.config.hidden_size, self.config.n_classes],
                        dtype=tf.float64)

        b_2 = tf.get_variable("b_2", initializer=tf.zeros(shape=[1, self.config.n_classes], dtype=tf.float64))
        x = tf.unstack(x, axis=1)
        #b = tf.Print(x, [x], "shape of xadf")
        outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float64)
        #print(outputs, "outputs")
        outputs = tf.convert_to_tensor(outputs)
        #outpus: seq X batch X hidden
        time_step = self.config.max_length
        batch, hidden = self.config.batch_size, self.config.hidden_size

        #W_y = hidden, vocab
        #
        inner_reshape = tf.reshape(outputs, [time_step*batch, hidden]) #76800 , 50
        product = tf.matmul(inner_reshape, W_y)  #50, 663 >  50918400
        z = tf.reshape(product, [time_step, batch, self.config.n_classes]) + b_2
        ##prediction at everytime step, batch x  vocab
        #idjeifj#z = tf.stack(all_steps_preds)   #### time_steps x batch x vocab
        return z

    def add_loss_op(self, pred):
        #pred : time_steps x batch x vocab
        # labels_placeholder is of shape : batch * max_step
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.transpose(self.labels_placeholder), logits=pred)
        batch_loss = tf.reduce_mean(losses)
        return batch_loss ###vector of batchsize (each elemnt is the mean loss)

    def doc_load(self, dir_path):
        files = os.listdir(dir_path)
        paths = list(map(lambda x: os.path.join(dir_path, x),files))       ###each of these files is a batch of text (batch x max length) :this needs to be pre-randomized
        paths.sort()
        all_batches = [np.load(open(p, "rb")).reshape((-1,self.config.max_length)) for p in paths]
        batches_reduced = []
        for i in range(len(all_batches)):
            batches_reduced += np.split(all_batches[i], self.config.batch_split/self.config.batch_size)
        #all_batches = [all_batches[i][0,:self.config.max_length].reshape((self.config.batch_size,self.config.max_length)) for i in range(10)]
        batches_reduced = batches_reduced[:self.config.max_num_batches]
        print("batches size", len(batches_reduced))    ### 1000
        print("batch size", batches_reduced[0].shape)  ###32 x 50
        return len(batches_reduced), batches_reduced

    def time_step_preds(self, preds):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=preds)
        print("exp shape", tf.exp(losses).shape) # batch x 50
        return tf.exp(-losses)

    def produce_labels(self, examples):
        labels = copy.deepcopy(examples)
        reduced = labels[:,1:]
        labels = np.hstack((reduced, np.full((labels.shape[0],1), self.config.end_token)))   #### remember to replace end token with the right token
        return labels

    def save_forwardprop(self):
        with tf.Session() as session:
            session.run(self.init)
            self.saver.restore(session, os.path.join(weight_path, "1"))
            example_labels = []    #1:good, 0:bad
            example_data = []
            for b in range(self.config.max_num_batches):
                print("max_num_batches", self.config.max_num_batches)
                print("num_batches", len(self.good_batches))
                gD = self.good_batches[b]
                bD = self.bad_batches[b]
                gL = self.produce_labels(self.good_batches[b])
                bL = self.produce_labels(self.bad_batches[b])

                print("label shape", gL.shape)
                print("batch from class", self.good_batches[b].shape)
                print("batch copied shape", gD.shape)
                print("split shape", len(tf.split(self.forward_prop(session, gD, gL), self.config.batch_size, axis=0)))
                example_data += tf.split(self.forward_prop(session, gD, gL), self.config.batch_size, axis=0)
                example_labels += [1]*self.config.batch_size
                example_data += tf.split(self.forward_prop(session, bD, bL), self.config.batch_size, axis=0)
                example_labels += [0]*self.config.batch_size
            print("example label count")
            print(len(example_labels))
            print(len(example_labels))
            assert (len(example_labels) == 2 * self.config.batch_size * self.config.max_num_batches)
            shuffle = np.arange(len(example_labels))
            np.random.shuffle(shuffle)
           
            example_labels = [example_labels[i] for i in shuffle]
            example_data = [example_data[i].eval(feed_dict={self.input_placeholder:gD, self.labels_placeholder:gL})[0] for i in shuffle]
        np.save(open('TEMP', 'wb'),example_labels)
        np.save(open('TEMPdata','wb'),example_data)

    def forward_prop(self, sess, example_batch, label_batch):  ##returns preds (time_steps x batch_size) one batch at a time
        forward_preds, forward_perplexity = sess.run([self.pred, self.loss], feed_dict={self.input_placeholder: example_batch, self.labels_placeholder: label_batch})
        #forward_preds : time_steps x batch x vocab
        #input preds: time_steps x batch
        return self.time_step_preds(forward_preds)


def make_input():
    embeddings = np.load(Config.glove_matrix)
    LM = Revived_Language_Model(pretrained=embeddings, config=Config)
    LM.save_forwardprop()
   
   

if __name__ == "__main__":
    make_input()
