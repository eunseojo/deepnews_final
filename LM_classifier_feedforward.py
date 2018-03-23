import sys
import pickle
import os
import time
import datetime
import getopt
import numpy as np
from collections import OrderedDict
import tensorflow as tf
import math

class Config:
    max_length = 600  # text size
    embed_size = 50
    batch_size = 256
    n_classes = 1
    hidden_size = 100
    n_epochs = 100
    lr = 0.0001
    dropout = 0.0

    def __init__(self):
        pass

class FeedforwardModel():
    '''
    This is the Feedforward Model class. It takes in a 100 dimensional BOW vector per document example and
     makes a binary prediction (high or low quality news)
    '''

    def __init__(self, data, config, labels):
        '''
        :param data: tuple of data (train, dev, test) with corresponding lengths of documents
        :param config: configuration for our current model initialization
        :param labels: tuple of labels (train, dev, test) corresponding with data
        Initialize class variables
        '''
        self.documents  = np.array(data[0]).T       #### TRAIN DATA (embed size x m examples)
        assert(self.documents.shape[0] == config.embed_size)
        self.labels = np.array(labels[0]).reshape(1,len(labels[0]))    #### TRAIN LABELS (1 x m examples)
        assert(self.labels.shape[1] == self.documents.shape[1])
        self.dev_examples  = np.array(data[1]).T
        self.dev_labels = np.array(labels[1]).reshape(1,len(labels[1]))
        self.test_examples  = np.array(data[2]).T
        self.test_labels = np.array(labels[2]).reshape(1,len(labels[2]))
        #self.new_test_examples  = np.array(data[3][0]).T
        #self.new_test_labels = np.array(labels[3]).reshape(1,len(labels[3]))
        
        self.config = config
        self.num_batch = int(self.documents.shape[1]/config.batch_size)     #### number of train batches (take floor)
        print("number of batches: " + str(self.num_batch))
        print("each batch contains: " + str(self.config.batch_size))
        print("size of train set: " + str((self.documents.shape[1])))
        print("size of dev set: " + str((self.dev_examples).shape[1]))
        print("size of test set: " + str((self.test_examples).shape[1]))
        #print("size of new test set: " + str((self.new_test_examples).shape[1]))

        ### initialize placeholders ###
        self.input_placeholder = tf.placeholder(shape=[self.config.embed_size, None], dtype=tf.float64, name="input_placeholder")  #(600,m)
        self.labels_placeholder = tf.placeholder(shape=[1, None], dtype=tf.float64, name="labels_placeholder")   # (1,m)

        ### initialize weights ###
        xavier = tf.contrib.layers.xavier_initializer()
        self.W_1 = tf.get_variable("W_1", initializer = xavier, shape =[self.config.hidden_size, self.config.embed_size], dtype=tf.float64)
        self.b_1 = tf.get_variable("b_1", initializer= tf.zeros_initializer, shape =[self.config.hidden_size, 1], dtype=tf.float64)
        self.W_2 = tf.get_variable("W_2", initializer = xavier, shape =[self.config.n_classes, self.config.hidden_size], dtype=tf.float64)
        self.b_2 = tf.get_variable("b_2", initializer=tf.zeros_initializer, shape =[self.config.n_classes, 1], dtype = tf.float64)

        ### forward prop ###
        # two-layer feedforward model (relu activation)
        # A1 = relu(W_1, x) + b_1
        # Z2 = W_2 x A1 + b_2

        self.A1 = tf.nn.relu(tf.matmul(self.W_1,self.input_placeholder) + self.b_1)

        self.Z2 = tf.matmul(self.W_2,self.A1) + self.b_2

        ### sigmoid cross entropy loss averaged over m examples ###
        self.labels1 = tf.transpose(self.labels_placeholder)
        self.logits = tf.transpose(self.Z2)

        self.sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels1,logits=self.logits)
        self.loss = tf.reduce_sum(self.sigmoid_cross_entropy)

        #### optimizer (Adam) ####
        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)

        #### log acc/loss lists ###
        self.print_train_loss_list = []
        self.print_dev_acc_list = []
        self.print_test_loss_list = []
        self.print_dev_loss_list = []
        self.print_train_acc_list = []
        self.print_new_test_loss_list = []
        self.print_new_test_acc_list = []

    def print_helper(self, write_file):
        '''
        :param write_file: path to open file
        Files Details of the model to given write_file
        '''

        write_file.write("max_length: " + str(self.config.max_length) + "\n")
        write_file.write("embed_size:" + str(self.config.embed_size) + "\n")
        write_file.write("classes: " + str(self.config.n_classes) + "\n")
        write_file.write("hidden_size: " + str(self.config.hidden_size) + "\n")
        write_file.write("n_epochs: " + str(self.config.n_epochs) + "\n")
        write_file.write("learn_rate: "+ str(self.config.lr) + "\n")
        write_file.write("batch_size: " + str(self.config.batch_size)+ "\n")
        write_file.write("layers: " + str(1) + "\n")
        write_file.write("num_buckets: " + str(self.num_batch) + "\n")
        write_file.write("batch_size: " + str(self.config.batch_size) + "\n")
        write_file.write("size_test_set: " + str(len(self.test_examples))  + "\n")
        write_file.write("size_dev_set: " + str(len(self.dev_examples)) + "\n")
        write_file.write("size_train_set: " + str(len(self.documents)) + "\n")
        write_file.flush()

    def random_mini_batches(self, X, Y, mini_batch_size):
        '''
        Creates a list of random minibatches from (X, Y)
        :param X: input data (n_x , m)
        :param Y: "true" labels (1, m)
        :param mini_batch_size: size of mini-batches (integer)
        :return: mini_batches: list of synchronous (mini_batches_X, mini_batches_Y)
        '''

        m = X.shape[1]                  # number of training examples
        mini_batches = []
        # np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        #shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
        shuffled_Y = Y[:, permutation].reshape((1,m))
        
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, self.num_batch):
            mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def sigmoid(self, z):
        s = 1.0 / (1.0 + np.exp(-1.0 * z))
        return s

    def train(self, file_print):

        '''
        Trains the model with the data/labels/config fed during initialization

        :param file_print: path to log file
        '''
        self.print_helper(file_print)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.config.n_epochs):   #iterate through epochs
                # average_loss_over_epoch = 0
                print ("-----epoch no." + str(i) + "------")
                epoch_cost_batches = 0
                epoch_preds = []
                epoch_labels = []
                num_minibatches = int(self.documents.shape[1] / self.config.batch_size)
                minibatches = self.random_mini_batches(X = self.documents,Y = self.labels, mini_batch_size = self.config.batch_size)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , minibatch_cost = sess.run([self.train_op, self.loss], feed_dict={self.input_placeholder: minibatch_X, self.labels_placeholder: minibatch_Y})
                
                    epoch_cost_batches += minibatch_cost / num_minibatches
                    self.print_train_loss_list.append(minibatch_cost)

                print("epoch cost is %s" % str(epoch_cost_batches))
                Z2_preds = sess.run(self.Z2, feed_dict={self.input_placeholder: self.documents})
                train_preds = Z2_preds > 0
                assert(len(train_preds[0]) == len(self.labels[0]))
                train_epoch_accuracy = sum([(train_preds[0][_] == self.labels[0][_]) for _ in range(len(self.labels[0]))])/float(len(self.labels[0]))
                print ("train accuracy in current epoch %s" % str(train_epoch_accuracy))
                self.print_train_acc_list.append(train_epoch_accuracy)


                #### calculate dev accuracy ###
                Z2_preds_new, loss_new = sess.run([self.Z2, self.loss], feed_dict={self.input_placeholder: self.test_examples, self.labels_placeholder: self.test_labels})
                new_preds = Z2_preds_new > 0
                assert(len(new_preds[0]) == len(self.test_labels[0]))
                self.print_new_test_loss_list.append(loss_new)
                new_accuracy = sum([(new_preds[0][_] == self.test_labels[0][_]) for _ in range(len(self.test_labels[0]))])/float(len(self.test_labels[0]))
                self.print_new_test_acc_list.append(new_accuracy)
                print ("new_test accuracy in current epoch %s" % str(new_accuracy))

                Z2_preds_dev, loss_dev = sess.run([self.Z2, self.loss], feed_dict={self.input_placeholder: self.dev_examples, self.labels_placeholder: self.dev_labels})
                dev_preds = Z2_preds_dev > 0
                assert(len(dev_preds[0]) == len(self.dev_labels[0]))
                self.print_dev_loss_list.append(loss_dev)
                dev_accuracy = sum([(dev_preds[0][_] == self.dev_labels[0][_]) for _ in range(len(self.dev_labels[0]))])/float(len(self.dev_labels[0]))
                self.print_dev_acc_list.append(dev_accuracy)
                print ("dev accuracy in current epoch %s" % str(dev_accuracy))
                
                #### print results ####
            file_print.write("train_loss_per_batch: "+ str(self.print_train_loss_list)+"\n")
            file_print.write("train_acc_per_epoch: " + str(self.print_train_acc_list) + "\n")
            file_print.write("dev_loss: "+ str(self.print_dev_loss_list)+ "\n")
            file_print.write("dev_acc: " + str(self.print_dev_acc_list) + "\n")
            file_print.write("new_test_loss: "+ str(self.print_new_test_loss_list)+ "\n")
            file_print.write("new_test_acc: " + str(self.print_new_test_acc_list) + "\n")


def do_train(input_path):

    try:
        train_path = open(input_path+"train", "rb")
        dev_path = open(input_path+"dev", "rb")
        test_path = open(input_path+"test", "rb")

        train_label_path = open(input_path+"train_label", "rb")
        dev_label_path = open(input_path+"dev_label", "rb")
        test_label_path = open(input_path+"test_label", "rb")


#        new_test_path = open(input_path+"new_test", "rb")

    except IOError:
        print ("Could not open file from " + input_path)
        sys.exit()

    dev = np.load(dev_path)      # (m x embedding)
    test = np.load(test_path)
    train = np.load(train_path)
    train_labels = np.load(train_label_path)   # (m x 1)
    test_labels = np.load(test_label_path)
    dev_labels = np.load(dev_label_path)

    print_files = "./print_files"
    if not os.path.isdir(print_files):
        os.makedirs(print_files)

    output_dir = print_files + "/{:%Y%m%d_%H%M%S}".format(datetime.datetime.now() + "_LM_classifier")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    file_print = open(output_dir + "/run_result.txt", "w")
    all_data = (train, dev, test)
    all_labels = (train_labels, dev_labels, test_labels)

    start = time.time()

    our_model = FeedforwardModel(data=all_data, config=Config(),
                                 labels=all_labels)
    elapsed = (time.time() - start)
    print ("BUILDING THE MODEL TOOK " + str(elapsed) + "SECS")
    our_model.train(file_print)


def main(argv):
    global input_path
    input_path = ""
    try:
        opts, args = getopt.getopt(argv,"i:",["input"])
    except getopt.GetoptError:
        print ('test.py -i')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--input_path"):
            input_path = arg

    if input_path == "":
        print ("Must enter the input_path to the directory containing train/dev/test data")
        sys.exit()

    do_train(input_path)

if __name__ == "__main__":
   main(sys.argv[1:])
