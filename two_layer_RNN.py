import sys
import getopt
import pickle
import os
import time, datetime
import tensorflow as tf
import numpy as np
import copy






'''
Input for class LM has to be "intted" text & glove matrix (from glove2matrix.py)


This language model only TRAINS. We want to train our weights (will test on dev later)
We need a function that 1) ints all the documents and then 2) randomizes the divvied catted document

the files that you need to use with this are glove2matrix.py & the int series(process_small_....py ; word2int.py) & pickle2numpy.py



*** sampling for fun
*** backprop to (x) word vectors (OO)
*** normalize batches
*** hyper parameter tuning

'''


class Config:
    max_length = 25  # longest sequence to parse
    n_classes = 1 #size of vocabulary (since we're predicting at every sequence)
    dropout = 0.0
    ##model_option = "vanilla"   ##  "dynamic" or "vanilla"
    embed_size = 100
    hidden_size = 3000
    batch_size = 32 
    n_epochs = 100
    dev_size = 5000
    dev_max_num_batches = 1
    cell = "lstm"
    #max_grad_norm = 10.
    lr = 0.0001
    clip_gradients = False  ## check why false?
    max_grad_norm = 0
    ##end_token = 5
    #glove_matrix = "/Users/eunseo/Downloads/drive-download-20180304T214740Z-001/100d_glove_matrix.npy"
    glove_matrix = "../POSGlove/100d_glove_matrix.npy"
    ##path_to_aashiq = "./aashiqdict"

    def __init__(self):
        pass

class RNN_classifier():
    def __init__(self,input_directory, pretrained, config):
        self.config = config
        self.pretrained_embeddings = tf.get_variable(initializer = pretrained, name = "pretrained_embeddings")   ###pretrained embeddings (vocab x embedding_size)

        #placeholders
        self.input_placeholder = tf.placeholder(shape=[self.config.max_length, None], dtype=tf.int32)   ### currently working with fixed length
        self.labels_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float64)                       ### shape = (m x 1)

        #regularization
        #self.regularizer = tf.contrib.layers.l2_regularizer(scale = 0.03)
        #self.l1 = tf.contrib.layers.l1_regularizer(scale = 0.03)

        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        ## self.aashiq = pickle.load(open(self.config.path_to_aashiq, "rb"))


        self.num_batches, self.all_batches = self.doc_load(os.path.join(input_directory, 'pi'), self.config.batch_size)
        _, self.all_y = self.doc_load(os.path.join(input_directory, 'y'), self.config.batch_size)

        _, self.dev_batches = self.doc_load(os.path.join(input_directory, 'dev_pi'),  self.config.dev_size)
        _, self.dev_batches_y = self.doc_load(os.path.join(input_directory, 'dev_y'), self.config.dev_size)


        self.shuffled = np.arange(self.num_batches)
        np.random.shuffle(self.shuffled)
        #self.shuffled = np.random.shuffle(np.arange(self.num_batches)) #self.shuffle_batches()

        self.print_train_acc_list = []
        #self.print_dev_loss_list = []
        #self.print_dev_acc_list = []
        self.print_train_loss_list = []
        ##self.print_sample = []
        self.print_dev_loss_list = []
        self.print_dev_acc_list = []

    def print_helper(self, file):
        '''
        @This function prints metadata about the class to input1) file
        '''

        file.write("max_length: " + str(self.config.max_length) + "\n")
        file.write("embed_size: " + str(self.config.embed_size) + "\n")
        file.write("classes: " + str(self.config.n_classes) + "\n")
        file.write("hidden_size: " + str(self.config.hidden_size) + "\n")
        file.write("n_epochs: " + str(self.config.n_epochs) + "\n")
        file.write("learn_rate: "+ str(self.config.lr) + "\n")
        file.write("batch_size: " + str(self.config.batch_size)+ "\n")
        file.write("layers: " + str(1) + "\n")
        file.write("cell type: " + str(self.config.cell) + "\n")
        file.write("clip_gradients: " + str(self.config.clip_gradients) + "\n")
        file.flush()

    def add_prediction_op(self):
        x = self.add_embedding()    ### batch x max_size(time_step) x embedding_size(num_features)
#        with tf.variable_scope("RNN"):

        lstm_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
#        lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell1]*2)
        xavier = tf.contrib.layers.xavier_initializer()

        #W_1 = tf.get_variable("W_1", initializer = xavier, shape = [self.config.hidden_size, self.config.hidden_size])
        #b_1 = tf.get_variable("b_1", initializer=tf.zeros(shape[1,self.config.hidden_size],dtype=tf.float64)
        W_y = tf.get_variable("W_y", initializer=xavier, shape=[self.config.hidden_size, self.config.n_classes],
                        dtype=tf.float64)

        b_2 = tf.get_variable("b_2", initializer=tf.zeros(shape=[1, self.config.n_classes], dtype=tf.float64))

        

        index = tf.get_variable("index", initializer = self.config.max_length - 1 , dtype=tf.int32, trainable = False)
        #tf.get_variable_scope().reuse_variables()
        # if self.config.model_option == "dynamic":
        #     z, regu_loss = self.dynamic(cell, x, U, b_2)
        # #preds = tf.transpose(tf.pack(preds), perm=[1, 0, 2])

        all_steps_preds = []
        ## h = tf.zeros(shape=[self.config.batch_size, self.config.hidden_size], dtype=tf.float64)
        ## c = tf.zeros(shape=[self.config.batch_size, self.config.hidden_size], dtype=tf.float64)
        ## state = c,h
        #x = tf.split(x, self.config.max_length, axis=1)
        x = tf.unstack(x, axis=1)
        #b = tf.Print(x, [x], "shape of xadf")
        outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float64)
        #print(outputs, "outputs")
        last_output = outputs[-1]
        outputs = tf.convert_to_tensor(last_output) ## batch x hidden
        #outpus: seq X batch X hidden
        #       600 x 128 x 50
        #for time_step in range(self.config.max_length):

        #   output, h = cell(x[:, time_step], scope="RNN") ##x: (batch x embedding_size)

            #h_drop = tf.nn.dropout(h, keep_prob=dropout_rate)
            # print z.get_shape()
            # add prediction that's before the softmax layer
        ## time_step = self.config.max_length
        ## batch, hidden = self.config.batch_size, self.config.hidden_size

        #W_y = hidden, vocab
        #
        #output_a = tf.gather(outputs, index, axis = 0)
        ## inner_reshape = tf.reshape(outputs, [self.config.max_length * self.config.batch_size, self.config.hidden_size]) #76800 , 50
        z = tf.matmul(outputs, W_y) + b_2 #50, 663 >  50918400
        #batch_size * 1
        return z

    def add_loss_op(self, pred):
        #pred : time_steps x batch x vocab
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels_placeholder, logits = pred)
        batch_loss = tf.reduce_mean(losses)
        #assert(loss_batches.get_shape()[0] == self.config.batch_size)
        return batch_loss ###vector of batchsize (each elemnt is the mean loss)

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        #gradients = optimizer.compute_gradients(loss)
        # gradients_only = [grad[0] for grad in gradients]
        # variables = [grad[1] for grad in gradients]

        # if self.config.clip_gradients:
        #     gradients_only, _ = tf.clip_by_global_norm(gradients_only, self.config.max_grad_norm)
        #     # combine variables and gradients together
        #     new_gradients = []
        #     for i in range(len(gradients)):
        #         new_gradients.append((gradients_only[i], variables[i]))
        #     gradients = new_gradients
        #
        # print ("before grad norm")
        # self.grad_norm = tf.global_norm(gradients_only)
        # train_op = optimizer.apply_gradients(gradients)
        return optimizer

    def add_embedding(self):
        flattened = tf.reshape(self.input_placeholder, [-1])
        look_up = tf.nn.embedding_lookup(params=self.pretrained_embeddings, ids=flattened)
        ##look_up = tf.Print(look_up, [self.pretrained_embeddings[0,:]], "embedding")        ## this checks if the embeding is updating
        embeddings = tf.reshape(look_up, [-1, self.config.max_length, self.config.embed_size])
        #### THIS DEF WORKS; WE CHECKED ON COLAB --ES; Shank
        #### batch x max_len x embed_size = batch x 600 x 100
        return embeddings

    def doc_load(self, dir_path, size):
        files = sorted(os.listdir(dir_path))
        paths = list(map(lambda x: os.path.join(dir_path, x),files))       ###each of these files is a batch of text (batch x max length) :this needs to be pre-randomized
        all_batches = [np.load(open(p, "rb")).reshape((size, -1)) for p in paths]
        return len(all_batches), all_batches






    def calculate_dev(self, sess,fileprint):
        
        print("Before dev")
        dev_loss = 0.0
        dev_acc = 0.0

        for i in range(self.config.dev_max_num_batches):
            X = self.dev_batches[i]
            Y = self.dev_batches_y[i]

            pred, loss = sess.run([self.pred, self.loss], feed_dict={self.input_placeholder: X.T, self.labels_placeholder: Y})

            dev_loss += loss/self.config.dev_max_num_batches
            pred = np.asarray(pred > 0)
            accuracy = np.mean(1 - np.abs(Y - pred))

            dev_acc += accuracy/self.config.dev_max_num_batches

        self.print_dev_loss_list.append(dev_loss)
        print("dev loss: %s" % str(dev_loss))
        self.print_dev_acc_list.append(dev_acc)
        print ("dev accuracy: %s" % str(dev_acc))
        fileprint.write("dev_loss: "+ str(self.print_dev_loss_list)+ "\n")
        fileprint.write("dev_acc: "+ str(self.print_dev_acc_list)+ "\n")
        self.print_dev_loss_list = []
        self.print_dev_acc_list = []



    def train(self, output_dir, file_print):
        self.print_helper(file_print)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


        ### HAVE TO SAVE WEIGHTS NOW!!!!!!#####
        weights_dir = output_dir + "/weights"
        if not os.path.isdir(weights_dir):
            os.makedirs(weights_dir)

        with tf.Session() as session:
            session.run(init)
            for i in range(self.config.n_epochs):
                batch_count = 0
                for j in range(self.num_batches):   ### one full epoch iteration through the doc generator
                    one_batch_X = self.all_batches[self.shuffled[j]]
                    #### print(one_batch_X.shape)

                    one_batch_Y = self.all_y[self.shuffled[j]] # label

                    ## one_batch_Y = copy.deepcopy(one_batch_X)
                    ## reduced = one_batch_Y[:,1:]
                    ## one_batch_Y = np.hstack((reduced, np.full((one_batch_Y.shape[0],1), self.config.end_token)))   #### remember to replace end token with the right token
                    #### print ("batch no. : " + str(batch_count), "    epoch no.: ", str(i))

                    #$one_batch_Y, one_batch_X \in (batch x max_length)$

                    _, batch_loss, pred = session.run([self.train_op, self.loss, self.pred], feed_dict={
                                                                                            self.input_placeholder: one_batch_X.T,
                                                                                      self.labels_placeholder: one_batch_Y})
                    #print ("batch_loss")
                    #### print (batch_loss)

                    pred = np.asarray(pred > 0)
                    #print('prediction:', pred)
                    #print('Y actual:', one_batch_Y)
                    # print ("session ran")
                    ## pred = np.argmax(preds, axis=2)
                    ## print(pred)
                    ## print(self.sample(pred)[0])
                    ## sum_lengths = self.config.batch_size * self.config.max_length
                    ## sum_corr = sum(sum(np.array(pred.T) == np.array(one_batch_Y)))
                    # batch_average = batch_loss
                    #print('shit:', np.abs(one_batch_Y - pred))
                    accuracy_per_batch = np.mean(1 - np.abs(one_batch_Y - pred))
                    # print ("debug: ", batch_average, " batch_average")
                    self.print_train_loss_list.append(batch_loss)
                    #### print ("batch average loss is ", batch_loss)

                    self.print_train_acc_list.append(accuracy_per_batch)
                    #### print ("batch accuracy is ", accuracy_per_batch)
                    ##self.print_sample.append(self.sample(pred)[0])
                    batch_count += 1

                self.calculate_dev(session,file_print)

                np.random.shuffle(self.shuffled)
                loss_epoch_mean = np.mean(np.asarray(self.print_train_loss_list))
                #### print('average epoch loss: ', loss_epoch_mean)
                file_print.write("average_train_loss: "+ str(self.print_train_loss_list)+ "\n")
                file_print.write("average_epoch_loss: "+ str(loss_epoch_mean)+"\n")
                file_print.write("average_train_acc: " + str(self.print_train_acc_list) + "\n")
                ##file_print.write("sample: " + str(self.print_sample) + "\n")

                
                if (i % 4 == 0) :
                    saver.save(session, os.path.join(weights_dir, str(i)))
                    print ("weights saved at epoch", str(i))

                self.print_train_loss_list = []
                self.print_train_acc_list = []

            file_print.close()


def do_train(input_path):
    embeddings = np.load(Config.glove_matrix)
    print_files = "./print_files"
    if not os.path.isdir(print_files):
        os.makedirs(print_files)
    output_dir = print_files + "/{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    #print(embeddings)
    file_print = open(output_dir + "/run_result.txt", "w")
    # This takes a long time because it's building the graph
    start = time.time()
    print("bulding model")
    our_model = RNN_classifier(input_directory = input_path, pretrained = embeddings, config = Config)
    elapsed = (time.time() - start)
    print ("BUILDING THE MODEL TOOK " + str(elapsed) + "SECS")
    our_model.train(output_dir, file_print)

def main(argv):
    input_path = ""
    try:
        opts, args = getopt.getopt(argv,"i:",["input"])
    except getopt.GetoptError:
        print ('test.py -i input_file_path')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i"):
            input_path = arg
        ##other arguments

    if input_path == "":
        print ("test.py -i input_file_path")
        sys.exit()
    do_train(input_path)

if __name__ == "__main__":
    main(sys.argv[1:])

