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
    max_length = 50  # should be factor of 600 ;longest sequence to parse
    n_classes = 672 #size of vocabulary (since we're predicting at every sequence)
    dropout = 0.0
    model_option = "vanilla"   ##  "dynamic" or "vanilla"
    embed_size = 100
    hidden_size = 1000
    batch_split = int(128*(600/max_length))
    batch_size = 32 ## should be factor of 128
    n_epochs = 200
    cell = "lstm"    
    #max_grad_norm = 10.
    lr = 0.009
    clip_gradients = True
    max_grad_norm = 5.0
    end_token = 328
    #glove_matrix = "/Users/eunseo/Downloads/drive-download-20180304T214740Z-001/100d_glove_matrix.npy"
    glove_matrix = "./POSGlove/100d_glove_matrix.npy"
    path_to_aashiq = "./aashiqdict"
    max_num_batches = 32000
    dev_max_num_batches = 1000
    good_dev_directory = "./FinalData/batches/ML600BS128/good_dev"
    bad_dev_directory = "./FinalData/batches/ML600BS128/bad_dev"

    def __init__(self):
        pass

class Language_Model():
    def __init__(self,input_directory, pretrained, config):
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
        self.train_op = self.add_training_op(self.loss)
        self.aashiq = pickle.load(open(self.config.path_to_aashiq, "rb"))

        self.input_directory = input_directory
        self.num_batches, self.all_batches = self.doc_load(self.input_directory)
        self.shuffled = np.arange(self.num_batches)
        np.random.shuffle(self.shuffled)

        self.num_good_dev_batches, self.good_dev_batches = self.dev_load(self.config.good_dev_directory)
        print(self.good_dev_batches[0].shape, "shape of dev")
        self.num_bad_dev_batches, self.bad_dev_batches = self.dev_load(self.config.bad_dev_directory)

        #self.shuffled = np.random.shuffle(np.arange(self.num_batches)) #self.shuffle_batches() 

        self.print_train_acc_list = []
        self.print_good_dev_loss_list = []
        self.print_good_dev_acc_list = []
        self.print_bad_dev_loss_list = []
        self.print_bad_dev_acc_list = []
        self.print_train_loss_list = []
        self.print_sample = []

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


    def vanilla_rnn(self, cell, x, U, b_2):
        with tf.variable_scope("RNN"):
            all_steps_preds = []
            h = tf.zeros(shape=[self.config.batch_size, self.config.hidden_size], dtype=tf.float64)
            for time_step in range(self.config.max_length):
                output, h = cell(x[:, time_step], h, scope="RNN") ##x: (batch x embedding_size)
                tf.get_variable_scope().reuse_variables()
                #h_drop = tf.nn.dropout(h, keep_prob=dropout_rate)
                # print z.get_shape()
                # add prediction that's before the softmax layer
                z = tf.matmul(h, U) + b_2    #####prediction at everytime step, batch x  vocab
                all_steps_preds.append(z)
        return tf.stack(all_steps_preds)   #### time_steps x batch x vocab

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

        xavier = tf.contrib.layers.xavier_initializer()
        lstm_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size, initializer=xavier)
#        lstm_cell  = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.config.dropout, output_keep_prob=self.config.dropout)
        W_y = tf.get_variable("W_y", initializer=xavier, shape=[self.config.hidden_size, self.config.n_classes],
                        dtype=tf.float64)

        b_2 = tf.get_variable("b_2", initializer=tf.zeros(shape=[1, self.config.n_classes], dtype=tf.float64))
        #tf.get_variable_scope().reuse_variables()
        # if self.config.model_option == "dynamic":
        #     z, regu_loss = self.dynamic(cell, x, U, b_2)
        # #preds = tf.transpose(tf.pack(preds), perm=[1, 0, 2])

        # all_steps_preds = []
        # h = tf.zeros(shape=[self.config.batch_size, self.config.hidden_size], dtype=tf.float64)
        # c = tf.zeros(shape=[self.config.batch_size, self.config.hidden_size], dtype=tf.float64)
        # state = c,h
        #x = tf.split(x, self.config.max_length, axis=1)
        x = tf.unstack(x, axis=1)
        #b = tf.Print(x, [x], "shape of xadf")
        outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float64)
        #print(outputs, "outputs")
        outputs = tf.convert_to_tensor(outputs)
        #outpus: seq X batch X hidden
        #       600 x 128 x 50
        #for time_step in range(self.config.max_length):

        #   output, h = cell(x[:, time_step], scope="RNN") ##x: (batch x embedding_size)

            #h_drop = tf.nn.dropout(h, keep_prob=dropout_rate)
            # print z.get_shape()
            # add prediction that's before the softmax layer
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
        #assert(loss_batches.get_shape()[0] == self.config.batch_size)
        return batch_loss ###vector of batchsize (each elemnt is the mean loss)

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)

        if self.config.clip_gradients:
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            optimizer = optimizer.apply_gradients(zip(gradients, variables))
            return optimizer
        else:  
            return optimizer.minimize(loss)

    def doc_load(self, dir_path):
        files = os.listdir(dir_path)
        paths = list(map(lambda x: os.path.join(dir_path, x),files))       ###each of these files is a batch of text (batch x max length) :this needs to be pre-randomized
        paths.sort()
        all_batches = [np.load(open(p, "rb")).reshape((-1,self.config.max_length)) for p in paths]
        batches_reduced = []
        for i in range(len(all_batches)):
            batches_reduced += np.split(all_batches[i], self.config.batch_split/self.config.batch_size)
        #all_batches = [all_batches[i][0,:self.config.max_length].reshape((self.config.batch_size,self.config.max_length)) for i in range(10)]
        print("num of batches::: ", len(all_batches))
        print(all_batches[0].shape)
        batches_reduced = batches_reduced[:self.config.max_num_batches]
        return len(batches_reduced), batches_reduced

    def dev_load(self, dir_path):
        files = os.listdir(dir_path)
        paths = list(map(lambda x: os.path.join(dir_path, x),files))       ###each of these files is a batch of text (batch x max length) :this needs to be pre-randomized
        paths.sort()
        all_batches = [np.load(open(p, "rb")).reshape((-1,self.config.max_length)) for p in paths]
        batches_reduced = []
        for i in range(len(all_batches)):
            batches_reduced += np.split(all_batches[i], self.config.batch_split/self.config.batch_size)
        #all_batches = [all_batches[i][0,:self.config.max_length].reshape((self.config.batch_size,self.config.max_length)) for i in range(10)]
        batches_reduced = batches_reduced[:self.config.dev_max_num_batches]
        return len(batches_reduced), batches_reduced


    def shuffle_batches(self):
        shuffled_indices = np.random.shuffle(np.arange(self.num_batches))
        #self.all_batches = [self.all_batches[i] for i in shuffled_indices]
        return shuffled_indices

    def yield_docs(self, dir_path):
        '''
        @This function yields text dumps of dev/train sets from given intput1)dir_path
        '''
        files = os.listdir(dir_path)
        paths = list(map(lambda x: os.path.join(dir_path, x),files))       ###each of these files is a batch of text (batch x max length) :this needs to be pre-randomized
        for i in range(len(paths)):
            train_set = np.load(open(paths[i],"rb"))
            yield train_set   #### (batch x max_length) train batches

    def sample(self, preds):
        '''this function prints the predicted sequences'''
        shape = preds.shape  # time * batch
        flat_preds = preds.flatten()
        converted = [self.aashiq[pred] for pred in flat_preds]
        return np.reshape(converted, shape) # return time * batch

    def calculate_dev(self, sess,fileprint):

        sum_good_loss = 0.0
        sum_bad_loss = 0.0
        sum_good_acc = 0.0
        sum_bad_acc = 0.0

        for i in range(self.config.dev_max_num_batches):
            gD = self.good_dev_batches[i]
            bD = self.bad_dev_batches[i]
            gL = copy.deepcopy(gD)
            reduced = gL[:,1:]
            gL = np.hstack((reduced, np.full((gL.shape[0],1), self.config.end_token)))   #### remember to replace end token with the right token
            bL = copy.deepcopy(bD)
            reduced = bL[:,1:]
            bL = np.hstack((reduced, np.full((bL.shape[0],1), self.config.end_token)))   #### remember to replace end token with the right token

            good_preds, good_loss = sess.run([self.pred, self.loss], feed_dict={self.input_placeholder: gD, self.labels_placeholder: gL})
            bad_preds, bad_loss = sess.run([self.pred, self.loss], feed_dict={self.input_placeholder: bD, self.labels_placeholder: bL})
            sum_good_loss += good_loss/self.config.dev_max_num_batches
            sum_bad_loss += bad_loss/self.config.dev_max_num_batches
            assert(len(good_preds[0]) == len(gL.T[0]))
            good_accuracy = sum(sum((np.array(np.argmax(good_preds,axis=2))==np.array(gL.T))))/(self.config.max_length * self.config.batch_size)
            bad_accuracy =  sum(sum((np.array(np.argmax(bad_preds,axis=2))==np.array(bL.T))))/(self.config.max_length * self.config.batch_size)
            sum_good_acc += good_accuracy/self.config.dev_max_num_batches
            sum_bad_acc += bad_accuracy/self.config.dev_max_num_batches

        self.print_good_dev_loss_list.append(sum_good_loss)
        self.print_bad_dev_loss_list.append(sum_bad_loss)
        self.print_good_dev_acc_list.append(sum_good_acc)
        self.print_bad_dev_acc_list.append(sum_bad_acc)
        fileprint.write("good_dev_loss: "+ str(self.print_good_dev_loss_list)+ "\n")
        fileprint.write("bad_dev_loss: " + str(self.print_bad_dev_loss_list) + "\n")
        fileprint.write("good_dev_acc: "+ str(self.print_good_dev_acc_list)+ "\n")
        fileprint.write("bad_dev_acc: " + str(self.print_bad_dev_acc_list) + "\n")
        return sum_good_loss, sum_bad_loss, sum_good_acc, sum_bad_acc


    def train(self, file_print, output_dir):
        self.print_helper(file_print)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


        ### HAVE TO SAVE WEIGHTS NOW!!!!!!#####
        weights_dir = output_dir + "/weights"
        if not os.path.isdir(weights_dir):
            os.makedirs(weights_dir)

        with tf.Session() as session:
            session.run(init)
            accuracy_past_epoch = 0            
            gl, bl, ga, ba = 0,0,0,0
            for i in range(self.config.n_epochs):
                batch_count = 0
                accuracy_epoch_list = []
                for j in range(self.num_batches):   ### one full epoch iteration through the doc generator
                    one_batch_X = self.all_batches[self.shuffled[j]] # batch * max_step
                    print(one_batch_X.shape)
                    one_batch_Y = copy.deepcopy(one_batch_X)
                    reduced = one_batch_Y[:,1:]
                    one_batch_Y = np.hstack((reduced, np.full((one_batch_Y.shape[0],1), self.config.end_token)))   #### remember to replace end token with the right token
                    # ^ shape : batch * max_step
                    print ("batch no. : " + str(batch_count), "    epoch no.: ", str(i))

                    #$one_batch_Y, one_batch_X \in (batch x max_length)$

                    _, batch_loss, preds = session.run([self.train_op, self.loss, self.pred], feed_dict={
                                                                                            self.input_placeholder: one_batch_X,
                                                                                      self.labels_placeholder: one_batch_Y})
                    print ("batch_loss")
                    print (batch_loss)
                    # print ("session ran")
                    pred = np.argmax(preds, axis=2)
                    print(pred)
                    print(self.sample(pred)[:,0])
                    sum_lengths = self.config.batch_size * self.config.max_length
                    sum_corr = np.sum(np.array(pred.T) == np.array(one_batch_Y))
                    # batch_average = batch_loss
                    accuracy_per_batch = float(sum_corr)/sum_lengths
                    
                    print("accuracy past epoch = " , accuracy_past_epoch)
                    
                    print("past good/bad dev loss: %s" % str(gl)+", "+str(bl))
                    print("past good/bad dev acc: %s" % str(ga) +", "+str(ba))
                    accuracy_epoch_list.append(accuracy_per_batch)
                    # print ("debug: ", batch_average, " batch_average")
                    self.print_train_loss_list.append(batch_loss)
                    print ("batch accuracy is ", accuracy_per_batch)
                    self.print_train_acc_list.append(accuracy_per_batch)
                    print ("batch average loss is ", batch_loss)
                    self.print_sample.append(self.sample(pred)[0])
                    batch_count += 1

                gl, bl, ga, ba = self.calculate_dev(session,file_print)
                accuracy_past_epoch = sum(accuracy_epoch_list)/len(accuracy_epoch_list)
                print("avg epoch accuracy = ", accuracy_past_epoch)
                print("good/bad dev loss: %s" % str(gl)+", "+str(bl))
                print("good/bad dev acc: %s" % str(ga) +", "+str(ba))
                np.random.shuffle(self.shuffled)
                file_print.write("average_train_loss: " + str(self.print_train_loss_list)+ "\n")
                file_print.write("average_train_acc: " + str(self.print_train_acc_list) + "\n")
                file_print.write("sample: " + str(self.print_sample) + "\n")
                if i % 2 == 0:
                    saver.save(session, os.path.join(weights_dir, str(i)))
                #print ("weights saved at epoch", str(i))
            file_print.close()


def do_train(input_path):
    embeddings = np.load(Config.glove_matrix)
    print_files = "./print_files"
    if not os.path.isdir(print_files):
        os.makedirs(print_files)
    output_dir = print_files + "/{:%Y%m%d_%H%M%S}".format(datetime.datetime.now()) + "_" +str(Config.max_length)+"_"+str(Config.batch_size)+"_"+str(Config.lr)+"_"+str(Config.max_num_batches)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    #print(embeddings)
    file_print = open(output_dir + "/run_result.txt", "w")
    # This takes a long time because it's building the graph
    start = time.time()
    print("bulding model")
    our_model = Language_Model(input_directory=input_path, pretrained=embeddings, config=Config)
    elapsed = (time.time() - start)
    print ("BUILDING THE MODEL TOOK " + str(elapsed) + "SECS")
    our_model.train(file_print, output_dir)

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
