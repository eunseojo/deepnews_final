import sys
import getopt
import os
import re
import random
import pickle
import argparse
import getopt
import codecs
import nltk
from operator import add

#usage python word2int_cbow.py -i path/to/examples/dir -o pickle/output/path ## note path to glove is hardcoded

RANDWORD = "randomwordnotfound"
PADDING = "paddingtoken"

# class Config:

#     start_date = 1861
#     end_date = 1983
#     bucket_size = 10
#     num_buckets = (end_date-start_date)/bucket_size + 1
#     max_length = 600


def vocab_dict_glove(path_to_glove):
    '''
    @Gives each vocab token from the pretrained glove files a unique key to produce dictionary

    Inputs: 
    1) path_to_glove: [string] path to the GloVe downloaded embeddings

    Outputs:
    1) dict1: [hash_map] dictionary of all vocab words (k: word, v: list of float values) in the GloVE embeddings
    2) total_v: [int] size of total vocab 
    '''
    dict1 = {}
    with open(path_to_glove, "r") as reader:
        for line in reader:
            word_vals = line.strip().split()
            dict1[word_vals[0]] = list(map(float, word_vals[1:]))
    total_v = len(dict1.keys())
    return dict1, total_v


def get_cwob_score_labels(dict1, old_dir):

    files = [fi for fi in os.listdir(old_dir) if (fi[0] != "." and fi[-3:] == 'txt')] ## does not give ".."
    paths = map(lambda x: os.path.join(old_dir, x), files)


    all_x = [] # list of lists of tokens
    yhat_list = [] # list of y labels for all the files
    for f in paths: #iterate through every FILE:f
        with open(f, "r") as reader:
            tokenized = nltk.tokenize.word_tokenize(reader.read().lower())
            # get yhat (the y label of the document) which is at the end
            yhat_tup = tokenized[-2:]
            assert(yhat_tup[0] == 'training_label0610')
            assert(yhat_tup[1] == '0' or yhat_tup[1] == '1')
            tokenized = tokenized[:-2]

            token_sum = [0]*len(dict1['a'])
            num_words = 0
            for x in tokenized:
                if x in dict1:
                    num_words += 1
                    token_sum = list(map(add, dict1[x], token_sum))
            if num_words != 0:
                token_avg = [_ /float(num_words) for _ in token_sum]
                all_x.append(token_avg)
                yhat_list.append(yhat_tup[1])                
            
    #truncated_x, lengths = truncate_or_pad(all_x, dict1[PADDING])
    assert(len(all_x) == len(yhat_list))
    return all_x, list(map(lambda _: len(_),all_x)), yhat_list 



def cbow_glove(root_path, pretrained_vocab_path, pickle_path):
    '''
    @Main function to process input data. Tokens are assigned pretrained word embeddings and pickled.

    Inputs:
    1) root_path: [string] path to unprocessed input data
    2) pretrained_vocab_path: [string] path to downloaded pretrained embeddings
    3) pickle_path: [string] path to where the preprocessed data will be saved (pickled)

    pickles the following data to pickle path per set-type(test,dev,train)
    1) truncated_X:  [list[list[int]] all words of docs as ints
    2) lengths:      [list[int]] vectors of lengths of docs before padding
    3) dict:         [hashmap] vocab-key to int-value dictionary
    '''

    dict1, size_of_vocab = vocab_dict_glove(pretrained_vocab_path)
    print("size of vocabulary: " , size_of_vocab)
    for set_type in ["test","dev","train"]:
        all_x, lengths, yhat_list = get_cwob_score_labels(dict1, os.path.join(root_path, set_type))
        print('%s examples pickled: ' % (set_type), len(all_x))
        f = open(os.path.join(pickle_path, set_type), "wb")
        #print(all_x[:10],lengths[:10],yhat_list[:10])
        pickle.dump((all_x, lengths,yhat_list), f)
        # truncated_x has dimensions:
        #    num_docs x Config.max_length x Vocab
    print ("Finished preprocessing!")


def main(argv):

    '''
    @Processes the arguments to pass to main function. 

    Inputs:
    1) input_dir: path to where the input files are saved + (/test, /train, /dev sub-directories)
    2) pickled_output_dir: path to where the pickled preprocessed files will be saved

    Hard-coded:
    1) PRETRAINED_VOCAB_PATH: path to where the glove or other pretrained vectors are saved

    '''
    input_dir = ""
    pickled_root_path = "pickles"
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["help","input", "output"])
    except getopt.GetoptError:
        print ('test.py [-h|help] [-i|input_dir] [-o|pickled_output_dir]')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('word2int.py [-h|--help] [-i|-input_dir] [-o|--pickled_output_dir]')
            sys.exit()
        elif opt in ("-i", "--input_dir"):
            input_dir = arg
        elif opt in ("-o", "--output_dir"):
            pickled_root_path = arg

    if input_dir == "" or pickled_root_path == "":
        print ('word2int.py [-h|--help] [-i|-input_dir] [-o|--pickled_output_dir] check')
        sys.exit(2)
    PRETRAINED_VOCAB_PATH = "../glove.6B/glove.6B.100d.txt"     #path to pretrained word embeddings
    if not os.path.isdir(input_dir):
        print ("Error: input dir ", input_dir, " does not exist")
        sys.exit(2)

    if not os.path.isdir(pickled_root_path):
        os.makedirs(pickled_root_path)
    cbow_glove(input_dir, PRETRAINED_VOCAB_PATH, pickled_root_path)

if __name__ == "__main__":
    main(sys.argv[1:])

    
