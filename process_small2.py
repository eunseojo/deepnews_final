import os
import random
from shutil import copyfile
import pickle
import sys

## usage python process_small2.py dir/path/to/quality  dir/path/to/low/quality output/dir/path train_percentage
train_percent=float(sys.argv[4])
#_dev=200
#_test=200

# dir_quality = "../cs230_data/4k_trial/quality/"
# dir_low_quality = "../cs230_data/4k_trial/lowQuality/"

# output_dir = "./test_sample"

dir_quality = sys.argv[1]
dir_low_quality = sys.argv[2]

output_dir = sys.argv[3]

def return_paths_to_files(dir_path):
	return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f[-3:] == 'txt'] # only include txt files in the list of files

def shuffle_paths(paths):
    random.shuffle(paths)
    return paths

def add_tuples(paths, y_hat):
    return [(y_hat, path) for path in paths]

# def split(y_hat, articles):
#     train = (y_hat[:_train], articles[:_train])
#     dev = (y_hat[_train:_train+_dev], articles[_train:_train+_dev])
#     test = (y_hat[-_test:], articles[-_test:])
#     return train, dev, test

def split_tuples(shuff):
    assert(train_percent < 100)
    _train = int((train_percent/100)*len(shuff))
    _dev = int((len(shuff) - _train)/2)
    train = shuff[ : _train]
    dev = shuff[_train : _train + _dev]
    test = shuff[_train + _dev : ]
    print("Training examples: " , _train)
    print("Dev examples: " , _dev)
    print("Test examples: ", len(shuff) - (_train + _dev))
    return train, dev, test

def write_files(train,dev,test):
    sections = [("train", train), ("dev", dev), ("test", test)]

    for sec_name, sec_tup in sections:
        dir_name = os.path.join(output_dir,sec_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        for num, tup in enumerate(sec_tup): # tup is (yhat, article_path)
            copyfile(tup[1], os.path.join(dir_name,str(num) + "_" + tup[1].split("/")[-1]))
            with open(os.path.join(dir_name,str(num) + "_" + tup[1].split("/")[-1]),"a") as tfile:
                tfile.write("\n training_label0610 %s \n" % str(tup[0]))
            
#        pickle.dump(sec_tup[0],open(os.path.join(output_dir, "y_hat_" + str(sec_name)), "wb"))

if __name__ == "__main__":
    both_articles = add_tuples(return_paths_to_files(dir_quality), 1) + add_tuples(return_paths_to_files(dir_low_quality), 0)
    shuffled = shuffle_paths(both_articles)
    #y_hat, articles = map(list, zip(*shuffled))
    train, dev, test = split_tuples(shuffled)
    #print(train[0])
    write_files(train, dev, test)
    
