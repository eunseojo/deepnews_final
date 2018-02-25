import numpy as np
import sys
import pickle
import getopt
def do_clusters(input_path):
    try:
        train_path = open(input_path+"train", "rb")
        dev_path = open(input_path+"dev", "rb")
        test_path = open(input_path+"test", "rb")

    except IOError:
        print ("Could not open file from " + input_path)
        sys.exit()

    dev = pickle.load(dev_path)              #list of list (len3); 1st is list of list of ints; 2nd length of docs; 3rd Glove dictionary
    dev_np = np.array(dev[0])
    test = pickle.load(test_path)
    train = pickle.load(train_path)
    train_np = np.array(train[0])
    test_np = np.array(test[0])
    
    y_labels = []
    y_labels.append(list(map(int,train[2])))
    y_labels.append(list(map(int,dev[2])))
    y_labels.append(list(map(int,test[2])))
    
    # get centroids
    centroid1 = np.zeros((1,len(dev[0][0]))) # quality
    print((centroid1.shape[1]))
    centroid2 = np.zeros((1,len(dev[0][0]))) # low quality
    for i in range(len(y_labels[0])):
        if y_labels[0][i] == 1:
            centroid1 = centroid1 + train_np[i]
        else:
            centroid2 = centroid2 + train_np[i]
    num_train_1s = sum(y_labels[0])
    num_train_2s = len(y_labels[0]) - num_train_1s
    centroid1 = centroid1 / float(num_train_1s)
    centroid2 = centroid2 / float(num_train_2s)

    # predictions:
    train_preds = []
    for i in range(train_np.shape[0]):
        dis_cent1 = np.linalg.norm(centroid1-train_np[i])
        dis_cent2 = np.linalg.norm(centroid2-train_np[i])
        if dis_cent2 > dis_cent1:
            train_preds.append(1)
        else:
            train_preds.append(0)
    tr_accu = sum([y_labels[0][_] == train_preds[_] for _ in range(len(y_labels[0]))])/float(len(y_labels[0]))
    print("centroid classification accuracy on training set: ",tr_accu)

    dev_preds = []
    for i in range(dev_np.shape[0]):
        dis_cent1 = np.linalg.norm(centroid1-dev_np[i])
        dis_cent2 = np.linalg.norm(centroid2-dev_np[i])
        if dis_cent2 > dis_cent1:
            dev_preds.append(1)
        else:
            dev_preds.append(0)
    dev_accu = sum([y_labels[1][_] == dev_preds[_] for _ in range(len(y_labels[1]))])/float(len(y_labels[1]))
    print(len(dev_preds))
    print("centroid classification accuracy on dev set: ",dev_accu)

    test_preds = []
    for i in range(test_np.shape[0]):
        dis_cent1 = np.linalg.norm(centroid1-test_np[i])
        dis_cent2 = np.linalg.norm(centroid2-test_np[i])
        if dis_cent2 > dis_cent1:
            test_preds.append(1)
        else:
            test_preds.append(0)
    test_accu = sum([y_labels[2][_] == test_preds[_] for _ in range(len(y_labels[2]))])/float(len(y_labels[2]))
    print(len(test_preds))
    print("centroid classification accuracy on test set: ",test_accu)

    

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

    do_clusters(input_path)

if __name__ == "__main__":
   main(sys.argv[1:])
