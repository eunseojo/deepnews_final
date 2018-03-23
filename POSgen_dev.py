## python POSgen_batch.py input/pickle/path output/save/path batch_size
#python POSgen_batch.py ../FinalData/Dev_good.p ../FinalData/batches/good_dev/ 128

def gen_batch(pickle_path, save_path, batch_size):
    import pickle
    import numpy as np
    import os

    ### generates (batch_size x max_length) matrix files (saved .npy)

    save_path_pi = os.path.join(save_path, 'dev_pi')
    save_path_y = os.path.join(save_path, 'dev_y')

    if not os.path.isdir(save_path_pi):
        os.makedirs(save_path_pi)
    if not os.path.isdir(save_path_y):
        os.makedirs(save_path_y)

    max_size = 25
    batch_size = int(batch_size)

    pi_good = pickle.load(open(os.path.join(pickle_path, 'good.p'), "rb"))
    pi_bad = pickle.load(open(os.path.join(pickle_path, 'bad.p'), "rb"))

    batch_num = min(int(len(pi_good)/batch_size/max_size), int(len(pi_bad)/batch_size/max_size))

    end = batch_num * batch_size * max_size

    pi_good = pi_good[:end]
    pi_bad = pi_bad[:end]

    pi = np.expand_dims(np.hstack((pi_good, pi_bad)), axis = 1)
    y = np.hstack((np.ones((1, batch_num * batch_size),dtype = int), np.zeros((1, batch_num * batch_size), dtype = int)))
    batch_num = 2 * batch_num

    pi = pi.reshape(batch_num * batch_size, max_size)
    shuffled = np.arange(batch_num * batch_size)
    np.random.shuffle(shuffled)
    pi = pi[shuffled, :]
    print(y.shape)
    y = y[0, shuffled]

    print("dividing into ", str(batch_num), " batches of size ", str(batch_size))
    split_arrays_pi = np.split(pi, batch_num) ##split arrays into batch_size (discard leftovers)
    split_arrays_y = np.split(y, batch_num) ##split arrays into batch_size (discard leftovers)

    for index in range(10):#len(split_arrays_pi)):
        np.save(os.path.join(save_path_pi, str(index)), split_arrays_pi[index])
        np.save(os.path.join(save_path_y, str(index)), split_arrays_y[index])

if __name__ == '__main__':
    import sys
    gen_batch(sys.argv[1],sys.argv[2],sys.argv[3])
