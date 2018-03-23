import pickle
import numpy as np
import os



### generates (batch x max_length) matrix files (saved .npy)

pickle_path = "pickles/train"
save_path = "split_pickles_LM"
#ENDOFDOC = 30394 #random int

if not os.path.isdir(save_path):
    os.makedirs(save_path)

batch_size = 64
pi = pickle.load(open(pickle_path, "rb"))


batch_num = int(len(np.array(pi[0]))/batch_size)
end = batch_num * batch_size
print("dividing into ", str(batch_num), " batches of size ", str(batch_size))
split_arrays = np.split(np.array(pi[0][:end]), batch_num) ##split arrays into batch_size (discard leftovers)
for e, arra in enumerate(split_arrays):
    np.save(os.path.join(save_path, str(e)), arra)
