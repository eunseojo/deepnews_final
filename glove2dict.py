import pickle
input_path = ""
output_path = ""


'''
@Gives each vocab token from the pretrained glove files a unique key to produce dictionary

Inputs:
1) path_to_glove: [string] path to the GloVe downloaded embeddings

Outputs (write):
1) dict (as pickle): [hash_map] dictionary of all vocab words (k: word, v: int) in the GloVE embeddings
'''
dict = {}
with open(input_path, "r") as reader:
    for r in enumerate(reader):
        word = r[1].strip().split()[0]
        dict[r[0]] = word
    pickle.dump(dict, open(output_path, "wb"))