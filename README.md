# deepnews_final
deepnews project finalized scripts

Description of Files (in sequence)

Data Extraction: 
1) get_content.py & get_content_hp.py :: Files for extracting raw text for 

Processing:
1) process_small2.py :: File to splitting data to Train, Dev, Test sets (input: directories of low/high quality news; output: directories of train, dev, test) 
2) word2int_cbow.py :: File for converting text to integers for embedding look up and taking average over words for cbow (intput: directories from process_smal2.py output saved as pickles)
3) SamplePOS.py :: generates sample POS tagged text

Models & Testing: 
1) feedforward_cbow_prints.py :: File for running the CBOW feedforward model (input, 100 dimensional averaged vectors per doc; output: generates a print_files directories with logs)
2) centroids.py :: File to binary classify based on averaged centroids as comparison (input: pickles from word2int_cbow.py)

Description:
1) plot_dev_cost.py :: File for plotting charts
