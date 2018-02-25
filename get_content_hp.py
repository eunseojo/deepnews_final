import pandas
import bs4
import ast


global count 

def write_files(m):
	dirname = "HPextract"
	if '/' in m[1]:
    	filename = './' + dirname + '/' + m[1].translate({ord(c): None for c in '/'}) + ".txt" 
    else:
    	filename = './' + dirname + '/' + m[1] + ".txt"
    with open(filename,"a+") as f:
    	count += 1
    	print(count)
        f.write(m[2])


file = pandas.read_csv("/Users/Ayush/CS230/data_huff_130K.csv")

body = file['summary']
title = file['title']
m = len(body)

count = 0 

for i in range(m):
	count += 1
	bodytext = bs4.BeautifulSoup(ast.literal_eval(body[i])['content'],"lxml").get_text()[:-185]
	write_input = ('1', title[i][0:60], bodytext)
	write_files(write_input)