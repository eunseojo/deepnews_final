import sys
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import bs4

dirname = "ft_processed_all"
def write_files(m):
    if '/' in m[1]:
        filename = './' + dirname + '/' + m[1].translate({ord(c): None for c in '/'}) + ".txt" 
    else:
        filename = './' + dirname + '/' + m[1] + ".txt"
    with open(filename,"a+") as f:
        f.write(m[2].encode('utf8'))
    return m

def gen_wc_keyvalues(m):
    wc = len(re.split('\W+',m[2]))
    if wc > 1000:
        return (10,1)
    else:
        return (wc//100, 1)
    
    
conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

df = spark.read.json('./FT-archive-last-30-days/*.json')
df_articles = df.select("id","title","bodyXML")

df_artRDD = df_articles.rdd.map(lambda m: (m.id,m.title,m.bodyXML))
df_artRDD = df_artRDD.filter(lambda m: m[2] != None)
df_textRDD = df_artRDD.map(lambda m: (m[0],m[1],bs4.BeautifulSoup(m[2],'xml').get_text()))
df_writetemp = df_textRDD.map(write_files)
#df_writetemp.count()

# data analysis
wc_distri_RDD = df_textRDD.map(gen_wc_keyvalues).reduceByKey(lambda v1, v2: v1 + v2)
print(wc_distri_RDD.collect())






