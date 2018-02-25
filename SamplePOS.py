#NLTK

def stopWorddir(path,pathto) :
    import nltk
    import os
    #from nltk.corpus import stopwords
    
    #Create a directory if it doesnt exist
    
    directory = pathto

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)  
        
    for filename in os.listdir(path):
        File = open(os.path.join(path,filename),'r+')
        sentence = File.read()
        
        
        #Get a different stopwords set
        #stops=set(stopwords.words('english'))
        
        
        
        file = open('Stopwords.txt');
        stops = set();
        for line in file :
            stops.add(line[:-1]);
            
        
        
        ###NLTK
        newtext = "";
        tokens = nltk.word_tokenize(sentence)
        #print(tokens)
        tagged = nltk.pos_tag(tokens)
       # taggednew = [];
        for tag in tagged :
            if(tag[0] in stops) :
                newtext = newtext+" "+tag[0].lower()
            else:
                newtext = newtext+" "+tag[1]
        
        File.close()
        File = open(os.path.join(pathto,filename),'w+')
        File.write(newtext);
        File.close()
        
        
        #s = []
        #for word  in taggednew:
        #    s.append(word[1])
        
        
        
        
        
        ###Stanford
        #from pycorenlp import StanfordCoreNLP
        #nlp = StanfordCoreNLP('http://localhost:9000')
        #
        ##Read from file
        #text = (
        #    """At eight o'clock on Thursday morning
        #... Arthur didn't feel very good. He was collecting apples. Aashiq's dog is happy."""
        #    )
        #output = nlp.annotate(text, properties={
        #    'annotators': 'tokenize,ssplit,pos,depparse,parse',
        #    'outputFormat': 'json'
        #})
        ##OUT = []
        #text = "";
        #for sentence in output['sentences']:
        #    for word in sentence['tokens']:
        #        #OUT.append((word['word'] ,word['pos']))
        #        if(word['word'] not in stops) :
        #            text = text+ " "+word['pos']
        #        else :
        #            text = text+" " + word['word'].lower()
        #
        #print(text)
        
        #print(output['sentences'][0]['pos'])
        #output = nlp.tokensregex(text, pattern='/Pusheen|Smitha/', filter=False)
        #print(output)
        #output = nlp.semgrex(text, pattern='{tag: VBD}', filter=False)
        #print(output)
                
if __name__ == '__main__':
    import sys
    stopWorddir(sys.argv[1],sys.argv[2])
