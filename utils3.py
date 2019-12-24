import config
import numpy as np
import sys

datasetTrain = []
datasetDev = []
datasetTest = []

def read_data(fname, type, test=False):
    """
    open and preprocess data (train/ dev/test)
    :params: fname, type, test  type - 'pos'/'ner', test -'True'/'False', fname-path to file
    :return list - preprocessed file 
    """
    sentencesList = []
    with open(fname) as input_file:
        sentence = []
        for line in input_file.read().split('\n'):
            if len(line) == 0:
                sentencesList.append( [("<s>", "S"), ("<s>", "S")] +
                        sentence +
                [("</s>", "S"), ("</s>", "S")])
                sentence = []                
                continue
            if type == 'ner':
               for words in line.strip().split():
                   if test:
                      text = words.strip().split() 
                      sentence.append((text[0],""))   
                   else:
                       text, label = words.strip().split("/",1)
                       sentence.append((text, label))
            if type == 'pos':
                if test:
                   text = line.strip().split() 
                   sentence.append((text[0],"")) 
                else:
                    text, label = line.strip().split(" ",1)
                    sentence.append((text, label))  
        return sentencesList
   
  
datasetTrain = [item for sublist in read_data(sys.argv[1], sys.argv[4]) for item in sublist]
datasetDev = [item for sublist in read_data(sys.argv[2], sys.argv[4]) for item in sublist]
datasetTest = [item for sublist in read_data(sys.argv[3], sys.argv[4], True) for item in sublist] 


vocab = []
if len(sys.argv) > 5:
   wordVectors = np.loadtxt(sys.argv[5])
   config.VOCAB = len(wordVectors)  
   embeddingVocab = np.loadtxt(sys.argv[6], dtype=str).tolist() 
   vocab = [word for word in embeddingVocab if word != '']
   config.VOCAB = len(wordVectors)

 

