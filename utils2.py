import sys
import config
import numpy as np

windowTrain = []
windowDev = []
windowTest = []
wordVectors = []

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


trainData = [item for sublist in read_data(sys.argv[1], sys.argv[4]) for item in sublist]
devData = [item for sublist in read_data(sys.argv[2],sys.argv[4]) for item in sublist]
testData = [item for sublist in read_data(sys.argv[3], sys.argv[4], True) for item in sublist] 

for i in range(len(trainData)):
  if trainData[i][1] != 'S':
    windowTrain.append(([trainData[i-2][0],trainData[i-1][0],trainData[i][0],trainData[i+1][0],trainData[i+2][0]],trainData[i][1]))

for i in range(len(devData)): 
  if devData[i][1] != 'S':
    windowDev.append(([devData[i-2][0],devData[i-1][0],devData[i][0],devData[i+1][0],devData[i+2][0]],devData[i][1]))

for i in range(len(testData)):
  if testData[i][1] != 'S':
    windowTest.append(([testData[i-2][0],testData[i-1][0],testData[i][0],testData[i+1][0],testData[i+2][0]]))


tag = [a[1].replace('-','') for a in trainData ]
tag = [t for t in tag if t.isalpha() and t!='S'] 
tags = set(tag)
tags.add("UUUNKKK")

if len(sys.argv) > 5:
   wordVectors = np.loadtxt(sys.argv[5])
   embeddingVocab = np.loadtxt(sys.argv[6], dtype=str).tolist() 
   vocab = [word for word in embeddingVocab if word != '']
   config.VOCAB = len(wordVectors)
else:
   words = [a[0] for a in trainData]
   vocab = set(words)
   vocab.add("UUUNKKK")
   config.VOCAB = len(vocab)

config.OUTPUT_SIZE = len(tags)

wordToId = { word: i for i, word  in enumerate(vocab)} 
idToWord = { i: word for i, word in enumerate(vocab)}
tagToId = { tag: i for i, tag in enumerate(tags)}
idToTaG = { i: tag for i, tag in enumerate(tags)}


def getWordOfId(Id):
    return idToWord[Id]

def getIdOfWord(word):

    ans = None 
    if word in wordToId:
       ans = wordToId[word]
    else:
       ans = wordToId["UUUNKKK"]
    print("ans:  ", ans)   
    return ans 
  
def getTagOfId(Id):
    return idToTaG[Id]

def getIdOfTag(tag):  
    ans = None 
    if tag in tagToId:
       ans = tagToId[tag]
    else:
       ans = tagToId["UUUNKKK"]
    return ans 




