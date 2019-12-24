import utils3
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import config
import numpy as np
import sys

class Tagger:
    def __init__(self, dataset): 
       self.dataset = dataset
       self.words = []
       self.prefix = [] 
       self.suffix = []
       self.vocabPrefix = []
       self.vocabSuffix = []
       self.vocab = []
       self.tags = []
       self.tag = []
       self.w = []
       self.wordsPrefix = []
       self.wordsSuffix = []
       self.idToSuffix = {} 
       self.suffixToId = {}  
       self.idToPrefix = {}  
       self.prefixToId = {}  
       self.wordToId = {} 
       self.idToWord = {}
       self. tagToId = {}
       self.idToTag = {}
       

    def run(self):

        for i in range(len(self.dataset)):
          if self.dataset[i][1] != 'S':
           self.words.append(([self.dataset[i-2][0],self.dataset[i-1][0],self.dataset[i][0],self.dataset[i+1][0],self.dataset[i+2][0]],self.dataset[i][1]))

        for i in range(len(self.dataset)):
          if self.dataset[i][1] != 'S':
           self.prefix.append(([self.dataset[i-2][0][:3],self.dataset[i-1][0][:3],self.dataset[i][0][:3],self.dataset[i+1][0][:3],self.dataset[i+2][0][:3]]))

        for i in range(len(self.dataset)):
          if self.dataset[i][1] != 'S':
           self.suffix.append(([self.dataset[i-2][0][-3:],self.dataset[i-1][0][-3:],self.dataset[i][0][-3:],self.dataset[i+1][0][-3:],self.dataset[i+2][0][-3:]])) 

        
        if utils3.vocab:
           self.w  = utils3.vocab
           self.vocab = set(self.w)        
        else:
           self.w = [a[0] for a in self.dataset]
           self.vocab = set(self.w)
           self.vocab.add("UUUNKKK")
  

        self.tag = [a[1].replace('-','') for a in self.dataset]   
        self.wordsPrefix = [a[0][:3] for a in self.dataset]
        self.wordsSuffix = [a[0][-3:] for a in self.dataset]
        self.vocabPrefix = set(self.wordsPrefix)
        self.vocabPrefix.add("UUUNKKK")

        
        self.vocabSuffix = set(self.wordsSuffix)
        self.vocabSuffix.add("UUUNKKK")


        self.tags = [t for t in self.tag if t.isalpha() and t!='S']
        self.tags = set(self.tags)  
        self.tags.add("UUUNKKK") 
    

        self.idToSuffix = {i: suffix for i, suffix  in enumerate(self.vocabSuffix)} 
        self.suffixToId = {suffix: i for i, suffix  in enumerate(self.vocabSuffix)} 
        self.idToPrefix = {i: prefix for i, prefix  in enumerate(self.vocabPrefix)} 
        self.prefixToId = {prefix: i for i, prefix  in enumerate(self.vocabPrefix)} 
        self.wordToId = {word: i for i, word  in enumerate(self.vocab)} 
        self.idToWord = {i: word for i, word in enumerate(self.vocab)}
        self.tagToId = {tag: i for i, tag in enumerate(self.tags)}
        self.idToTag = {i: tag for i, tag in enumerate(self.tags)}

        
        config.OUTPUT_SIZE = len(self.tags)
        config.VOCAB_SUFFIX = len(self.vocabSuffix)
        config.VOCAB_PREFIX = len(self.vocabPrefix) 
        
    def getPrefixOfId(self,Id):
        return self.idToPrefix[Id]

    def getIdOfPrefix(self,word):
        ans = None 
        if word in self.prefixToId:
           ans = self.prefixToId[word]
        else:
           ans = self.prefixToId["UUUNKKK"]   
        return ans 

    def getSuffixOfId(self,Id):
        return self.idToSuffix[Id]

    def getIdOfSuffix(self,word):
        ans = None 
        if word in self.suffixToId:
           ans = self.suffixToId[word]
        else:
           ans = self.suffixToId["UUUNKKK"]   
        return ans 

    def getWordOfId(self,Id):
        return self.idToWord[Id]

    def getIdOfWord(self,word):
        ans = None 
        if word in self.wordToId:
           ans = self.wordToId[word]
        else:
           ans = self.wordToId["UUUNKKK"]   
        return ans 
  
    def getTagOfId(self,Id):
        return self.idToTag[Id]

    def getIdOfTag(self,tag):
        ans = None 
        if tag in self.tagToId:
           ans = self.tagToId[tag]
        else:
           ans = self.tagToId["UUUNKKK"]
        return ans   


class MLP(nn.Module):
    def __init__(self):
        """
        model parameters:
        VOCAB 
        EMBED_DIM
        BATCH_SIZE
        HIDDEN_DIM
        OUTPUT_SIZE  
        """
        super(MLP, self).__init__()

        self.embeds = nn.Embedding(config.VOCAB, config.EMBED_DIM)
        self.embeds.shape = torch.Tensor(config.BATCH_SIZE, 5*config.EMBED_DIM) 

        self.embeds_prefix = nn.Embedding(config.VOCAB_PREFIX, config.EMBED_DIM) 
        self.embeds_prefix.shape = torch.Tensor(config.BATCH_SIZE, 5*config.EMBED_DIM)

        self.embeds_suffix = nn.Embedding(config.VOCAB_SUFFIX, config.EMBED_DIM)
        self.embeds_suffix.shape = torch.Tensor(config.BATCH_SIZE, 5*config.EMBED_DIM)

        self.linear1 = nn.Linear(5*config.EMBED_DIM, config.HIDDEN_DIM)
        self.linear2 = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_SIZE)
        self.tanh = nn.Tanh()


    def forward(self, inputs, inputs_prefix, inputs_suffix):
        """
        forward model
        :params: inputs
        :return: y_hat  
        """  
        embeds_out = self.embeds(inputs).view(self.embeds.shape.size()) 
        embeds_out_prefix = self.embeds_prefix(inputs_prefix).view(self.embeds_prefix.shape.size())
        embeds_out_suffix = self.embeds_suffix(inputs_suffix).view(self.embeds_suffix.shape.size())

        out1 = self.linear1(embeds_out+embeds_out_prefix+embeds_out_suffix) 
        tanh_out1 = self.tanh(out1) 
        y_hat = self.linear2(tanh_out1)

        return y_hat 


class Solver(object):
    def __init__(self):

      self.model = MLP()
      self.cel = nn.CrossEntropyLoss() 
      self.initialize()    

    def initialize(self):  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LR) 
    
    def epoch(self, dataset,datasetSuffix,datasetPrefix, tagger, optimize):
        """
        run model N iterations
        :params dataset, optimizer: 
        """ 
        accuracyDataSet = []
        lossDataSet = []
        for epoch in range(config.NUM_EPOCHS):
            totalLoss, totalAccuracy , goodAccuracy = 0, 0, 0
            random.shuffle(dataset) 
            for i in range(0, len(dataset)-config.BATCH_SIZE, config.BATCH_SIZE):

                batch = dataset[i:i+config.BATCH_SIZE]
                batch_suffix = datasetSuffix[i:i+config.BATCH_SIZE]
                batch_prefix = datasetPrefix[i:i+config.BATCH_SIZE]

                x_batch = [[tagger.getIdOfWord(w) for w in a] for a,b in batch]
                x_batch_suffix = [[tagger.getIdOfSuffix(s[-3:]) for s in a] for a in batch_suffix] 
                x_batch_prefix = [[tagger.getIdOfPrefix(p[:3]) for p in a] for a in batch_prefix]
               
                y_batch = [tagger.getIdOfTag(b) for a,b in batch]
                x_prefix = Variable(torch.LongTensor(x_batch_prefix))

                x_suffix = Variable(torch.LongTensor(x_batch_suffix))  
                y = Variable(torch.LongTensor(y_batch)) 
                x = Variable(torch.LongTensor(x_batch)) 

                y_hat = self.model(x, x_prefix, x_suffix)
                loss = self.cel(y_hat, y)
                totalLoss += loss.data

                if optimize: 
                   self.optimizer.zero_grad() 
                   loss.backward()
                   self.optimizer.step()

                goodAccuracy, totalAccuracy = self.getAccuracy((y_hat.data).numpy(), (y.data).numpy())
                goodAccuracy += goodAccuracy
                totalAccuracy += totalAccuracy

            accuracyDataSet.append(goodAccuracy/totalAccuracy)
            lossDataSet.append(totalLoss/(len(dataset)/config.BATCH_SIZE))
            print('Loss : {0:.6f}'.format(totalLoss/(len(dataset)/config.BATCH_SIZE))) 
            print('Accuracy : {0:.6f}'.format( goodAccuracy/totalAccuracy)) 
           
        self.getGraph("Accuracy", accuracyDataSet)
        self.getGraph("Loss", lossDataSet) 


    def getAccuracy(self,y_hats, y):
        good, bad = 0, 0
        for i in range(0, config.BATCH_SIZE):
          y_hat = np.argmax(y_hats[i])

          if y_hat == y[i]:
             good += 1 
          else:
             bad += 1
        return good, (good + bad)


    def getGraph(self, yLabel, x_data):
        plt.figure()
        y_data = [i for i in range(0, config.NUM_EPOCHS)]
        plt.xlabel('Iterations')
        plt.ylabel(yLabel)
        plt.plot(y_data, x_data)
        plt.grid(True)
        plt.savefig('plot'+yLabel+'3'+'.png')  

 
    def prediction(self, testDataset,testDatasetSufix,testDatasetPreffix, tagger, train, type):
        predOutput = []
        for i in range(0, len(testDataset)-config.BATCH_SIZE, config.BATCH_SIZE):       
            batch = testDataset[i:i+config.BATCH_SIZE]

            batchLength = len(batch)
            while batchLength < config.BATCH_SIZE:
               item = batch[-1]
               batch.append(item)

            x_batch = [[ tagger.getIdOfWord(w) for w in a] for a,b in batch]
            x_batch_suffix = [[tagger.getIdOfSuffix(s[-3:]) for s in a] for a,b in batch] 
            x_batch_prefix = [[tagger.getIdOfPrefix(p[:3]) for p in a] for a,b in batch]

            x = Variable(torch.LongTensor(x_batch))
            x_prefix = Variable(torch.LongTensor(x_batch_prefix))
            x_suffix = Variable(torch.LongTensor(x_batch_suffix))  

            y_hat = self.model(x, x_prefix, x_suffix) 
            y_hat = (y_hat.data).numpy()
  
            for j in range(config.BATCH_SIZE):
                y_hat_index = np.argmax(y_hat[j])
                input = train.getWordOfId(y_hat_index) +"   "+train.getTagOfId(y_hat_index) 
                predOutput.append(input)

        test = open('test4.'+type,'w')           
        test.write('\n'.join(predOutput)) 
        test.close()

  
if __name__ == "__main__":

 train = Tagger(utils3.datasetTrain)
 train.run() 
 solver = Solver()

 if len(sys.argv) > 5: 
    solver.model.embeds.weight.data.copy_(torch.from_numpy(utils3.wordVectors))
 
 print('Start training ...')
 solver.epoch(train.words,train.suffix,train.prefix, train, True)
 
 dev = Tagger(utils3.datasetDev)
 dev.run()
 print('Start validation ...')
 solver.epoch(dev.words,dev.suffix,dev.prefix, dev, False)
 
 test = Tagger(utils3.datasetTest)
 test.run()
 print('Start prediction ... ') 
 solver.prediction(test.words,test.suffix,test.prefix, test, train, sys.argv[4])
 



