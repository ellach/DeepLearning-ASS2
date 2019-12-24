import torch
import torch.nn as nn
import config
from torch.autograd import Variable
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import utils2
 
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
        self.linear1 = nn.Linear(5*config.EMBED_DIM, config.HIDDEN_DIM)
        self.linear2 = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_SIZE)
        self.dropout = nn.Dropout()
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        """
        forward model
        :params: inputs
        :return: y_hat  
        """   
        embeds_out = self.embeds(inputs).view(self.embeds.shape.size())  
        out1 = self.linear1(self.dropout(embeds_out)) 
        tanh_out1 = self.tanh(out1) 
        out2 = self.linear2(tanh_out1) 
        return out2 

class Solver(object):
    def __init__(self):

      self.mlp = MLP()
      self.cel = nn.CrossEntropyLoss() 
      self.initialize()

    def initialize(self):  
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=config.LR) 

    def epoch(self, dataset, optimize):
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
                x_batch = [[utils2.getIdOfWord(w) for w in a] for a,b in batch]
                y_batch = [utils2.getIdOfTag(b) for a,b in batch]

                x = Variable(torch.LongTensor(x_batch)) 
                y = Variable(torch.LongTensor(y_batch)) 

                y_hat = self.mlp(x)
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
            print('Accuracy : {0:.6f}'.format( goodAccuracy/totalAccuracy )) 
 
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
        plt.savefig('plot'+yLabel+'.png')  

 
    def prediction(self, testDataset, type):
        predOutput = []
        for i in range(0, len(testDataset)-config.BATCH_SIZE, config.BATCH_SIZE):       
            batch = testDataset[i:i+config.BATCH_SIZE]

            batchLength = len(batch)
            while batchLength < config.BATCH_SIZE:
               item = batch[-1]
               batch.append(item)

            x_batch = [[utils2.getIdOfWord(w) for w in a] for a in batch]
            x = Variable(torch.LongTensor(x_batch))
            y_hat = self.mlp(x)

            y_hat = (y_hat.data).numpy()
            for j in range(config.BATCH_SIZE):
                y_hat_index = np.argmax(y_hat[j])
                result = utils2.getWordOfId(y_hat_index) +"   "+utils2.getTagOfId(y_hat_index)    
                predOutput.append(result)

        test = open('test3.'+type,'w')           
        test.write('\n'.join(predOutput)) 
        test.close()

  
if __name__ == "__main__":

 solver = Solver() 

 if len(utils2.wordVectors)>0:
    solver.mlp.embeds.weight.data.copy_(torch.from_numpy(utils2.wordVectors))

 print('Start training ...')
 solver.epoch(utils2.windowTrain, True)

 print('Start validation ...')
 solver.epoch(utils2.windowDev, False)
  
 print('Start test ...')
 solver.prediction(utils2.windowTest,sys.argv[4]) 





