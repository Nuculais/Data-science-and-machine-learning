# -*- coding: utf-8 -*-
"""
Recurrent Neural Network trained on Reddit comments (yeah, I know)
"""

import theano
import theano.tensor as T

import csv
import numpy as np
import itertools
import operator
import nltk
import sys
import matplotlib.pyplot as plt
from utils import *
from datetime import datetime

#from collections import Counter

nltk.download("book")


vocaSize = 8000
unknownToken = "UNKNOWN_TOKEN"
sentenceStartToken = "SENTENCE_START"
sentenceEndToken = "SENTENCE_END"

#Reading data and appending SENTENCE_START and SENTENCE_END tokens
with open('commentsData.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace = True)
    reader.next()
    
    #Split comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
   
    #Append start and end tokens
    sentences = ["%s %s %s" % (sentenceStartToken, x, sentenceEndToken) for x in sentences]
print("Parsed %d sentences" % (len(sentences)))

#Tokenizing sentences into words
tokenizedSen = [nltk.word_tokenize(sen) for sen in sentences]
tokenizedSen = [[sentenceStartToken] + sen + [sentenceEndToken] for sen in tokenizedSen]

#Counting word frequencies
wordFreq = nltk.FreqDist(itertools.chain(*tokenizedSen))
print("Found %d unique word tokens." % len(wordFreq.items())

#Building wordToIndex and IndexToWord vectors from most common words
vocab = wordFreq.most_common(vocaSize-1)
indexToWord = [x[0] for x in vocab]
indexToWord.append(unknownToken)
wordToIndex = dict([(w,i) for i,w in enumerate(indexToWord)])
#print("Using vocabulary size %d", % vocaSize)
#print("Least frequent word is '%s' and it appears %d times." % (vocab[-1][0], vocab[-1][1]))

#Replace all unknown words with the unknownToken
for i, sen in enumerate(tokenizedSen):
    tokenizedSen[i] = [w if w in wordToIndex else unknownToken for w in sen]
    
print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after preprocessing: '%s'" % tokenizedSen[0])

#Creating of training data
X_train= np.asarray([[wordToIndex[w] for w in sen[:-1]] for sen in tokenizedSen])
y_train= np.asarray([[wordToIndex[w] for w in sen[1:]] for sen in tokenizedSen])



#Numpy implementation
class NumpyRNN:
    def _init_(self, word_dim, hidden_dim=100, bptt_truncate=4):
        #Assign instance variables
        self.word_dim=word_dim
        self.hidden_dim=hidden_dim
        self.bptt_truncate=bptt_truncate
        #Radomly initialize network parameters
        self.U=np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V=np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W=np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        

#Forward propagation
def ForwardPropagation(self, x):
    #Time steps
    T = len(x)
    #Save hidden states to var s, add an additional element for the initial hidden, set to 0
    s = np.zeros((T+1(self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    #Save the outputs of each time step for later
    o = np.zeros((T, self.word_dim))
    
    for t in np.arange(T):
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dor(s[t]))
        return[o,s]
    
NumpyRNN.forward_propagation = forward_propagation


#Forward propagation returning the index with the hoghest score
def Predict(self, x):
    o,s = self.forward_propagation(x)
    return np.argmax(o,axis=1)

NumpyRNN.predict = predict


#Example
np.random.seed(10)
model = NumpyRNN(vocaSize)
o,s = model.forward_propagation(X_train[10])
print(o.shape)
print(o)

predictions = model.Predict(X_train)
print(predictions.shape)
print(predictions)


#Loss calculation, Cross-entropy
def CalcTotalLoss(self,x,y):
    L=0
    for i in np.arange(len(y)):
        self.forward_propagation(x[i])
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        L += -1* np.sum(np.log(correct_word_predictions))
        return L
    
def CalcLoss(self,x,y):
    #(Total loss)/(number oftraining examples)
    N = np.sum((len(y_i) for y_i in y))
    return self.CalcTotalLoss(x,y)/N
NumpyRNN.CalcTotalLoss = CalcTotalLoss
NumpyRNN.CalcLoss = CalcLoss

print("Expected loss for random predictions: %f", % np.log(vocaSize))
print("Actual loss: %f", model.CalcLoss(X_train[:1000], y_train[:1000]))

#BPTT - backpropagation Though Time
def BPTT(self, x, y):
    T = len(y)
    #forward propagation
    o,s = self.forward_propagation(x)
    #Accumulate gradients in varibles
    dLdU=np.zeros(self.U.shape)
    dLdV=np.zeros(self.V.shape)
    dLdW=np.zeros(self.W.shape)
    delta_o= o
    delta_o[np.arange(len(y)),y] -=1.
    
    #Backwards outputs
    for T in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        delta_t = self.V.T.dot(delta_o[t]) * (1-(s[t]**2))
        #BPTT
        for bptt_step in np.arange(max(0, t-self.bptt_truncate),t+1)[::-1]:
            dLdW += np.outer(delta_t, s[bptt_step-1])
            dLdU[:,x[bptt_step]] += delta_t
            #Update delta
            delta_t = self.W.T.dot(delta_t) * (1-s[bptt_step-1] ** 2)
            return [dLdU, dLdV, dLdW]
        
        NumpyRNN.BPTT = BPTT