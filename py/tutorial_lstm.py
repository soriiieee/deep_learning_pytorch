# -*- coding: utf-8 -*-

import os,sys

#torch module import 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
torch.manual_seed(1)

#sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#----------------------------------------------
"""
tutorials LSTM official
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)


def prepare_sequence(seq,to_ix):
  """
  word2vec
  https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
  """
  idxs = [ ti_ix[w] for w in seq]
  return torch.tensor(idxs, dtype=torch.long)

traninig_data = [
      # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

# print(traninig_data)
word2idx={}
tag2idx={"DET":0,"NN":1, "V":2}
for sent,tags in traninig_data:
  for word in sent:
    if word not in word2idx:
      word2idx[word] = len(word2idx)

print(word2idx)
print(tag2idx)
EMBEDDING_DIM =6
HIDDEN_DIM=6
class LSTMTagger(nn.Module):
  def __init__(self, enbedding_dim, hidden_dim,vocab_size, tagset_size):
    super(LSTMTagger, self).__init__()

    self.hidden_dim = hidden_dim
    self.word_embeddins = nn.Embedding(vocab_size, embedding_dim)

    self.lstm = nn.LSTM(embedding_dim, tagset_size)
    self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

  def forward(self,sentence)
  




sys.exit()