import torch
import torch.nn as nn 


class LSTMGenerator(nn.Module):
    def __init__(self, src_dict, target_dict):
        super(self, LSTMGenerator).__init__()  
        
        
        self.src_dict = src_dict
        self.target_dict = target_dict
        
        
        
        

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dims, hidden_dims, num_layers):
        super(self, LSTMGenerator).__init__()  
        
        
        self.embedding = nn.embedding(vocab_size, embedding_dims)
        self.rnn = nn.LSTM(input_size=embedding_dims, hidden_size=hidden_dims, num_layers=num_layers, bidirectional=False)
        
        
    def forward(self, input):
        batch_size, seq_len = input.size()
        embedded = self.embedding(input)
        
        embedded = embedd.resha()
        
        
        
        
        
        
