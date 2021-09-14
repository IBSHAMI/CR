import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_ratio):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=dropout_ratio, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        # x shape: (Batch_size, seq_length)

        outputs, hidden = self.lstm(x)
        # outputs shape: (seq_length, N, hidden_size)

        return hidden

    # def init_hidden(self, batch_size_num, train_on_gpu):
    #     ''' Initializes hidden state '''
    #     # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    #     # initialized to zero, for hidden state and cell state of LSTM
    #     weight = next(self.parameters()).data
    #
    #     if (train_on_gpu):
    #         hidden = (weight.new(self.num_layers, batch_size_num, self.hidden_size).zero_().cuda(),
    #                   weight.new(self.num_layers, batch_size_num, self.hidden_size).zero_().cuda())
    #     else:
    #         hidden = (weight.new(self.num_layers, batch_size_num, self.hidden_size).zero_(),
    #                   weight.new(self.num_layers, batch_size_num, self.hidden_size).zero_())
    #
    #     return hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_ratio, output_size):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.input_size = input_size

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=dropout_ratio, batch_first=True)

        # # BATCH_NORMALIZATION layer
        self.batch1d = nn.BatchNorm1d(hidden_size)

        # fully connected layer
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, x, hidden):
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        batch_size = x.size(0)
        x = x.unsqueeze(0)

        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.batch1d(out)

        out = self.fc1(out)

        out = self.dropout(out)
        out = self.batch1d(out)

        out = self.fc2(out)

        out = out.view(batch_size, -1, self.output_size)

        return out, hidden

    # def init_hidden(self, batch_size_num, train_on_gpu):
    #     ''' Initializes hidden state '''
    #     # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    #     # initialized to zero, for hidden state and cell state of LSTM
    #     weight = next(self.parameters()).data
    #
    #     if (train_on_gpu):
    #         hidden = (weight.new(self.num_layers, batch_size_num, self.hidden_size).zero_().cuda(),
    #                   weight.new(self.num_layers, batch_size_num, self.hidden_size).zero_().cuda())
    #     else:
    #         hidden = (weight.new(self.num_layers, batch_size_num, self.hidden_size).zero_(),
    #                   weight.new(self.num_layers, batch_size_num, self.hidden_size).zero_())
    #
    #     return hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, train_on_gpu):
        batch_size = x.shape[1]
        target_len = y.shape[0]

        # outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        # hidden_encoder = self.encoder.init_hidden(batch_size, train_on_gpu)
        print(x.type())
        hidden = self.encoder(x)

        outputs, hidden = self.decoder(x, hidden)



        # for t in range(1, target_len):
        #     # Use previous hidden, cell as context from encoder at start
        #     output, hidden, cell = self.decoder(x, hidden, cell)
        #
        #     # Store next output prediction
        #     outputs[t] = output
        #
        #     # Get the best word the Decoder predicted (index in the vocabulary)
        #     best_guess = output.argmax(1)
        #
        #     # With probability of teacher_force_ratio we take the actual next word
        #     # otherwise we take the word that the Decoder predicted it to be.
        #     # Teacher Forcing is used so that the model gets used to seeing
        #     # similar inputs at training and testing time, if teacher forcing is 1
        #     # then inputs at test time might be completely different than what the
        #     # network is used to. This was a long comment.
        #     x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
