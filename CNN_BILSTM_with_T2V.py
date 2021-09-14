import torch.nn as nn
import torch.nn.functional as F
import torch


class Time2Vector(nn.Module):
    def __init__(self, input_size, activation="sin", dropout_ratio=0.4):
        super(Time2Vector, self).__init__()

        # input size
        self.input_size = input_size


        # non-periodic/linear vector
        self.fc1 = nn.Linear(input_size, input_size)

        # periodic/linear vector
        self.fc2 = nn.Linear(input_size, input_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout_ratio)

        # activation function
        if activation == "sin":
            self.activation = torch.sin
        else:
            self.activation = torch.cos

    def forward(self, x):
        # periodic layer
        out_periodic = self.fc1(x)
        out_periodic = self.dropout(out_periodic)

        # non-periodic layer
        out_nonperiodic = self.activation(self.fc2(x))
        out_nonperiodic = self.dropout(out_nonperiodic)

        # output
        out = torch.cat([out_periodic, out_nonperiodic], -1)

        return out


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_ratio, output_size, kernel_size=1):
        super(CNN_BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        # dropout layer
        self.dropout = nn.Dropout(dropout_ratio)

        # Time2Vector
        self.t2v = Time2Vector(self.input_size)


        # Conv1
        self.conv1 = nn.Conv1d(self.input_size*3, (self.input_size*3)*4, kernel_size=kernel_size, bias=False)
        self.conv2 = nn.Conv1d((self.input_size*3)*4, (self.input_size * 3) * 4, kernel_size=kernel_size, bias=False)


        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_ratio,
                            batch_first=True, bidirectional=True, bias=False)

        # fully connected layer
        self.fc1 = nn.Linear(hidden_size * 2, 128, bias=False)
        self.fc2 = nn.Linear(128, self.output_size)

        # # BATCH_NORMALIZATION layer
        self.batch1d1_conv1 = nn.BatchNorm1d((self.input_size*3)*4)
        self.batch1d1_conv2 = nn.BatchNorm1d((self.input_size*3)*4)
        self.batch1d2 = nn.BatchNorm1d(hidden_size * 2)
        self.batch1d3 = nn.BatchNorm1d(64)

    def forward(self, x, hidden):

        # time_embeddings
        out_embed = self.t2v(x)

        # combine with input
        x = torch.cat([x, out_embed], -1)

        # get size of input to conv
        batch_size, seq_len, features = x.size()

        # reshape input to conv layer
        x = x.view(batch_size * seq_len, features)

        # go throw conv layers
        x = F.leaky_relu(self.conv1(x))

        x = self.dropout(x)
        x = self.batch1d1_conv1(x)

        x = F.leaky_relu(self.conv2(x))

        x = self.dropout(x)
        x = self.batch1d1_conv2(x)

        # reshape x for LSTM layer
        x = x.view(batch_size, seq_len, -1)

        # lstm layer x[B, seq_len, input_size]
        x, hidden = self.lstm(x, hidden)

        # get the last output
        x = x[:, -1, :]

        # drop out and batch_norm
        x = self.dropout(x)
        x = self.batch1d2(x)

        # first fc layer
        x = F.leaky_relu(self.fc1(x))

        # drop out and batch_norm
        x = self.dropout(x)
        x = self.batch1d3(x)

        # first fc layer
        x = self.fc2(x)

        return x, hidden

    def init_hidden(self, batch_size, train_on_gpu):
        '''
        Initializes hidden state
        '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.num_layers*2, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.num_layers*2, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers*2, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers*2, batch_size, self.hidden_size).zero_())

        return hidden





