import torch.nn as nn
import torch.nn.functional as F


class LSTM_FutureChange(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_ratio, output_size):
        super(LSTM_FutureChange, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        # dropout layer
        self.dropout = nn.Dropout(dropout_ratio)

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_ratio,
                            batch_first=True)

        # fully connected layer
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, self.output_size)

        # # BATCH_NORMALIZATION layer
        self.batch1d1 = nn.BatchNorm1d(60)
        self.batch1d2 = nn.BatchNorm1d(hidden_size)
        self.batch1d3 = nn.BatchNorm1d(64)

    def forward(self, x, hidden):

        batch_size = x.size(0)

        # lstm layer x[B, seq_len, input_size]
        x, hidden = self.lstm(x, hidden)

        # get the last output
        x = x[:, -1, :]

        # drop out and batch_norm
        x = self.dropout(x)
        x = self.batch1d2(x)

        # first fc layer
        x = F.relu(self.fc1(x))

        # drop out and batch_norm
        x = self.dropout(x)
        x = self.batch1d3(x)

        # first fc layer
        x = F.sigmoid(self.fc2(x))

        return x, hidden

    def init_hidden(self, batch_size, train_on_gpu):
        '''
        Initializes hidden state
        '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

        return hidden


class LSTM_FutureChangeGeneral(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_ratio, output_size):
        super(LSTM_FutureChangeGeneral, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        # dropout layer
        self.dropout = nn.Dropout(dropout_ratio)

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_ratio,
                            batch_first=True)

        # fully connected layer
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, self.output_size)

        # # BATCH_NORMALIZATION layer
        self.batch1d1 = nn.BatchNorm1d(60)
        self.batch1d2 = nn.BatchNorm1d(hidden_size)
        self.batch1d3 = nn.BatchNorm1d(64)

    def forward(self, x, hidden):

        batch_size = x.size(0)

        x = self.batch1d1(x)
        # lstm layer x[B, seq_len, input_size]
        x, hidden = self.lstm(x, hidden)

        # get the last output
        x = x[:, -1, :]

        # drop out and batch_norm
        x = self.dropout(x)
        x = self.batch1d2(x)

        # first fc layer
        x = F.relu(self.fc1(x))

        # drop out and batch_norm
        x = self.dropout(x)
        x = self.batch1d3(x)

        # first fc layer
        x = F.sigmoid(self.fc2(x))

        return x, hidden

    def init_hidden(self, batch_size, train_on_gpu):
        '''
        Initializes hidden state
        '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

        return hidden
