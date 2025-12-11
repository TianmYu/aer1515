import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden_size)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out