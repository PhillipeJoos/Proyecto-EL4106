from torch import nn

class RNNModel(nn.Module):
    def __init__(
        self,
        rnn_type,
        n_input_channels,
        hidd_size=256,
        out_features = 3,
        num_layers=1,
    ):
        """
        Para utilizar una vanilla RNN entregue rnn_type="RNN"
        Para utilizar una LSTM entregue rnn_type="LSTM"
        Para utilizar una GRU entregue rnn_type="GRU"
        """
        super().__init__()

        self.rnn_type = rnn_type

        if rnn_type == "GRU":
            self.rnn_layer = nn.GRU(n_input_channels, hidd_size, batch_first=True, num_layers=num_layers)

        elif rnn_type == "LSTM":
            self.rnn_layer = nn.LSTM(n_input_channels, hidd_size, batch_first=True, num_layers=num_layers)

        elif rnn_type == "RNN":
            self.rnn_layer = nn.RNN(n_input_channels, hidd_size, batch_first=True, num_layers=num_layers)

        else:
            raise ValueError(f"rnn_type {rnn_type} not supported.")

        self.net = nn.Sequential(
            nn.Linear(hidd_size, out_features),
        )

        self.flatten_layer = nn.Flatten()

    def forward(self, x):
        if self.rnn_type == "GRU":
            out, h = self.rnn_layer(x)

        elif self.rnn_type == "LSTM":
            out, (h, c) = self.rnn_layer(x)

        elif self.rnn_type == "RNN":
            out, h = self.rnn_layer(x)

        out = h[-1]

        return self.net(out)