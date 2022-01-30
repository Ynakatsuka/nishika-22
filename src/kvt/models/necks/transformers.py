import torch
import torch.nn as nn
from kvt.models.necks import gem1d


class FirstTokenPool(nn.Module):
    def forward(self, hidden_states, **kwargs):
        # hidden_states: (bs, seq_len, dim)
        return hidden_states[:, 0]


class LastTokenPool(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # hidden_states: (bs, seq_len, dim)
        # attention_mask: (bs, seq_len)
        if attention_mask is not None:
            bs, dim = hidden_states.shape[0], hidden_states.shape[-1]
            mask = attention_mask == 1
            hidden_states = hidden_states[mask].reshape(bs, -1, dim)
        return hidden_states[:, -1]


class BertPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, **kwargs):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # hidden_states: (bs, seq_len, dim)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertMaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, **kwargs):
        # hidden_states: (bs, seq_len, dim)
        return torch.max(hidden_states, dim=-1)


class BertAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # hidden_states: (bs, seq_len, dim)
        # attention_mask: (bs, seq_len)
        if attention_mask is not None:
            bs, dim = hidden_states.shape[0], hidden_states.shape[-1]
            mask = attention_mask == 1
            hidden_states = hidden_states[mask].reshape(bs, -1, dim)
        return torch.mean(hidden_states * attention_mask, dim=-1)


class BertGeMPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # hidden_states: (bs, seq_len, dim)
        # attention_mask: (bs, seq_len)
        if attention_mask is not None:
            bs, dim = hidden_states.shape[0], hidden_states.shape[-1]
            mask = attention_mask == 1
            hidden_states = hidden_states[mask].reshape(bs, -1, dim)
        return gem1d(hidden_states.transpose(1, 2)).transpose(1, 2)


class BertLSTMPool(nn.Module):
    def __init__(
        self, hidden_size, num_layers=1, dropout=0, bidirectional=False
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # hidden_states: (bs, seq_len, dim)
        if attention_mask is not None:
            bs, dim = hidden_states.shape[0], hidden_states.shape[-1]
            mask = attention_mask == 1
            hidden_states = hidden_states[mask].reshape(bs, -1, dim)

        output, _ = self.rnn(
            hidden_states
        )  # (bs, seq_len, dim) or (bs, seq_len, 2*dim)

        return output[:, -1]  # last hidden state
