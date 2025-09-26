from torch.ao.nn.quantizable.modules.activation import MultiheadAttention as MultiheadAttention
from torch.ao.nn.quantizable.modules.rnn import LSTM as LSTM, LSTMCell as LSTMCell

__all__ = ['LSTM', 'LSTMCell', 'MultiheadAttention']
