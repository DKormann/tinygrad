#%%
from tinygrad import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

class LSTMCell:
  def __init__(self, input_size, hidden_size):
    k = hidden_size ** -0.5
    self.w_ih = Tensor.randn(input_size, hidden_size * 4).realize() * k
    self.b_ih = Tensor.randn(hidden_size * 4).realize() * k
    self.w_hh = Tensor.randn(hidden_size, hidden_size * 4).realize() * k
    self.b_hh = Tensor.randn(hidden_size * 4).realize() * k

  def __call__(self, x):
    h = Tensor.zeros(x.shape[1], self.w_hh.shape[0])
    c = Tensor.zeros(x.shape[1], self.w_hh.shape[0])
    res = []
    for t in range(x.shape[0]):

      gates = x[t].linear(self.w_ih, self.b_ih) + h.linear(self.w_hh, self.b_hh)
      i, f, g, o = gates.chunk(4, 1)
      i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
      c = (f * c) + (i * g).realize()
      h = (o * c.tanh()).realize()

      # res.append(h)
      # continue

      hc = Tensor.stack([h,c])
      res.append(hc)
      h = hc[0]
      c = hc[1]

    # return Tensor.stack(res).realize()
    return Tensor.stack(res)[:,0].realize()

class LSTM:
  def __init__(self,input_size, hidden_size, layers):
    self.cells = [LSTMCell(input_size, hidden_size) if i == 0 else LSTMCell(hidden_size,hidden_size) for i in range(layers)]
  
  def __call__(self,x:Tensor):
    for cell in self.cells: x = cell(x)
    return x.realize()


BS, T, dim = 4, 100, 1024
  
Tensor.manual_seed()
x = Tensor.randn(T, BS, dim)
x.requires_grad = True

lstm = LSTM(dim, dim, 1)
opt = Adam(get_parameters(lstm), lr=0.001)

p = lstm(x)

# %%
