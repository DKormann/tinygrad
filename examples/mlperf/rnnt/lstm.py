#%%

from tinygrad import Tensor, nn
from tinygrad.nn import Linear, Embedding
from tinygrad.engine.realize import method_cache

method_cache.clear()

HIDDEN_SIZE=1024 
LAYERS=4

B = 2
T = 150

class LSTMCell:
  def __init__(self, input_size, hidden_size):
    self.hidden_size = hidden_size
    self.w_ih = nn.Linear(input_size, hidden_size * 4)
    self.w_hh = nn.Linear(hidden_size, hidden_size * 4)

  def __call__(self, x):
    h = Tensor.zeros(x.shape[1], self.hidden_size)
    c = Tensor.zeros(x.shape[1], self.hidden_size)
    res = []
    for t in range(x.shape[0]):

      gates = self.w_ih(x[t]) + self.w_hh(h)
      i, f, g, o = gates.chunk(4, 1)
      i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
      h = (o * c.tanh()).realize()
      c = (f * c) + (i * g).realize()

      res.append(Tensor.stack(h,c))
      h = res[-1][0]
      c = res[-1][1]
    
    ret = res[0].unsqueeze(0)
    for e in res[1:]: ret = ret.cat(e.unsqueeze(0) , dim=0).realize()
    return ret[:,0].realize()

class LSTM:
  def __init__(self):
    self.cells = [LSTMCell(HIDDEN_SIZE,HIDDEN_SIZE) for i in range(LAYERS)]
  
  def __call__(self,x:Tensor):
    for cell in self.cells: x = cell(x)
    return x.realize()



net = LSTM()
opt = nn.optim.Adam(nn.state.get_parameters(net), lr=1e-3)
x = Tensor.rand(B,T,HIDDEN_SIZE)
assert net(x).shape == (B, T, HIDDEN_SIZE)

# %%
