#%%
from tinygrad import Tensor
from tinygrad.tensor import Function

from tinygrad.nn import Embedding, Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
# from tinygrad.jit import TinyJit
from tinygrad import TinyJit

import numpy as np

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
      h = (o * c.tanh()).realize()
      c = (f * c) + (i * g).realize()

      res.append(Tensor.stack([h,c]))
      h = res[-1][0]
      c = res[-1][1]
    
    return Tensor.stack(res)[:,0].realize()

class LSTM:
  def __init__(self,input_size, hidden_size, layers,_):
    self.cells = [LSTMCell(input_size, hidden_size) if i == 0 else LSTMCell(hidden_size,hidden_size) for i in range(layers)]
  
  def __call__(self,x:Tensor):
    for cell in self.cells: x = cell(x)
    return x.realize(), None

def logsumexp(a:Tensor, b:Tensor):
    mx = Tensor.maximum(a,b).maximum(-1e10)
    s = (a-mx).exp() + (b-mx).exp()
    return s.log() + mx

inf = float('inf')

def shear(d:Tensor,value = 0):
    B,X,Y,C = d.shape
    d = d.pad(((0,0),(0,Y),(0,0),(0,0)),value=value)
    d = d.transpose(1,2).reshape((B,-1,C))
    d = d[:,:(X+Y-1)*Y,:].realize()
    return d.reshape((B,Y,X+Y-1,C)).transpose(1,2)

def unshear(x:Tensor):
    B,X,Y = x.shape
    x = x.reshape((B,-1,))
    x = x.pad(((0,0),(0,X),))
    x = x.reshape((B,X,Y+1))
    return x.shrink(((0,B),(0,X),(0,Y+1-X)))

class TransducerLoss(Function):

  def forward(self, d:Tensor, labels:Tensor):
    self.B,self.X,self.Y,self.C = d.shape

    self.labels = Tensor(labels).pad(((0,0),(0,1)))
    self.lattice = shear(Tensor(d), 0.)
    self.X = self.X+self.Y-1
    assert self.lattice.shape == (self.B,self.X,self.Y,self.C), f"{self.lattice.shape}"

    self.skip = shear(Tensor(d)[:,:,:,-1:],1.)[:,:,:,0].log()

    self.p = self.lattice[
      Tensor(np.arange(self.B).reshape((-1,1,1))),
      Tensor(np.arange(self.X).reshape((1,-1,1))),
      Tensor(np.arange(self.Y).reshape((1,1,-1))),
      self.labels.reshape((self.B,1,-1))].log()

    assert self.p.shape == (self.B, self.X, self.Y)
    self.a = [Tensor([0]*self.B).reshape(-1,1).pad(((0,0),(0,self.Y-1),),-inf).realize()]

    for x in range(0,self.X-1):
      self.a.append(logsumexp(
        (self.a[-1] + self.skip[:,x,:]).realize(),
        (
          self.a[-1][:,:-1].pad(((0,0),(1,0),),-inf).realize() + self.p[:,x,:-1].pad(((0,0),(1,0),),-inf)
        ).realize()
      ))

    return (-self.a[-1].max(1).sum()).lazydata
    
  def backward(self, g):

    self.b = [None] * (self.X-1) + [Tensor.ones(self.B,self.Y)]
    for x in range(self.X-2,-1,-1):
      self.b[x] = (
        logsumexp(
          self.b[x+1] + self.skip[:,x,:],
          self.b[x+1][:,1:].pad(((0,0),(0,1),),-inf).realize() + self.p[:,x,:].realize()
        )).realize()

    self.skg, self.p_grad = None, None

    for a,b in zip(self.a[:-1], self.b[1:]):
      sg = (a + b).reshape(self.B, 1,-1)
      self.skg = sg if self.skg is None else self.skg.cat(sg,dim=1).realize()
      pg = a.unsqueeze(1) + b[:,1:].pad(((0,0),(0,1),),-inf).unsqueeze(1)
      self.p_grad = pg if self.p_grad is None else self.p_grad.cat(pg,dim=1).realize()
    
    self.skg = (unshear(self.skg.transpose(1,2)) - self.b[0][:,0].unsqueeze(1).unsqueeze(1)).transpose(1,2).exp().realize()
    self.p_grad = (unshear(self.p_grad.transpose(1,2))).transpose(1,2).realize() - self.b[0][:,0].unsqueeze(1).unsqueeze(1)

    self.p_grad = self.p_grad.exp().unsqueeze(-1).mul(Tensor.eye(self.C-1)[self.labels].unsqueeze(1))
    grad = self.p_grad.cat(self.skg.unsqueeze(-1), dim=-1).pad(((0,0),(0,1),(0,0),(0,0)))

    assert not (grad.numpy() == 0).all()

    return (-grad).lazydata, None

# ci, maxx, maxy = load_data(5)
maxx,maxy = 100,100
BS = 16
dim = 1024
VOCAB = 28

def mask(d,X_lens, Y_lens, maxX, maxY, vocab=28):

  d = d.pad(((0,0),(0,1),(0,0),(0,0))) # padding after X for masking
  xrange = Tensor.arange(maxX+1)
  mask = (xrange.unsqueeze(-1) < X_lens).T

  d = d * mask.unsqueeze(-1).unsqueeze(-1)
  mask = Tensor.arange

  yrange = Tensor.arange(maxY + 1)
  mask = (yrange.unsqueeze(-1).unsqueeze(-1)) <= Y_lens
  mask = mask.transpose(0,2)
  d = d * mask.unsqueeze(-1)

  line = (yrange.unsqueeze(0) == Y_lens.unsqueeze(-1))
  line = line.unsqueeze(1) * (xrange.unsqueeze(-1) >= X_lens.unsqueeze(1).unsqueeze(-1))
  line = line.unsqueeze(3).pad(((0,0),(0,0),(0,0),(vocab,0)))

  d = d + line
  return d

class Model:
  def __init__(self,dropout= 0):
    self.dropout = dropout
    self.xemb = Embedding(VOCAB, dim)
    self.yemb = Embedding(VOCAB, dim)
    self.encoder = LSTM(dim, dim, 1, dropout)
    self.decoder = LSTM(dim, dim, 1, dropout)
    self.joint = Linear(dim, VOCAB+1)

  def join(self,x:Tensor,y:Tensor,xlens:Tensor,ylens:Tensor,maxx,maxy):
    global px, py, pxy
    x = self.xemb(x)
    y = self.yemb(y)

    px,_ = self.encoder(x.T) # lstm expects (S,B,D)
    py,_ = self.decoder(y.T)
    py = py.pad(((1,0),(0,0),(0,0)))
    pxy = px.T.unsqueeze(2) + py.T.unsqueeze(1)
    pxy = self.joint(pxy).softmax(-1)
    pxy = mask(pxy, xlens, ylens, maxx, maxy)
    return pxy

def init_model():
  global model, opt
  Tensor.manual_seed()
  model = Model()
  for p in get_parameters(model):p.realize()
  opt = Adam(get_parameters(model), lr=0.001)

init_model()

def forward(x,y,xlens,ylens,maxx,maxy):
  p = model.join(x,y,xlens,ylens,maxx,maxy)
  L = TransducerLoss.apply(p, y)
  return L.realize()

def step(x,y,xlens, ylens, maxx,maxy):
  L = forward(x,y,xlens,ylens,maxx,maxy)
  opt.zero_grad()
  L.backward()
  for p in opt.params: 
    gnp = p.grad.numpy()
    if (gnp == 0).all(): print("zg")

  opt.step()
  lnp = L.numpy()
  while lnp == 0: 
    print(lnp, end= "zl", flush=True)
    lnp = L.numpy()
  return L.realize()

hist = []
Tensor.manual_seed()
modelA = Model()
for p in get_parameters(modelA): p.realize()
optA = Adam(get_parameters(modelA), lr=0.001)
Tensor.manual_seed()
modelB = Model()
for p in get_parameters(modelB): p.realize()
optB = Adam(get_parameters(modelB), lr=0.001)

stepA = TinyJit(step)
stepB = TinyJit(step)


for i in range(100):
  print(end=".")

  x = Tensor.randint(16,100,high=28)
  y = Tensor.randint(16,100,high=28)
  xlens = Tensor([100]*16)
  ylens = Tensor([100]*16)

  model = modelA
  opt = optA
  LA = stepA(x,y,xlens,ylens,maxy,maxy)
  model = modelB
  opt = optB
  LB = stepB(x,y,xlens,ylens,maxy,maxy)

  lnpa, lnpb = LA.numpy(), LB.numpy()
  if lnpa != lnpb:
    print("fail", lnpa, lnpb)
    break
  

