#%%
from tinygrad import Tensor
from tinygrad.lazy import LazyBuffer
from tinygrad.nn import Linear, Embedding
from tinygrad.tensor import Function
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, ReduceOps
from tinygrad.dtype import dtypes

from train import shear, unshear, inf, mask, logsumexp

from utils import imshow, analysis
import numpy as np


def analysis():
  global A, B, AB, skip, p
  def stack(X):
    R = X[0].unsqueeze(1)
    for x in X[1:]:
        R = R.cat(x.unsqueeze(1), dim=1).realize()
    return unshear(R.transpose(1,2)).transpose(1,2)

  A = stack(ctx.a).realize()
  B = stack(ctx.b).realize()
  AB = A[0] + B[0]
  AB = AB - AB.max()
  skip = unshear(ctx.skip.transpose(1,2)).transpose(1,2).realize()
  p = unshear(ctx.p.transpose(1,2)).transpose(1,2).realize()

def logsumexp(a:Tensor, b:Tensor):
    mx = Tensor.maximum(a,b).maximum(-1e10)
    s = (a-mx).exp() + (b-mx).exp()
    return s.log() + mx

class TransducerLoss(Function):

  def forward(self, d:Tensor, labels:Tensor):
    self.B,self.X,self.Y,self.C = d.shape

    self.labels = Tensor(labels).pad(((0,0),(0,1)))
    self.lattice = shear(Tensor(d), 0.)
    self.X = self.X+self.Y-1
    self.skip = shear(Tensor(d)[:,:,:,-1:],1.)[:,:,:,0].log()

    self.p = self.lattice[
      Tensor(np.arange(self.B).reshape((-1,1,1))),
      Tensor(np.arange(self.X).reshape((1,-1,1))),
      Tensor(np.arange(self.Y).reshape((1,1,-1))),
      self.labels.reshape((self.B,1,-1))].log()

    self.a = [Tensor([0]*self.B).reshape(-1,1).pad(((0,0),(0,self.Y-1),),-inf).realize()]

    for x in range(0,self.X-1):
      self.a.append(logsumexp((self.a[-1] + self.skip[:,x,:]).realize(),
        (self.a[-1][:,:-1].pad(((0,0),(1,0),),-inf).realize() + self.p[:,x,:-1].pad(((0,0),(1,0),),-inf)).realize()))

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

#%%
def MAX(a:LazyBuffer, b:LazyBuffer): return a.e(BinaryOps.CMPLT, b).e(TernaryOps.WHERE, b, a)
def EXP(a:LazyBuffer): return a.e(BinaryOps.MUL, a.const(1/math.log(2))).e(UnaryOps.EXP2)
def LOG(a:LazyBuffer): return a.e(UnaryOps.LOG2).e(BinaryOps.MUL, a.const(math.log(2)))
def PAD(a:LazyBuffer, arg, value):
  res = a.pad(arg)
  v = res.const(value)
  z = a.const(True).pad(arg).cast(dtypes.bool)
  return z.e(TernaryOps.WHERE,res,v)
def logsumexpB(a:LazyBuffer, b:LazyBuffer):
  mx = MAX(MAX(a,b),Tensor.full(a.shape,-1e10).lazydata)
  ea = EXP(a.e(BinaryOps.SUB, mx))
  eb = EXP(b.e(BinaryOps.SUB, mx))
  s = ea.e(BinaryOps.ADD, eb)
  return LOG(s).e(BinaryOps.ADD, mx)

class LossOp (Function):
  def forward(self, p:LazyBuffer, skip:LazyBuffer, a:LazyBuffer, b:LazyBuffer):

    self.BS,self.X,self.Y = p.shape
    self.A, self.B = [a], [None] * (self.X-1) + [b]

    for x in range(0,self.X-1):
      skipper = self.A[-1].e(BinaryOps.ADD, skip.shrink(((0,self.BS),(x,x+1),(0,self.Y))).reshape((self.BS,self.Y)))
      frw = PAD(self.A[-1].shrink(((0,self.BS),(0,self.Y-1))), ((0,0),(1,0)), -inf)
      frw2 = PAD(p.shrink(((0,self.BS),(x,x+1),(0,self.Y-1))).reshape((self.BS,self.Y-1)),((0,0),(1,0),), -inf)
      self.A.append(logsumexpB(skipper, frw.e(BinaryOps.ADD, frw2)).realsized
                    )

    return (self.A[-1].r(ReduceOps.MAX,(self.BS,1)).r(ReduceOps.SUM,(1,1)).e(UnaryOps.NEG).reshape(()))

  def backward(self, g:LazyBuffer):pass

def LossForward(d:Tensor, labels:Tensor):
  labels = labels.pad(((0,0),(0,1)))
  BS,X,Y,C = d.shape
  lattice = shear(d, 0.)
  X = X+Y-1
  assert lattice.shape == (BS,X,Y,C), f"{lattice.shape}"
  
  p = lattice[Tensor(np.arange(BS).reshape((-1,1,1))),Tensor(np.arange(X).reshape((1,-1,1))),Tensor(np.arange(Y).reshape((1,1,-1))),labels.reshape((BS,1,-1))].log()
  skip = lattice[:,:,:,-1].log()

  a = Tensor([0]*BS).reshape(-1,1).pad(((0,0),(0,Y-1),),-inf).realize()
  b = Tensor.ones(BS,Y)

  L = LossOp.apply(p, skip, a, b)
  return L


BS = 4
maxx = 100
maxy = 140

d = Tensor.rand(BS, maxx, maxy+1, 29).softmax()

d = mask(d,Tensor([maxx]*BS),Tensor([maxy]*BS),maxx,maxy)
d.requires_grad = True
labels = Tensor.randint(BS,maxy,high=29)

T_Loss = TransducerLoss.apply(d, labels)
print ("baseline:",T_Loss.numpy())

T2_Loss = LossForward(d, labels)
print (T2_Loss.numpy())
# %%
#test op idx
t = Tensor.arange(10).lazydata
t = t.shrink(((4,5),))

Tensor(t).numpy()

t = t.pad(((4,5),))
Tensor(t).numpy()


Tensor.ones
#%%
Tensor.manual_seed()
import math

def logsumexp(a:Tensor, b:Tensor):
  mx = Tensor.maximum(a,b).maximum(-1e10)
  s = (a-mx).exp() + (b-mx).exp()
  return s.log() + mx

a,b = Tensor.rand(5), Tensor.rand(5)
print(logsumexp(a,b).numpy())


a,b = a.lazydata, b.lazydata

def MAX(a:LazyBuffer, b:LazyBuffer): return a.e(BinaryOps.CMPLT, b).e(TernaryOps.WHERE, b, a)
def EXP(a:LazyBuffer): return a.e(BinaryOps.MUL, a.const(1/math.log(2))).e(UnaryOps.EXP2)
def LOG(a:LazyBuffer): return a.e(UnaryOps.LOG2).e(BinaryOps.MUL, a.const(math.log(2)))

def logsumexp(a:LazyBuffer, b:LazyBuffer):
  mx = MAX(MAX(a,b),Tensor.full(a.shape,-1e10).lazydata)
  ea = EXP(a.e(BinaryOps.SUB, mx))
  eb = EXP(b.e(BinaryOps.SUB, mx))
  s = ea.e(BinaryOps.ADD, eb)
  return LOG(s).e(BinaryOps.ADD, mx)

print(Tensor(logsumexp(a,b)).numpy())

#%%
x = Tensor(([[1,1],[2,3]])).lazydata
r = x.r(ReduceOps.MAX,(2,1))
Tensor(r).numpy()
