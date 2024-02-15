#%%
from utils import imshow
from train import *

#%%
class Loss:

  def __init__(self,d:Tensor, labels:Tensor):
    Tensor.no_grad = True

    self.d = d
    self.B,self.X,self.Y,self.C = d.shape

    self.labels = labels.pad(((0,0),(0,1)))
    self.lattice = shear(d, 0.)
    self.X = self.X+self.Y-1
    assert self.lattice.shape == (self.B,self.X,self.Y,self.C), f"{self.lattice.shape}"

    self.skip = shear(d[:,:,:,-1:],1.)[:,:,:,0].log()

    self.p = self.lattice[
      Tensor.arange(self.B).reshape((-1,1,1)),
      Tensor.arange(self.X).reshape((1,-1,1)),
      Tensor.arange(self.Y).reshape((1,1,-1)),
      self.labels.reshape((self.B,1,-1))].log()

    assert self.p.shape == (self.B, self.X, self.Y)
    self.a = [Tensor.zeros(self.B).reshape(-1,1).pad(((0,0),(0,self.Y-1),),-inf).realize()]

    for x in range(0,self.X-1):
      self.a.append(logsumexp(
        (self.a[-1] + self.skip[:,x,:]).realize(),
        (
          self.a[-1][:,:-1].pad(((0,0),(1,0),),-inf).realize() + self.p[:,x,:-1].pad(((0,0),(1,0),),-inf)
        ).realize()
      ))
    self.value = -self.a[-1].max(1).sum().numpy()

  def backward(self):
    Tensor.no_grad = True
    self.b: list[None | Tensor] = [None] * (self.X-1) + [Tensor.ones(self.B,self.Y)]
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
    grad = - self.p_grad.cat(self.skg.unsqueeze(-1), dim=-1).pad(((0,0),(0,1),(0,0),(0,0)))

    Tensor.no_grad = False

    # inject loss gradient into autograd
    loss = (self.d - self.d.detach() + grad).square().sum() / 2
    loss.backward()
    imshow(self.d.grad[0,:,:,-1])


BS = 4
maxx = 10
maxy = 14

Tensor.manual_seed()

d = Tensor.rand(BS, maxx, maxy+1, 29).softmax()
labels = Tensor.randint(BS,maxy,high=28)
d = mask(d,Tensor([maxx]*BS),Tensor([maxy]*BS),maxx,maxy)
d.requires_grad = True

L = Loss(d,labels)
L.value
L.backward()

#%%
imshow(L.d.grad.numpy()[0,:,:,28])
