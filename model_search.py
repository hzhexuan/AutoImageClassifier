import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *
from torch.autograd import Variable
from .genotypes import PRIMITIVES, Genotype
import numpy as np

class MixedOp_DARTS(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp_DARTS, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell_DARTS(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
      #multiplier
    super(Cell_DARTS, self).__init__()
    self.reduction = reduction
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp_DARTS(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network_DARTS(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, num_reduction=2, input_size=32):
    super(Network_DARTS, self).__init__()
    self.num_reduction = num_reduction
    num_reduction = int(num_reduction) + 1
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    self.input_size = input_size
    if(self.input_size == 224):
        self.stem0 = nn.Sequential(
          nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(C // 2),
          nn.ReLU(inplace=True),
          nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(C),
        )
    
        self.stem1 = nn.Sequential(
          nn.ReLU(inplace=True),
          nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(C),
        )
        C_prev_prev, C_prev, C_curr = C, C, C
        reduction_prev = True
    else:
        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
          nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
          nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
    
    self.cells = nn.ModuleList()
    for i in range(layers):
      if i in [j * layers//num_reduction for j in range(1, num_reduction)]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell_DARTS(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
    
  def new(self):
    model_new = Network_DARTS(self._C, self._num_classes, self._layers, self._criterion, num_reduction = self.num_reduction, input_size = self.input_size).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    if(self.input_size == 224):
       s0 = self.stem0(input)
       s1 = self.stem1(s0) 
    else:
        s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

class MixedOp_SF(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp_SF, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, operation):
    return self._ops[operation](x)


class Cell_SF(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
      #multiplier
    super(Cell_SF, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp_SF(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, operations):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, operations[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network_SF(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, num_reduction=2, input_size = 32):
    super(Network_SF, self).__init__()
    num_reduction = int(num_reduction) + 1
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.input_size = input_size
    
    if(self.input_size == 224):
        self.stem0 = nn.Sequential(
          nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(C // 2),
          nn.ReLU(inplace=True),
          nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(C),
        )
    
        self.stem1 = nn.Sequential(
          nn.ReLU(inplace=True),
          nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(C),
        )
        C_prev_prev, C_prev, C_curr = C, C, C
        reduction_prev = True
    else:
        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
          nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
          nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
    
    self.cells = nn.ModuleList()
    for i in range(layers):
      if i in [j * layers//num_reduction for j in range(1, num_reduction)]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell_SF(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.mv = 0
    self._initialize_alphas()
    self.num_k = sum(1 for i in range(self._steps) for n in range(2+i))
    self.num_ops = len(PRIMITIVES)

  def forward(self, input, target):
    self.sampling()
    if(self.input_size == 224):
       s0 = self.stem0(input)
       s1 = self.stem1(s0) 
    else:
        s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        operations = self.sample[:14]
      else:
        operations = self.sample[14:]
      s0, s1 = s1, cell(s0, s1, operations)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    loss = self._criterion(logits, target)
    self.mv = 0.9 * self.mv + 0.1 * loss.cpu().data
    grads = (loss - self.mv) * (torch.eye(8).cuda()[self.sample] - self.con)
    return loss, grads, logits

  def sampling(self):
      self.con = F.softmax(torch.cat([self.alphas_reduce, self.alphas_normal], dim = 0), dim = -1)
      self.sample = torch.multinomial(self.con, 1, replacement = True).cuda()[:,0]

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    self.alphas_normal = Variable(torch.log(F.softmax(1e-3*torch.randn(k, num_ops).cuda(), dim = -1)), requires_grad=True)
    self.alphas_reduce = Variable(torch.log(F.softmax(1e-3*torch.randn(k, num_ops).cuda(), dim = -1)), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

class Network_RAM(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, num_reduction=2, input_size = 32):
    super(Network_RAM, self).__init__()
    num_reduction = int(num_reduction) + 1
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.input_size = input_size
    
    if(self.input_size == 224):
        self.stem0 = nn.Sequential(
          nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(C // 2),
          nn.ReLU(inplace=True),
          nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(C),
        )
    
        self.stem1 = nn.Sequential(
          nn.ReLU(inplace=True),
          nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(C),
        )
        C_prev_prev, C_prev, C_curr = C, C, C
        reduction_prev = True
    else:
        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
          nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
          nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
    
    self.cells = nn.ModuleList()
    for i in range(layers):
      if i in [j * layers//num_reduction for j in range(1, num_reduction)]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell_SF(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.num_k = sum(1 for i in range(self._steps) for n in range(2+i))
    self.num_ops = len(PRIMITIVES)
    self._initialize_alphas()
    self.sample = None
    self.paths_proba = None #probability of each path
    self.con = None
    
  def sampling(self):
      with torch.no_grad():
          self.con = F.softmax(torch.cat([self.alphas_reduce, self.alphas_normal], dim = 0), dim = -1)
          self.sample = torch.multinomial(self.con, 2 * self.num_k, replacement = True)
          sample_proba = self.con[range(2 * self.num_k), self.sample[range(2 * self.num_k), range(2 * self.num_k)]]
          self.paths_proba = torch.prod(self.con[np.reshape(range(2 * self.num_k), [-1, 1]), self.sample[range(2 * self.num_k)]], dim = 0) / sample_proba
      
  def forward(self, input):
    self.sampling()
    operations_reduce = self.sample[:self.num_k, 0]
    operations_normal = self.sample[self.num_k:, 0]
    if(self.input_size == 224):
       s0 = self.stem0(input)
       s1 = self.stem1(s0) 
    else:
        s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        operations = operations_reduce
      else:
        operations = operations_normal
      s0, s1 = s1, cell(s0, s1, operations)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def forward_i_op(self, args):
    with torch.no_grad():
        input, target, i, op = args
        operations = self.sample[:, i]
        operations[i] = op
        operations_reduce = operations[:self.num_k]
        operations_normal = operations[self.num_k:]
        if(self.input_size == 224):
           s0 = self.stem0(input)
           s1 = self.stem1(s0) 
        else:
            s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
          if cell.reduction:
            operations = operations_reduce
          else:
            operations = operations_normal
          s0, s1 = s1, cell(s0, s1, operations)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        loss = self._criterion(logits, target)
        return loss

  def forward_i(self, args):
      with torch.no_grad():
          input, target, i = args
          losses = torch.stack(list(map(self.forward_i_op, zip([input]*self.num_ops, [target]*self.num_ops, [i]*self.num_ops, range(self.num_ops)))))
          return torch.reshape(losses, [-1])
    
  def sampleforward(self, input, target):
      with torch.no_grad():
          self.sampling()
          losses = torch.stack(list(map(self.forward_i, zip([input]*2 * self.num_k, [target]*2 * self.num_k, range(2 * self.num_k)))))
          return losses

  def train_alphas(self, input, target):
      with torch.no_grad():
          self.sampling()
          losses = self.sampleforward(input, target)
          q_times_loss = losses * self.con
          return  losses, self.con * (losses - torch.sum(q_times_loss, dim = -1, keepdim = True))

  def _initialize_alphas(self):

    self.alphas_normal = Variable(torch.log(F.softmax(1e-3*torch.randn(self.num_k, self.num_ops).cuda(), dim = -1)), requires_grad=True)
    self.alphas_reduce = Variable(torch.log(F.softmax(1e-3*torch.randn(self.num_k, self.num_ops).cuda(), dim = -1)), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype