import torch
import torch.nn as nn
from .operations import *


class Cell_ToKeras(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell_ToKeras, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

class Network_ToKeras(nn.Module):

  def __init__(self, C, num_classes, layers, genotype, num_reduction, input_size):
    super(Network_ToKeras, self).__init__()
    num_reduction = int(num_reduction) + 1
    self._layers = layers
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
      cell = Cell_ToKeras(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    if(self.input_size == 224):
       s0 = self.stem0(input)
       s1 = self.stem1(s0) 
    else:
        s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits


