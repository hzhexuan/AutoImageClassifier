import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import os
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from torch.autograd import Variable
from .model_search import Network_DARTS, Network_SF, Network_RAM
from .model import Network
from .model_ToKeras import Network_ToKeras
from .architect import Architect
from .utils import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MyDataset(Dataset):
    
    def __init__(self, data, target, transform=None):
        super(MyDataset, self).__init__()
        self.data = np.uint8(data)
        self.target = torch.Tensor(np.argmax(target, axis = 1)).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon, frequence):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)
    self.FREC = torch.Tensor(frequence).cuda()
    
  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs / self.num_classes / self.FREC).mean(0).sum()
    return loss

class ImageClassifier():
    
    def __init__(self, train_input, train_target, test_input, test_target,
                label_smooth=0, cutout = 16, save="EXP"):
        #input images are supposed to be [num_samples, H, W, C]
        #targets are supposed to be one-hot, i.e.[num_samples, num_classes]
        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input
        self.test_target = test_target
          
        self.mean = np.mean(self.train_input, axis = (0, 1, 2))
        self.std = np.std(self.train_input, axis = (0, 1, 2))
        self.frequence = classfrequence(self.train_target)
        self.num_class = self.train_target.shape[-1]
        self.criterion = CrossEntropyLabelSmooth(self.num_class, label_smooth, self.frequence)
        self.cutout = cutout
        #create save dir
        self.save = 'search-{}-{}'.format(save, time.strftime("%Y%m%d-%H%M%S"))
        create_exp_dir(self.save, scripts_to_save=glob.glob('*.py'))
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        self.genotype = None
        self.finalfit_init_channels = None
        self.finalfit_layers = None
        self.num_reduction = None
        self.num_layers_search = None
        self.init_channels_search = None
        self.image_size = self.train_input.shape[2]
        print("Image size is:", self.image_size)
        print("Mean value is:", self.mean)
        print("std is:", self.std)
    
    def transform_train(self, input):
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(self.image_size, padding=int(self.image_size/8)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(self.mean/255, self.std/255),])
        if (self.cutout > 0):
            transform.transforms.append(Cutout(self.cutout))
        return transform(input)

    def transform_test(self, input):
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(self.mean/255, self.std/255),])
        return transform(input)
      
    def fit(self, seed=0, gpu=0, lr=0.025, weight_decay=3e-4, momentum=0.9, 
            batch_size=64, epochs=50, learning_rate_min=0.001, arch_learning_rate=3e-4,
            arch_weight_decay=1e-3, unrolled=True, report_freq=50, grad_clip=5, 
            method="DARTS", num_reduction=2, init_channels_search=16, num_layers_search = 8):
      self.init_channels_search = init_channels_search
      self.num_layers_search = num_layers_search
      if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

      np.random.seed(seed)
      torch.cuda.set_device(gpu)
      cudnn.benchmark = True
      torch.manual_seed(seed)
      cudnn.enabled=True
      torch.cuda.manual_seed(seed)
      logging.info('gpu device = %d' % gpu)
      criterion = self.criterion.cuda()
      
      train_data = MyDataset(self.train_input, self.train_target, self.transform_train)
      test_data = MyDataset(self.test_input, self.test_target, self.transform_test)
      
      if(method == "DARTS"):
          model = Network_DARTS(self.init_channels_search, self.num_class, self.num_layers_search, criterion, num_reduction = num_reduction)
          model = model.cuda()
          logging.info("param size = %fMB", count_parameters_in_MB(model))
        
          optimizer = torch.optim.SGD(
              model.parameters(),
              lr,
              momentum=momentum,
              weight_decay=weight_decay)
        
          num_train = len(train_data)
          indices = list(range(num_train))
          split = int(np.floor(0.5 * num_train))
        
          train_queue = torch.utils.data.DataLoader(
              train_data, batch_size=batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
              pin_memory=True, num_workers=2)
        
          valid_queue = torch.utils.data.DataLoader(
              train_data, batch_size=batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
              pin_memory=True, num_workers=2)
        
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(epochs), eta_min=learning_rate_min)
        
          architect = Architect(model, momentum, weight_decay, arch_learning_rate, arch_weight_decay)
        
          for epoch in range(epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
        
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)
        
            print(F.softmax(model.alphas_normal, dim=-1))
            print(F.softmax(model.alphas_reduce, dim=-1))  
            # training
            train_acc, train_obj = train_DARTS(train_queue, valid_queue, model, architect,
                                         criterion, optimizer, lr, unrolled, report_freq,
                                         grad_clip)
            logging.info('train_acc %f', train_acc)
        
            # validation
            with torch.no_grad():
                valid_acc, valid_obj = infer_DARTS(valid_queue, model, criterion, report_freq)
            logging.info('valid_acc %f', valid_acc)
            save(model, os.path.join(self.save, 'weights.pt'))
            
      elif(method == "SF"):
          num_train = len(train_data)
          indices = list(range(num_train))
          split = int(np.floor(0.5 * num_train))
        
          train_queue = torch.utils.data.DataLoader(
              train_data, batch_size=batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
              pin_memory=True, num_workers=2)
        
          valid_queue = torch.utils.data.DataLoader(
              train_data, batch_size=batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
              pin_memory=True, num_workers=2)
          
          test_queue = torch.utils.data.DataLoader(
              test_data, batch_size=batch_size, shuffle=False,
              pin_memory=True, num_workers=2)
          
          model = Network_SF(self.init_channels_search, self.num_class, 
                                          self.num_layers_search, criterion, num_reduction = num_reduction)
          model.cuda()
          logging.info("param size = %fMB", count_parameters_in_MB(model))
          
          optimizer = torch.optim.SGD(
              model.parameters(),
              lr,
              momentum=momentum,
              weight_decay=weight_decay)
          optimizer_arc = torch.optim.Adam(model.arch_parameters(), 
                                           arch_learning_rate)
          for epoch in range(epochs):
            logging.info('epoch %d', epoch)
            
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)
            
            print(F.softmax(model.alphas_normal, dim = -1))
            print(F.softmax(model.alphas_reduce, dim = -1))
            # training
            train_acc, train_obj = train_SF(train_queue, valid_queue, model, criterion, 
                                         optimizer, optimizer_arc, report_freq, grad_clip)
            logging.info('train_acc %f %f', train_obj, train_acc)
            # validation
            with torch.no_grad():
                valid_acc, valid_obj = infer_SF(test_queue, model, criterion, report_freq)
            logging.info('valid_acc %f %f', valid_obj, valid_acc)
      
      elif(method == "RAM"):
          
          num_train = len(train_data)
          indices = list(range(num_train))
          split = int(np.floor(0.5 * num_train))
        
          train_queue = torch.utils.data.DataLoader(
              train_data, batch_size=batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
              pin_memory=True, num_workers=2)
        
          valid_queue = torch.utils.data.DataLoader(
              train_data, batch_size=batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
              pin_memory=True, num_workers=2)
          
          test_queue = torch.utils.data.DataLoader(
              test_data, batch_size=batch_size, shuffle=False,
              pin_memory=True, num_workers=2)
          
          model = Network_RAM(self.init_channels_search, self.num_class, 
                                          self.num_layers_search, criterion, num_reduction = num_reduction)
          model.cuda()
          logging.info("param size = %fMB", count_parameters_in_MB(model))
          
          optimizer = torch.optim.SGD(
              model.parameters(),
              lr,
              momentum=momentum,
              weight_decay=weight_decay)
          optimizer_arc = torch.optim.Adam(model.arch_parameters(), 
                                           arch_learning_rate)
          for epoch in range(epochs):
            logging.info('epoch %d', epoch)
            
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)
            
            print(F.softmax(model.alphas_normal, dim = -1))
            print(F.softmax(model.alphas_reduce, dim = -1))
            # training
            train_acc, train_obj = train_RAM(train_queue, valid_queue, model, criterion, 
                                         optimizer, optimizer_arc, report_freq, grad_clip)
            logging.info('train_acc %f %f', train_obj, train_acc)
            # validation
            with torch.no_grad():
                valid_acc, valid_obj = infer_RAM(test_queue, model, criterion, report_freq)
            logging.info('valid_acc %f %f', valid_obj, valid_acc) 
      
      else:
          raise Exception("Invalid method!")
          
      self.genotype = model.genotype()
      logging.info('Final genotype = %s', self.genotype)
      torch.cuda.empty_cache()
    
    def finalfit(self, genotype=None, seed=0, gpu=0, init_channels=36, layers=20, 
                  lr=0.025, momentum=0.9, weight_decay=3e-4, batch_size=96, 
                  epochs=600, drop_path_prob=0.2, auxiliary=True, 
                  auxiliary_weight=0.4, grad_clip=5, report_freq=50, num_reduction=2):
      if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
      
      if(genotype == None and self.genotype == None):
          raise Exception("genotype is not found in Imageclassifier, please run Imageclassifier.fit() first or specify a genotype")
      elif(genotype == None):
          genotype = self.genotype
      else:
          self.genotype = genotype
      self.finalfit_init_channels = init_channels
      self.finalfit_layers = layers
      self.num_reduction = num_reduction
      
      np.random.seed(seed)
      torch.cuda.set_device(gpu)
      cudnn.benchmark = True
      torch.manual_seed(seed)
      cudnn.enabled=True
      torch.cuda.manual_seed(seed)
      logging.info('gpu device = %d' % gpu)
      
      if(self.image_size != 32 or num_reduction != 3):
          auxiliary = False
      model = Network(self.finalfit_init_channels, self.num_class, self.finalfit_layers, auxiliary, genotype, num_reduction = self.num_reduction)
      model = model.cuda()
      logging.info("param size = %fMB", count_parameters_in_MB(model))
      criterion = self.criterion.cuda()
      optimizer = torch.optim.SGD(
          model.parameters(),
          lr,
          momentum=momentum,
          weight_decay=weight_decay
          )
      
      train_data = MyDataset(self.train_input, self.train_target, self.transform_train)
      test_data = MyDataset(self.test_input, self.test_target, self.transform_test)
    
      train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    
      valid_queue = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))
      
      best_recall = 0
      for epoch in range(epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = drop_path_prob * epoch / epochs
        train_acc, train_obj = finalfit_train(train_queue, model, criterion, 
                                     optimizer, auxiliary, auxiliary_weight, grad_clip, report_freq)
        logging.info('train_acc %f', train_acc)
        with torch.no_grad():
            model.drop_path_prob = 0
            precision, recall, confusion_matrix, valid_acc, valid_obj = finalfit_infer(valid_queue, 
                                                                                       model, criterion, 
                                                                                       report_freq)
        logging.info('valid_acc, valid_precision, valid_recall: %f %f %f', 
                     valid_acc, precision, recall)
        logging.info('best_recall %f', best_recall)
        
        is_best = False
        if recall > best_recall:
          best_recall = recall
          is_best = True
          np.savetxt(self.save + "/confusion_matrix.txt", np.uint8(confusion_matrix), fmt = "%i")
    
      save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_recall': best_recall,
        'optimizer' : optimizer.state_dict(),
        }, is_best, self.save)
    
      torch.cuda.empty_cache()
    
    def path(self):
        return self.save
    
    def genotype(self):
        return self.genotype
    
    def finalfit_init_channels(self):
        return self.finalfit_init_channels
    
    def finalfit_layers(self):
        return self.layers
    
    def ToKeras(self, output_name, path=None, genotype=None, init_channels=None, layers=None):
        
        if(path == None):
            path = self.path() + "/model_best.pth.tar"
            if(self.finalfit_init_channels == None or self.finalfit_layers == None or self.genotype == None):
                raise Exception("Can not find finalfit_init_channels, finalfit_layers or genotype in ImageClassifier, run ImageClassifier.finalfit() first")
            init_channels = self.finalfit_init_channels
            layers = self.finalfit_layers
            if(self.genotype == None):
                raise Exception("Can not find genotype, run Imageclassifier.fit() first.")
            genotype = self.genotype
        elif(init_channels == None or layers == None or genotype == None):
            raise Exception("Should indicate init_channels, layers and genotype when a specific model path is given")
        
        model = Network_ToKeras(init_channels, self.num_class, layers, genotype=genotype, num_reduction=self.num_reduction)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict = False)
        model.eval()
        
        input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
        input_var = torch.FloatTensor(input_np)
        output = model(input_var)
        
        torch.onnx.export(model, (input_var), output_name + ".onnx",
                          verbose=True,
                          input_names=['input'],
                          output_names=['output']
        )
        import onnx
        from onnx2keras import onnx_to_keras
        onnx_model = onnx.load(output_name + ".onnx")
        k_model = onnx_to_keras(onnx_model, ['input'])
        k_model.save(output_name + ".h5")
    
    
def train_DARTS(train_queue, valid_queue, model, architect, criterion, optimizer, lr, unrolled, report_freq, grad_clip):
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def infer_DARTS(valid_queue, model, criterion, report_freq):
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def train_SF(train_queue, valid_queue, model, criterion, optimizer, optimizer_arc, report_freq, grad_clip):
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()
  model.train()
  for step, (input, target) in enumerate(train_queue):
    optimizer.zero_grad()
    optimizer_arc.zero_grad()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    #update arc parameters
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)
    with torch.no_grad():
        loss, grads, _ = model(input_search, target_search)
        grads_reduce = grads[:model.num_k]
        grads_normal = grads[model.num_k:]
        model.alphas_reduce.grad = Variable(grads_reduce)
        model.alphas_normal.grad = Variable(grads_normal)

    optimizer_arc.step()
    model.alphas_reduce.data = torch.log(F.softmax(model.alphas_reduce.data, dim = -1))
    model.alphas_normal.data = torch.log(F.softmax(model.alphas_normal.data, dim = -1))
    
    #update network parameters
    loss, grads, logits = model(input, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  return top1.avg, objs.avg

def infer_SF(valid_queue, model, criterion, report_freq):
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    loss, _, logits = model(input, target)

    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def train_RAM(train_queue, valid_queue, model, criterion, optimizer, optimizer_arc, report_freq, grad_clip):
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()
  model.train()
  
  for step, (input, target) in enumerate(train_queue):
    optimizer.zero_grad()
    optimizer_arc.zero_grad()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)
    
    #update arc parameters
    if(step % 16 == 0):
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(async=True)
        with torch.no_grad():
            loss, grads = model.train_alphas(input_search, target_search)
            grads_reduce = grads[:model.num_k]
            grads_normal = grads[model.num_k:]
            model.alphas_reduce.grad = Variable(grads_reduce)
            model.alphas_normal.grad = Variable(grads_normal)
        optimizer_arc.step()
        model.alphas_reduce.data = torch.log(F.softmax(model.alphas_reduce.data, dim = -1))
        model.alphas_normal.data = torch.log(F.softmax(model.alphas_normal.data, dim = -1))


    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  return top1.avg, objs.avg

def infer_RAM(valid_queue, model, criterion, report_freq):
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def finalfit_train(train_queue, model, criterion, optimizer, auxiliary, auxiliary_weight, grad_clip, report_freq):
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += auxiliary_weight*loss_aux
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def finalfit_infer(valid_queue, model, criterion, report_freq):
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()
  model.eval()
  y_true = np.zeros(1)
  y_pred = np.zeros(1)
  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)
    y_true = np.concatenate([y_true, target.data.cpu().numpy()])
    logits, _ = model(input)
    y_pred = np.concatenate([y_pred, np.argmax(logits.data.cpu().numpy(), axis = -1)])
    loss = criterion(logits, target)

    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return 100 * precision_score(y_true[1:], y_pred[1:], average = "macro"), 100 * recall_score(y_true[1:], y_pred[1:], average = "macro"), confusion_matrix(y_true[1:], y_pred[1:]), top1.avg, objs.avg
