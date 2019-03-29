import argparse
import os
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from utils import *
import time

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# distribute argument
parser.add_argument('--world-size', default=2, type=int,required=True)
parser.add_argument('--dist-url', default='tcp://yourip:freeport', type=str,required=True)
# parser.add_argument('--dist-url', default='tcp://127.0.0.1:freeport', type=str)
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu_use', default=2, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True)
parser.add_argument('--rank_start', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--node', default=0, type=int, help='which node in your clusters')

# training arggument
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('--workers', default=4, type=int, help='GPU id to use.')

def main():
  args = parser.parse_args()

  args.distributed = args.multiprocessing_distributed
  assert args.distributed and args.world_size>2, "This is an example for distribution training"

  ngpus_per_node = args.gpu_use
  if args.multiprocessing_distributed:
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    #NOTE:the first arg for main_worker is passed by mp.spawn, see source code for more


def main_worker(process_id,ngpus_per_node, args):
  args.gpu = process_id
  model = Net()
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)
  optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.1)

  if args.multiprocessing_distributed:
    args.rank = args.rank_start + process_id
  dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                          world_size=args.world_size, rank=args.rank)
  print("Process {} in node {} has been started.".format(args.rank, args.node))

  torch.cuda.set_device(args.gpu)
  model.cuda(args.gpu)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

  train_sampler = torch.utils.data.distributed.DistributedSampler(get_mnist(istrain=True), num_replicas=None, rank=None)
  test_sampler = torch.utils.data.distributed.DistributedSampler(get_mnist(istrain=False), num_replicas=None, rank=None)

  #different form
  train_loader = torch.utils.data.DataLoader(
    get_mnist(istrain=True), batch_size=args.batch_size, shuffle=None,
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

  test_loader = torch.utils.data.DataLoader(
    get_mnist(istrain=False), batch_size=args.batch_size, shuffle=None,
    num_workers=args.workers, pin_memory=True, sampler=test_sampler)
  acc_train=AverageMeter()
  acc_test=AverageMeter()
  best_acc=0
  for epoch in range(5):
    if args.distributed:
      train_sampler.set_epoch(epoch)
    #train epoch
    for i, (input, target) in enumerate(train_loader):
      input = input.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)
      output = model(input)
      loss = criterion(output, target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      acc=accuracy(output,target)
      acc_train.update(acc[0],n=input.size(0))

    #test epoch
    for i, (input, target) in enumerate(test_loader):
      input = input.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)
      with torch.no_grad():
        output = model(input)
      acc=accuracy(output,target)
      acc_test.update(acc[0],n=input.size(0))

    #the performances differ since different data are used
    print("rank {}th process after epoch{}: train_acc{:.2f},val_acc{:.2f}".format(args.rank,epoch,acc_train.avg.item(),acc_test.avg.item()))
    is_best = acc_test.avg.item() > best_acc
    best_acc = max(acc_test.avg.item(), best_acc)
    acc_train.reset()
    acc_test.reset()

    # save once per node
    if args.rank% args.gpu_use==0:
      save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc,
        'optimizer': optimizer.state_dict(),
      }, is_best)



if __name__ == '__main__':
  main()
