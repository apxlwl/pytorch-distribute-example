# pytorchv1.0-distribute-example
This example is mainly based on the [offical tutorial](https://github.com/pytorch/examples/tree/master/imagenet), but more readable.

## Requirements

- Install PyTorch1.0 ([pytorch.org](http://pytorch.org)).
- A machine with multiple gpus or a cluster with multiple nodes to enable distributed learning.

## Training

### Single node, multiple GPUs:
Note: For single node training, the --gpu_use should equal to --world_size 
```bash
python main.py --dist-url 'tcp://127.0.0.1:FREEPORT' --multiprocessing-distributed --rank_start 0 --world-size 2 --gpu_use 2
```

### Multiple nodes:
Take a cluster with 2 nodes(each node has 2 gpus) for example, the rank_start of the first node should be 0 and 0+gpu_use for the second node. 

The world-size should be sum of total gpu usage, since we follow the official recommendation to start each process with one gpu.

Node 0:
```bash
python main.py --dist-url 'tcp://ip:FREEPORT' --node 0 --multiprocessing-distributed --rank_start 0 --world-size 4 --gpu_use 2
```

Node 1:
```bash
python main.py --dist-url 'tcp://ip:FREEPORT' --node 1 --multiprocessing-distributed --rank_start 2 --world-size 4 --gpu_use 2
``` 


## Reference
[official tutorials](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

[official document](https://pytorch.org/docs/stable/distributed.html)
