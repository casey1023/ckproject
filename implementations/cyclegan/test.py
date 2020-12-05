import torch
lowest_gpu_usage =torch.cuda.memory_reserved()
print(lowest_gpu_usage)