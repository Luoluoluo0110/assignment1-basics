import torch
import torch.nn as nn
import numpy as np

def get_batch(dataset, batch_size, context_length, device):
    N = len(dataset)
    max_start = N - context_length - 1
    starts = np.random.randint(0, max_start+1, batch_size)
    
    inputs_list = []
    targets_list = []
    for i in starts:
        inputs_segment = dataset[i: i+context_length]
        targets_segment = dataset[i+1: i+1+context_length]
        inputs_list.append(inputs_segment)
        targets_list.append(targets_segment)
    inputs = torch.tensor(np.array(inputs_list), dtype=torch.long).to(device)
    targets = torch.tensor(np.array(targets_list), dtype=torch.long).to(device)    
    return inputs, targets