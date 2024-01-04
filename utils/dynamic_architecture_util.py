import numpy as np
import torch


def extend_array(input_buffer, output_size): # used for dynamic architecture, kept in for possible future use
    examples = []
    for example in input_buffer:
        res = np.zeros(output_size, dtype=np.float32)
        res[:len(example)] = example
        res = torch.tensor(res, dtype=torch.float32)
        examples.append(res)
    return torch.stack(examples)