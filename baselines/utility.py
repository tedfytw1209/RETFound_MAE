import torch
import numpy as np

def to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif torch.is_tensor(x):
        t = x
    else:
        raise TypeError(f"Unsupported type for tensor conversion: {type(x)}")
    if t.dtype != dtype:
        t = t.to(dtype)
    return t.to(device)