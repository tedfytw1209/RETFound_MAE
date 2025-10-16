import torch
import torch.nn as nn
import numpy as np
import quantus


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

def to_numpy(x, dtype=np.float32):
    """Convert tensor or array to numpy array.
    
    Args:
        x: Input data (tensor or numpy array)
        dtype: Target numpy dtype
    
    Returns:
        numpy array
    """
    if isinstance(x, np.ndarray):
        arr = x
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported type for numpy conversion: {type(x)}")
    
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr

class SufficiencyMetric():
    def __init__(self, model, device, threshold=0.5, return_aggregate=False):
        r"""Create sufficiency metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            threshold (float): threshold for sufficiency metric.
            return_aggregate (bool): whether to return aggregate score or per-sample scores.
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.return_aggregate = return_aggregate
        self.metric = quantus.Sufficiency(threshold=threshold, return_aggregate=return_aggregate)
        
    def __call__(self, x_batch, a_batch, y_batch = None, **kwargs):
        """Input batch images and explanations, return sufficiency metric.

        Args:
            x_batch (np.ndarray): All Input images. [N, C, H, W]
            a_batch (np.ndarray): All Input explanations. [N, H, W] -> [N, 1, H, W]
            y_batch (np.ndarray): All Input labels. [N]
        Returns:
            float or np.ndarray: sufficiency metric score.
        """
        x_batch = to_numpy(x_batch, dtype=np.float32)
        if y_batch is not None:
            y_batch = to_numpy(y_batch, dtype=np.int64)
        a_batch = to_numpy(a_batch, dtype=np.float32)
        if a_batch.ndim == 3:
            a_batch = np.expand_dims(a_batch, axis=1)
        return self.metric(self.model, x_batch, y_batch, a_batch, self.device)

class ConsistencyMetric():
    def __init__(self, model, device, discretise_func=quantus.discretise_func.top_n_sign, return_aggregate=False):
        r"""Create sufficiency metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            threshold (float): threshold for sufficiency metric.
            return_aggregate (bool): whether to return aggregate score or per-sample scores.
        """
        self.model = model
        self.device = device
        self.discretise_func = discretise_func
        self.return_aggregate = return_aggregate
        self.metric = quantus.Consistency(discretise_func=discretise_func, return_aggregate=return_aggregate)
        
    def __call__(self, x_batch, a_batch = None, y_batch = None, explain_func=quantus.explain, explain_func_kwargs={"method": "Saliency"}, **kwargs):
        """Input batch images and explanations, return sufficiency metric.

        Args:
            x_batch (np.ndarray): All Input images. [N, C, H, W]
            a_batch (np.ndarray): All Input explanations. [N, H, W] -> [N, 1, H, W]
            y_batch (np.ndarray): All Input labels. [N]
        Returns:
            float or np.ndarray: consistency metric score.
        """
        x_batch = to_numpy(x_batch, dtype=np.float32)
        if y_batch is not None:
            y_batch = to_numpy(y_batch, dtype=np.int64)
        if a_batch is not None:
            a_batch = to_numpy(a_batch, dtype=np.float32)
            if a_batch.ndim == 3:
                a_batch = np.expand_dims(a_batch, axis=1)
        result = self.metric(model=self.model, x_batch=x_batch, y_batch=y_batch, a_batch=None, 
                    explain_func=explain_func, explain_func_kwargs=explain_func_kwargs, device=self.device)
        
        return result

class PointingGameMetric():
    def __init__(self, model, device):
        r"""Create sufficiency metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            threshold (float): threshold for sufficiency metric.
            return_aggregate (bool): whether to return aggregate score or per-sample scores.
        """
        self.model = model
        self.device = device
        self.metric = quantus.PointingGame()
        
    def __call__(self, x_batch, a_batch = None, y_batch = None, explain_func=quantus.explain, explain_func_kwargs={"method": "Saliency"}, **kwargs):
        """Input batch images and explanations, return sufficiency metric.

        Args:
            x_batch (np.ndarray): All Input images. [N, C, H, W]
            a_batch (np.ndarray): All Input explanations. [N, H, W] -> [N, 1, H, W]
            y_batch (np.ndarray): All Input labels. [N]
        Returns:
            float or np.ndarray: consistency metric score.
        """
        x_batch = to_numpy(x_batch, dtype=np.float32)
        if y_batch is not None:
            y_batch = to_numpy(y_batch, dtype=np.int64)
        if a_batch is not None:
            a_batch = to_numpy(a_batch, dtype=np.float32)
            if a_batch.ndim == 3:
                a_batch = np.expand_dims(a_batch, axis=1)
        #TODO: Check difference between a_batch and s_batch
        result = self.metric(model=self.model, x_batch=x_batch, y_batch=y_batch, a_batch=None, s_batch=a_batch, explain_func=explain_func,    explain_func_kwargs=explain_func_kwargs, device=self.device)
        
        return result

class ComplexityMetric():
    def __init__(self, model, device):
        r"""Create sufficiency metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            threshold (float): threshold for sufficiency metric.
            return_aggregate (bool): whether to return aggregate score or per-sample scores.
        """
        self.model = model
        self.device = device
        self.metric = quantus.Complexity()
        
    def __call__(self, x_batch, a_batch = None, y_batch = None, explain_func=quantus.explain, explain_func_kwargs={"method": "Saliency"}, **kwargs):
        """Input batch images and explanations, return sufficiency metric.

        Args:
            x_batch (np.ndarray): All Input images. [N, C, H, W]
            a_batch (np.ndarray): All Input explanations. [N, H, W] -> [N, 1, H, W]
            y_batch (np.ndarray): All Input labels. [N]
        Returns:
            float or np.ndarray: consistency metric score.
        """
        x_batch = to_numpy(x_batch, dtype=np.float32)
        if y_batch is not None:
            y_batch = to_numpy(y_batch, dtype=np.int64)
        if a_batch is not None:
            a_batch = to_numpy(a_batch, dtype=np.float32)
            if a_batch.ndim == 3:
                a_batch = np.expand_dims(a_batch, axis=1)
        result = self.metric(model=self.model, x_batch=x_batch, y_batch=y_batch, a_batch=None, explain_func=explain_func,    explain_func_kwargs=explain_func_kwargs, device=self.device)
        
        return result

class RandomLogitMetric():
    def __init__(self, model, device, n_classes):
        r"""Create sufficiency metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            threshold (float): threshold for sufficiency metric.
            return_aggregate (bool): whether to return aggregate score or per-sample scores.
        """
        self.model = model
        self.device = device
        self.metric = quantus.RandomLogit(num_classes=n_classes, similarity_func=quantus.similarity_func.ssim)
        
    def __call__(self, x_batch, a_batch = None, y_batch = None, explain_func=quantus.explain, explain_func_kwargs={"method": "Saliency"}, **kwargs):
        """Input batch images and explanations, return sufficiency metric.

        Args:
            x_batch (np.ndarray): All Input images. [N, C, H, W]
            a_batch (np.ndarray): All Input explanations. [N, H, W] -> [N, 1, H, W]
            y_batch (np.ndarray): All Input labels. [N]
        Returns:
            float or np.ndarray: consistency metric score.
        """
        x_batch = to_numpy(x_batch, dtype=np.float32)
        if y_batch is not None:
            y_batch = to_numpy(y_batch, dtype=np.int64) 
        if a_batch is not None:
            a_batch = to_numpy(a_batch, dtype=np.float32)
            if a_batch.ndim == 3:
                a_batch = np.expand_dims(a_batch, axis=1)
        result = self.metric(model=self.model, x_batch=x_batch, y_batch=y_batch, a_batch=None, explain_func=explain_func,    explain_func_kwargs=explain_func_kwargs, device=self.device)
        
        return result