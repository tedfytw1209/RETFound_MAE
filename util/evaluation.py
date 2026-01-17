import sys
import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets

from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from typing import Optional
import torch.nn.functional as F

def _layer_importances_from_mask_and_heatmap(
    seg_mask: np.ndarray,
    heatmap_2d: np.ndarray,
    *,
    ignore_background: bool = True,
) -> np.ndarray:
    """
    Compute per-layer (per-label) importance scores by summing saliency values inside each region.

    Args:
        seg_mask: int-like array of shape (H, W). Labels represent layers/regions; background is label 0.
        heatmap_2d: non-negative array of shape (H, W).
        ignore_background: if True, excludes label 0 from the returned score vector.

    Returns:
        scores: 1D float array of per-label summed saliency, sorted by label id ascending (optionally skipping 0).
    """
    seg_mask = np.asarray(seg_mask)
    heatmap_2d = np.asarray(heatmap_2d, dtype=np.float64)
    if seg_mask.shape != heatmap_2d.shape:
        raise ValueError(f"seg_mask shape {seg_mask.shape} must match heatmap shape {heatmap_2d.shape}")

    # Force non-negative saliency mass
    heatmap_2d = np.where(heatmap_2d > 0.0, heatmap_2d, 0.0)

    labels = np.unique(seg_mask)
    if ignore_background:
        labels = labels[labels != 0]

    if labels.size == 0:
        return np.zeros((0,), dtype=np.float64)

    scores = np.zeros((labels.size,), dtype=np.float64)
    for i, lab in enumerate(labels):
        scores[i] = heatmap_2d[seg_mask == lab].sum()
    return scores

def shannon_entropy(scores: np.ndarray, eps: float = 1e-12) -> float:
    """
    Shannon entropy on normalized non-negative scores p_i = s_i / sum(s).
    Low entropy => focused importance across few layers.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return 0.0
    s = np.where(s > 0.0, s, 0.0)
    total = float(s.sum())
    if total <= 0.0:
        return 0.0
    p = s / (total + eps)
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum())

def gini_coefficient(scores: np.ndarray, eps: float = 1e-12) -> float:
    """
    Gini coefficient for non-negative scores.
    Close to 1 => few layers carry most importance.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x = np.where(x > 0.0, x, 0.0)
    s = float(x.sum())
    if s <= 0.0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    # Efficient Gini: (2*sum_i i*x_i)/(n*sum_x) - (n+1)/n where i is 1..n
    i = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * (i * x_sorted).sum()) / (n * (s + eps)) - (n + 1.0) / n
    return float(np.clip(g, 0.0, 1.0))

def dispersion_cv(scores: np.ndarray, eps: float = 1e-12) -> float:
    """
    Dispersion defined as coefficient of variation: std(scores)/mean(scores).
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        return 0.0
    mean = float(x.mean())
    if abs(mean) <= eps:
        return 0.0
    std = float(x.std(ddof=0))
    return float(std / (mean + eps))

def topk_ratio(scores: np.ndarray, k: int = 3, eps: float = 1e-12) -> float:
    """
    Ratio of top-k score mass over total mass.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x = np.where(x > 0.0, x, 0.0)
    total = float(x.sum())
    if total <= 0.0:
        return 0.0
    kk = int(min(max(k, 1), x.size))
    top = float(np.sort(x)[-kk:].sum())
    return float(top / (total + eps))

def _auc_trapz_update(prev_x, prev_y, x, y):
    return 0.5 * (y + prev_y) * (x - prev_x)
'''
From RISE (https://github.com/eclique/RISE)
@inproceedings{Petsiuk2018rise,
  title = {RISE: Randomized Input Sampling for Explanation of Black-box Models},
  author = {Vitali Petsiuk and Abir Das and Kate Saenko},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year = {2018}
}
'''

# Dummy class to store arguments
class Dummy():
    pass


# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])


# Image preprocessing function
preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Normalization for ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)

#HW = 224 * 224 # image area
#n_classes = 1000

def _gaussian2d(klen: int, ksig: float, device, dtype):
    ax = torch.arange(klen, device=device, dtype=dtype) - (klen - 1) / 2
    g1 = torch.exp(-(ax**2) / (2 * ksig**2))
    g1 = g1 / g1.sum()
    g2 = torch.outer(g1, g1)
    g2 = g2 / g2.sum()
    return g2[None, None, :, :]  # [1,1,k,k]

# cache by (device, dtype, klen, ksig, C)
_BLUR_CACHE = {}

def _gaussian_blur_like(x: torch.Tensor, klen: int, ksig: float) -> torch.Tensor:
    """
    Depthwise Gaussian blur matching x's device/dtype/channels.
    Always builds weight of shape [C,1,k,k].
    """
    C = x.shape[1]
    key = (x.device, x.dtype, klen, ksig, C)
    weight = _BLUR_CACHE.get(key)

    if weight is None:
        base = _gaussian2d(klen, ksig, x.device, x.dtype)  # [1,1,k,k]
        # Ensure final shape [C,1,k,k]
        weight = base.repeat(C, 1, 1, 1).contiguous()
        _BLUR_CACHE[key] = weight

    # conv2d supports channels-last input; make contiguous to be safe
    return F.conv2d(x.contiguous(), weight, padding=klen // 2, groups=C)

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn, img_size=224, n_classes=2):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.img_size = img_size
        self.n_classes = n_classes

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        pred = self.model(img_tensor.cuda())
        if hasattr(pred, 'logits'):
            pred = pred.logits
        else:
            pred = pred
        probs = torch.softmax(pred, dim=1)
        top, c = torch.max(probs, 1)
        c = c.cpu().numpy()[0]
        n_steps = (self.img_size*self.img_size + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, self.img_size*self.img_size), axis=1), axis=-1)
        for i in range(n_steps+1):
            pred = self.model(start.cuda())
            if hasattr(pred, 'logits'):
                pred = pred.logits
            else:
                pred = pred
            probs = torch.softmax(pred, dim=1)
            pr, cl = torch.topk(probs, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = probs[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, self.img_size*self.img_size)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, self.img_size*self.img_size)[0, :, coords]
        return scores


    #old evaluate function for backward compatibility
    @torch.no_grad()
    def evaluate_old(self, img_batch: torch.Tensor, exp_batch: np.ndarray, batch_size: int):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images. [N, C, H, W]
            exp_batch (np.ndarray): batch of explanations. [N, H, W]
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        self.model.eval()
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, self.n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            output = self.model(img_batch[i*batch_size:(i+1)*batch_size].cuda())
            if hasattr(output, 'logits'):
                preds = output.logits.cpu().detach()
            else:
                preds = output.cpu().detach()
            probs = torch.softmax(preds, dim=1)
            predictions[i*batch_size:(i+1)*batch_size] = probs
        img_batch = img_batch.cpu().float()
        top = np.argmax(predictions.numpy(), -1)
        n_steps = (self.img_size*self.img_size + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        sort_order = np.argsort(exp_batch.reshape(-1, self.img_size*self.img_size), axis=1)
        print('sort_order.shape', sort_order.shape)
        salient_order = np.flip(sort_order, axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(img_batch[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                output = self.model(start[j*batch_size:(j+1)*batch_size].cuda())
                if hasattr(output, 'logits'):
                    preds = output.logits
                elif isinstance(output, dict) and 'logits' in output:
                    preds = output['logits']
                else:
                    preds = output
                probs = torch.softmax(preds, dim=1)
                probs = probs.detach().cpu().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = probs
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, self.img_size*self.img_size)[r, :, coords] = finish.detach().cpu().numpy().reshape(n_samples, 3, self.img_size*self.img_size)[r, :, coords]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores

    # new evaluate function for better memory efficiency (but need to be tested)
    @torch.no_grad()
    def evaluate(
        self,
        img_batch: torch.Tensor,         # [N,C,H,W] on CPU or GPU
        exp_batch: np.ndarray,           # [N,H,W], numpy
        batch_size: int,
        use_amp: bool = True,
        device: Optional[torch.device] = None,
        channels_last: bool = True,
        block_step: int = 1,             # >1 means step by blocks of (block_step x block_step) pixels
    ) -> np.ndarray:
        """
        Memory-efficient evaluation of insertion/deletion AUC per sample.
        - Streams AUC (no huge score tensor).
        - Processes mini-batches only.
        - Optional pixel-block stepping to reduce steps.
        Returns:
            auc_per_sample: float array of shape (N,)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        N, C, H, W = img_batch.shape
        if batch_size < 0:
            batch_size = N
        exp_np = np.asarray(exp_batch, dtype=np.float32)

        # Optional channels-last for better mem bandwidth
        if channels_last:
            img_batch = img_batch.to(memory_format=torch.channels_last)

        # Build pixel order (descending saliency); allow block stepping
        if block_step > 1:
            # downsample explanation to blocks and rank blocks
            hh, ww = H // block_step, W // block_step
            exp_down = exp_np.reshape(N, hh, block_step, ww, block_step).mean(axis=(2,4))
            sort_block = np.argsort(exp_down.reshape(N, -1), axis=1)[:, ::-1]  # [N, hh*ww]
            # map blocks to pixel indices lazily per step
            def block_indices_for(sample_idx, lo, hi):
                # get blocks in [lo:hi) and expand to pixels
                blk_ids = sort_block[sample_idx, lo:hi]          # [K]
                by = blk_ids // ww
                bx = blk_ids % ww
                y0 = (by * block_step)[:, None] + np.arange(block_step)[None, :]
                x0 = (bx * block_step)[:, None] + np.arange(block_step)[None, :]
                yy = y0.reshape(-1)
                xx = x0.reshape(-1)
                return (yy[:, None] * W + xx[None, :]).reshape(-1)
            num_units = (H // block_step) * (W // block_step)
            step_unit = max(1, self.step // (block_step * block_step))
            n_steps = max(1, (num_units + step_unit - 1) // step_unit)
        else:
            sort_order = np.argsort(exp_np.reshape(N, -1), axis=1)   # [N, H*W]
            sort_order = np.flip(sort_order, axis=-1).copy()
            num_units = H * W
            step_unit = max(1, self.step)
            n_steps = max(1, (num_units + step_unit - 1) // step_unit)

        # Compute top class per sample in chunks
        top_classes = np.empty((N,), dtype=np.int64)
        with torch.inference_mode():
            for s in range(0, N, batch_size):
                e = min(s + batch_size, N)
                x = img_batch[s:e].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = self.model(x)
                    if hasattr(out, 'logits'):
                        logits = out.logits
                    elif isinstance(out, dict) and 'logits' in out:
                        logits = out['logits']
                    else:
                        logits = out
                    probs = torch.softmax(logits, dim=1)
                top_classes[s:e] = probs.argmax(dim=1).cpu().numpy()
                del x, out, logits, probs
            torch.cuda.empty_cache()

        # AUC accumulators
        auc_acc = np.zeros((N,), dtype=np.float64)
        prev_x = np.zeros((N,), dtype=np.float64)
        prev_y = np.zeros((N,), dtype=np.float64)

        def eval_and_accumulate(start_cpu: torch.Tensor, s: int, e: int, step_idx: int):
            x = start_cpu.to(device, non_blocking=True)
            if channels_last:
                x = x.to(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
                out = self.model(x)
                if hasattr(out, 'logits'):
                    logits = out.logits
                elif isinstance(out, dict) and 'logits' in out:
                    logits = out['logits']
                else:
                    logits = out
                logits = logits.float()
                logits = torch.clamp(logits, -50.0, 50.0)
                probs = torch.softmax(logits, dim=1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                idx = torch.from_numpy(top_classes[s:e]).to(device, dtype=torch.long)
                y = probs.gather(1, idx.unsqueeze(1)).squeeze(1)
            y_np = y.float().cpu().numpy().astype(np.float64)
            frac = float(step_idx) / float(n_steps)
            auc_acc[s:e] += 0.5 * (y_np + prev_y[s:e]) * (frac - prev_x[s:e])
            prev_x[s:e] = frac
            prev_y[s:e] = y_np
            del x, out, logits, probs, idx, y
            torch.cuda.empty_cache()

        # process mini-batches, mutate start in-place
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            B = e - s

            start = img_batch[s:e].cpu().float().clone()
            if callable(self.substrate_fn):
                with torch.inference_mode():
                    sub = self.substrate_fn(start.to(device))
                finish = sub.detach().cpu().to(start.dtype)
                del sub
            elif isinstance(self.substrate_fn, torch.Tensor):
                finish = self.substrate_fn.expand_as(start).clone()
            else:
                finish = torch.zeros_like(start)

            # initial point (0%)
            eval_and_accumulate(start, s, e, step_idx=0)

            start_flat = start.view(B, C, -1)
            finish_flat = finish.view(B, C, -1)

            for i in range(n_steps):
                lo = i * step_unit
                hi = min((i + 1) * step_unit, num_units)
                if lo >= hi:
                    break

                if block_step > 1:
                    # update pixel blocks per sample
                    for b in range(B):
                        sel_flat = block_indices_for(s + b, lo, hi)
                        sel = torch.from_numpy(sel_flat).long()
                        start_flat[b, :, sel] = finish_flat[b, :, sel]
                else:
                    # update top pixels per sample
                    idxs = sort_order[s:e, lo:hi]  # [B, K]
                    for b in range(B):
                        sel = torch.from_numpy(idxs[b]).long()
                        start_flat[b, :, sel] = finish_flat[b, :, sel]

                eval_and_accumulate(start, s, e, step_idx=i + 1)

            del start, finish, start_flat, finish_flat
            torch.cuda.empty_cache()
        
        auc_acc = np.nan_to_num(auc_acc, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(auc_acc, 0.0, 1.0)
    
class InsertionMetric(CausalMetric):
    def __init__(self, model, step=224, klen=11, ksig=5, img_size=224, n_classes=2):
        r"""Create insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        self.klen = klen
        self.ksig = ksig
        super().__init__(
            model, 'ins', step,
            substrate_fn=lambda x: _gaussian_blur_like(x, klen, ksig),
            img_size=img_size, n_classes=n_classes
        )
        
    def __call__(self, img_batch: torch.Tensor, exp_batch: np.ndarray, batch_size: int, **kwargs):
        """Input batch images and explanations, return AUC of insertion metric.

        Args:
            img_batch (tensor.float32): All Input images. [N, C, H, W]
            exp_batch (np.ndarray): All Input explanations. [N, H, W]
            batch_size (int): batch size for evaluation.

        Returns:
            float: average AUC of insertion metric for all images in batch.
        """
        # Evaluate insertion
        '''
        h = insertion.evaluate(torch.from_numpy(images.astype('float32')), exp, 100)
        scores['ins'].append(auc(h.mean(1)))
        '''
        #h = self.evaluate(img_batch, exp_batch, batch_size)
        #return auc(h.mean(1))
        auc = self.evaluate(img_batch, exp_batch, batch_size)
        if isinstance(auc, np.ndarray) and auc.ndim == 1:
            return auc                 # average AUC across samples
        elif isinstance(auc, np.ndarray) and auc.ndim == 2:
            return auc(auc.mean(1))                  # old path: AUC over mean curve
        else:
            raise ValueError(f"Unexpected shape from evaluate: {None if not isinstance(auc, np.ndarray) else auc.shape}")

class DeletionMetric(CausalMetric):
    def __init__(self, model, step=224, img_size=224, n_classes=2):
        r"""Create deletion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super().__init__(model, 'del', step, substrate_fn=torch.zeros_like, img_size=img_size, n_classes=n_classes)
        
    def __call__(self, img_batch: torch.Tensor, exp_batch: np.ndarray, batch_size: int, **kwargs):
        """Input batch images and explanations, return AUC of deletion metric.

        Args:
            img_batch (tensor.float32): All Input images. [N, C, H, W]
            exp_batch (np.ndarray): All Input explanations. [N, H, W]
            batch_size (int): batch size for evaluation.

        Returns:
            float: average AUC of deletion metric for all images in batch.
        """
        # Evaluate deletion
        '''
        h = deletion.evaluate(torch.from_numpy(images.astype('float32')), exp, 100)
        scores['del'].append(auc(h.mean(1)))
        '''
        #h = self.evaluate(img_batch, exp_batch, batch_size)
        #return auc(h.mean(1))
        auc = self.evaluate(img_batch, exp_batch, batch_size)
        if isinstance(auc, np.ndarray) and auc.ndim == 1:
            return auc                 # average AUC across samples
        elif isinstance(auc, np.ndarray) and auc.ndim == 2:
            return auc(auc.mean(1))                  # old path: AUC over mean curve
        else:
            raise ValueError(f"Unexpected shape from evaluate: {None if not isinstance(auc, np.ndarray) else auc.shape}")

class RelevanceMetric():
    
    def __init__(self, pooling_type='l2-norm', output_type='mass', reduce_type='none'):
        r"""Create relevance metric instance.
        
        Args:
            pooling_type (str): Pooling method for aggregating channel-wise relevance.
                Options: 'sum,abs', 'sum,pos', 'max-norm', 'l1-norm', 'l2-norm', 'l2-norm,sq'
            output_type (str): Output type for the relevance metric.
                Options: 'mass', 'rank'
            reduce_type (str): Reduction type for the relevance metric.
                Options: 'mean', 'sum', 'max', 'min', 'median', 'none'
        """
        valid_pooling_types = ['sum,abs', 'sum,pos', 'max-norm', 'l1-norm', 'l2-norm', 'l2-norm,sq']
        assert pooling_type in valid_pooling_types, f"pooling_type must be one of {valid_pooling_types}"
        self.pooling_type = pooling_type
        self.output_type = output_type
        self.reduce_type = reduce_type
        if self.reduce_type == 'mean':
            self.process_func = np.mean
        elif self.reduce_type == 'sum':
            self.process_func = np.sum
        elif self.reduce_type == 'max':
            self.process_func = np.max
        elif self.reduce_type == 'min':
            self.process_func = np.min
        elif self.reduce_type == 'median':
            self.process_func = np.median
        elif self.reduce_type == 'none':
            self.process_func = lambda x: x
        else:
            raise ValueError(f"Unsupported reduce_type: {self.reduce_type}")
        
    def pool_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Pool the relevance along the channel axis, according to the pooling technique specified by pooling_type.
        
        Args:
            heatmap (np.ndarray): Heatmap of shape (C, H, W)
            
        Returns:
            pooled_heatmap (np.ndarray): Pooled heatmap of shape (H, W)
        """
        C, H, W = heatmap.shape

        if self.pooling_type == "sum,abs":
            pooled_heatmap = np.abs(np.sum(heatmap, axis=0))

        elif self.pooling_type == "sum,pos":
            pooled_heatmap = np.sum(heatmap, axis=0)
            pooled_heatmap = np.where(pooled_heatmap > 0.0, pooled_heatmap, 0.0)
        
        elif self.pooling_type == "max-norm":
            pooled_heatmap = np.amax(np.abs(heatmap), axis=0)

        elif self.pooling_type == "l1-norm":
            pooled_heatmap = np.linalg.norm(heatmap, ord=1, axis=0)

        elif self.pooling_type == "l2-norm":
            pooled_heatmap = np.linalg.norm(heatmap, ord=2, axis=0)

        elif self.pooling_type == "l2-norm,sq":
            pooled_heatmap = (np.linalg.norm(heatmap, ord=2, axis=0)) ** 2

        assert pooled_heatmap.shape == (H, W) and np.all(pooled_heatmap >= 0.0)
        return pooled_heatmap
    
    def single_run(self, heatmap: np.ndarray, ground_truth: np.ndarray):
        """
        Evaluate a single image's relevance heatmap against ground truth.
        
        Given an image's relevance heatmap and a corresponding ground truth boolean ndarray, 
        compute two metrics:
         - relevance mass accuracy: ratio of relevance falling into the ground truth area 
           w.r.t. the total amount of relevance
         - relevance rank accuracy: ratio of pixels within the N highest relevant pixels 
           (where N is the size of the ground truth area) that effectively belong to the 
           ground truth area
        
        Args:
            heatmap (np.ndarray): Heatmap of shape (C, H, W), with dtype float
            ground_truth (np.ndarray): Ground truth mask of shape (H, W), with dtype bool
            
        Returns:
            dict: Dictionary with keys ["mass", "rank"] containing:
                - mass (np.float64): Relevance mass accuracy in [0.0, 1.0], higher is better
                - rank (np.float64): Relevance rank accuracy in [0.0, 1.0], higher is better
        """
        #print('Shape:',heatmap.shape, ground_truth.shape)
        C, H, W = heatmap.shape
        assert ground_truth.shape == (H, W), f"Ground truth shape {ground_truth.shape} must match heatmap spatial dims ({H}, {W})"

        # Support multiclass segmentation masks by treating any non-zero label as foreground.
        # (This keeps backward-compat with existing binary masks as well.)
        if ground_truth.dtype != np.bool_:
            ground_truth = (ground_truth > 0) & (ground_truth != 255)

        # Cast heatmap to float64 precision for better accuracy
        heatmap = heatmap.astype(dtype=np.float64)
        
        # Step 1: Pool the relevance across the channel dimension
        pooled_heatmap = self.pool_heatmap(heatmap)
        #print('Pooled shape:',pooled_heatmap.shape)

        # Step 2: Compute the ratio of relevance mass within ground truth w.r.t the total relevance
        relevance_within_ground_truth = np.sum(pooled_heatmap * np.where(ground_truth, 1.0, 0.0).astype(dtype=np.float64))
        relevance_total = np.sum(pooled_heatmap)
        relevance_mass_accuracy = 1.0 * relevance_within_ground_truth / (relevance_total + 1e-9)
        #print('In ground truth:', relevance_within_ground_truth, 'Total:', relevance_total, 'Mass acc:', relevance_mass_accuracy)
        assert (0.0 <= relevance_mass_accuracy) and (relevance_mass_accuracy <= 1.0)

        # Step 3: Order pixels by relevance and count how many of the top-N fall in ground truth
        pixels_sorted_by_relevance = np.argsort(np.ravel(pooled_heatmap))[::-1]
        assert pixels_sorted_by_relevance.shape == (H * W,)
        
        gt_flat = np.ravel(ground_truth)
        assert gt_flat.shape == (H * W,)
        
        N = np.sum(gt_flat)
        if N == 0:
            relevance_rank_accuracy = 0.0
            print("Warning: ground truth mask is empty.")
            print(ground_truth.mean(), ground_truth.min(), ground_truth.max())
            print(ground_truth)
        else:
            topk = pixels_sorted_by_relevance[:N]
            N_gt = int(np.sum(gt_flat[topk]))
            relevance_rank_accuracy = N_gt / N

        N_gt = np.sum(gt_flat[pixels_sorted_by_relevance[:int(N)]])
        relevance_rank_accuracy = np.clip(1.0 * N_gt / (N + 1e-9), 0.0, 1.0) #avoid errors
        #assert (0.0 <= relevance_rank_accuracy) and (relevance_rank_accuracy <= 1.0)
            
        return {"mass": relevance_mass_accuracy, "rank": relevance_rank_accuracy}
    
    def evaluate(self, heatmaps: np.ndarray, ground_truths: np.ndarray):
        """
        Evaluate a batch of heatmaps against ground truths.
        
        Args:
            heatmaps (np.ndarray): Batch of heatmaps of shape (N, C, H, W)
            ground_truths (np.ndarray): Batch of ground truth masks of shape (N, H, W)
            
        Returns:
            dict: Dictionary with keys ["mass", "rank"] containing arrays of scores for each image:
                - mass (np.ndarray): Array of relevance mass accuracies of shape (N,)
                - rank (np.ndarray): Array of relevance rank accuracies of shape (N,)
        """
        n_samples = heatmaps.shape[0]
        assert ground_truths.shape[0] == n_samples, "Number of heatmaps and ground truths must match"
        
        mass_scores = np.zeros(n_samples)
        rank_scores = np.zeros(n_samples)
        
        for i in tqdm(range(n_samples), desc='Evaluating relevance',disable=not sys.stdout.isatty()):
            result = self.single_run(heatmaps[i], ground_truths[i])
            mass_scores[i] = result['mass']
            rank_scores[i] = result['rank']
        
        return {"mass": mass_scores, "rank": rank_scores}
    
    def __call__(self,images: torch.Tensor, exp_batch: np.ndarray, gt_mask: np.ndarray, **kwargs):
        """
        Evaluate heatmaps against ground truths and return average scores.
        
        Args:
            images (torch.Tensor): Batch of images of shape (N, C, H, W), not used in this function
            exp_batch (np.ndarray): Batch of heatmaps of shape (N, H, W) or single heatmap of shape (H, W)
            gt_mask (np.ndarray): Batch of ground truth masks of shape (N, H, W) or single mask of shape (H, W)
            **kwargs: Additional keyword arguments (not used in this function)
            
        Returns:
            float or dict: Average relevance mass accuracy or dictionary with keys ["mass", "rank"] containing average scores
        """
        # Handle single image case (H, W) -> (1, H, W)
        if exp_batch.ndim == 2:
            exp_batch = exp_batch[np.newaxis, :, :]  # Add channel dimension
            result = self.single_run(exp_batch, gt_mask)
            if self.output_type == 'mass':
                return result["mass"]
            elif self.output_type == 'rank':
                return result["rank"]
            else:
                return result
        
        # Handle batch case (N, H, W) -> (N, 1, H, W)
        exp_batch = exp_batch[:, np.newaxis, :, :]  # Add channel dimension
        
        # Handle batch case
        results = self.evaluate(exp_batch, gt_mask)
        if self.output_type == 'mass':
            return self.process_func(results["mass"])
        elif self.output_type == 'rank':
            return self.process_func(results["rank"])
        else:
            return {"mass": self.process_func(results["mass"]), "rank": self.process_func(results["rank"])}

# -----------------------------------------------------------------------------
# Layer-importance distribution metrics (entropy / gini / dispersion / top-3 ratio)
# -----------------------------------------------------------------------------

class LayerImportanceDistributionMetric:
    """
    Computes a scalar metric on the distribution of per-layer importances.

    Per-layer importance is defined as the summed (non-negative) saliency mass inside each label region
    of the segmentation mask.
    """

    def __init__(
        self,
        *,
        ignore_background: bool = True,
        output_type: str = "entropy",
        pooling_type: str = "sum,abs",
    ):
        """
        Args:
            ignore_background: if True, excludes label 0 from layer set.
            output_type: one of {"entropy","gini","dispersion","top3_ratio"}.
            pooling_type: how to pool multi-channel heatmaps (if provided as CxHxW). Reuses RelevanceMetric.
        """
        self.ignore_background = ignore_background
        self.output_type = output_type
        self._pooler = RelevanceMetric(pooling_type=pooling_type, output_type="mass")
        valid = {"entropy", "gini", "dispersion", "top3_ratio"}
        if self.output_type not in valid:
            raise ValueError(f"output_type must be one of {sorted(valid)}, got {self.output_type}")

    def _metric_from_scores(self, scores: np.ndarray) -> float:
        if self.output_type == "entropy":
            return shannon_entropy(scores)
        if self.output_type == "gini":
            return gini_coefficient(scores)
        if self.output_type == "dispersion":
            return dispersion_cv(scores)
        if self.output_type == "top3_ratio":
            return topk_ratio(scores, k=3)
        raise RuntimeError("unreachable")

    def single_run(self, heatmap: np.ndarray, seg_mask: np.ndarray) -> float:
        """
        heatmap: (H,W) or (C,H,W); seg_mask: (H,W) labels.
        """
        heatmap = np.asarray(heatmap)
        seg_mask = np.asarray(seg_mask)

        if heatmap.ndim == 3:
            heatmap_2d = self._pooler.pool_heatmap(heatmap.astype(np.float64))
        elif heatmap.ndim == 2:
            heatmap_2d = heatmap.astype(np.float64)
        else:
            raise ValueError(f"Unsupported heatmap ndim={heatmap.ndim}, expected 2 or 3.")

        scores = _layer_importances_from_mask_and_heatmap(
            seg_mask,
            heatmap_2d,
            ignore_background=self.ignore_background,
        )
        return float(self._metric_from_scores(scores))

    def __call__(self, images: torch.Tensor, exp_batch: np.ndarray, gt_mask: np.ndarray, **kwargs):
        """
        Args:
            images: unused (kept for evaluator compatibility)
            exp_batch: (B,H,W) or (H,W) or (B,C,H,W) or (C,H,W)
            gt_mask: (B,H,W) or (H,W) segmentation labels
        Returns:
            Per-sample array of shape (B,) for batch inputs, else a scalar float for single input.
        """
        exp = np.asarray(exp_batch)
        mask = np.asarray(gt_mask)

        # Disambiguate single vs batch using gt_mask ndim (in this repo, gt_mask is provided alongside exp_batch).
        if mask.ndim == 2:
            # Single sample: exp can be (H,W) or (C,H,W)
            if exp.ndim not in (2, 3):
                raise ValueError(f"Unsupported exp_batch ndim={exp.ndim} for single sample.")
            return self.single_run(exp, mask)

        if mask.ndim == 3:
            # Batch: exp can be (B,H,W) or (B,C,H,W)
            if exp.ndim not in (3, 4):
                raise ValueError(f"Unsupported exp_batch ndim={exp.ndim} for batch.")
            if exp.shape[0] != mask.shape[0]:
                raise ValueError(
                    f"Batch mismatch: exp_batch has {exp.shape[0]} samples, gt_mask has {mask.shape[0]} samples."
                )
            out = np.zeros((exp.shape[0],), dtype=np.float64)
            for i in range(exp.shape[0]):
                out[i] = self.single_run(exp[i], mask[i])
            return out

        raise ValueError(f"Unsupported gt_mask ndim={mask.ndim}, expected 2 (single) or 3 (batch).")

# Legacy functions for backward compatibility
def pool_heatmap(heatmap: np.ndarray, pooling_type: str) -> np.ndarray:
    """
    [DEPRECATED] Use RelevanceMetric class instead.
    
    Pool the relevance along the channel axis, according to the pooling technique specified by pooling_type.
    """
    metric = RelevanceMetric(pooling_type=pooling_type)
    return metric.pool_heatmap(heatmap)

def evaluate_single(heatmap: np.ndarray, ground_truth: np.ndarray, pooling_type: str):
    """
    [DEPRECATED] Use RelevanceMetric class instead.
    
    Given an image's relevance heatmap and a corresponding ground truth boolean ndarray of the same vertical and horizontal dimensions, return both:
     - the ratio of relevance falling into the ground truth area w.r.t. the total amount of relevance ("relevance mass accuracy" metric)
     - the ratio of pixels within the N highest relevant pixels (where N is the size of the ground truth area) that effectively belong to the ground truth area
       ("relevance rank accuracy" metric)
    Both ratios are calculated after having pooled the relevance across the channel axis, according to the pooling technique defined by the pooling_type argument.
    Args:
    - heatmap (np.ndarray):         of shape (C, H, W), with dtype float 
    - ground_truth (np.ndarray):    of shape (H, W), with dtype bool
    - pooling_type (str):           specifies how to pool the relevance across the channels, i.e. defines a mapping function f: R^C -> R^+
                                    that maps a real-valued vector of dimension C to a positive number (see details of each pooling_type in the function pool_heatmap)
    Returns:
    A dict wich keys=["mass", "rank"] and resp. values:
    - relevance_mass_accuracy (np.float64):     relevance mass accuracy, float in the range [0.0, 1.0], the higher the better.
    - relevance_rank_accuracy (np.float64):     relevance rank accuracy, float in the range [0.0, 1.0], the higher the better.
    """
    metric = RelevanceMetric(pooling_type=pooling_type)
    return metric.single_run(heatmap, ground_truth)
