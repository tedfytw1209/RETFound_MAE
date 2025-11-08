import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets

from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

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

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

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

    def evaluate(self, img_batch: torch.Tensor, exp_batch: np.ndarray, batch_size: int):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        
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
                else:
                    preds = output
                probs = torch.softmax(preds, dim=1)
                probs = probs.detach().cpu().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = probs
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, self.img_size*self.img_size)[r, :, coords] = finish.cpu().numpy().reshape(n_samples, 3, self.img_size*self.img_size)[r, :, coords]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores
    
class InsertionMetric(CausalMetric):
    def __init__(self, model, step=224, klen=11, ksig=5, img_size=224, n_classes=2):
        r"""Create insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        kern = gkern(klen, ksig)
        # Function that blurs input image
        blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
        #insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
        super().__init__(model, 'ins', step, blur, img_size=img_size, n_classes=n_classes)
        
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
        h = self.evaluate(img_batch, exp_batch, batch_size)
        return auc(h.mean(1))
    
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
        h = self.evaluate(img_batch, exp_batch, batch_size)
        return auc(h.mean(1))

class RelevanceMetric():
    
    def __init__(self, pooling_type='l2-norm', output_type='mass'):
        r"""Create relevance metric instance.
        
        Args:
            pooling_type (str): Pooling method for aggregating channel-wise relevance.
                Options: 'sum,abs', 'sum,pos', 'max-norm', 'l1-norm', 'l2-norm', 'l2-norm,sq'
            output_type (str): Output type for the relevance metric.
                Options: 'mass', 'rank'
        """
        valid_pooling_types = ['sum,abs', 'sum,pos', 'max-norm', 'l1-norm', 'l2-norm', 'l2-norm,sq']
        assert pooling_type in valid_pooling_types, f"pooling_type must be one of {valid_pooling_types}"
        self.pooling_type = pooling_type
        self.output_type = output_type
        
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
        C, H, W = heatmap.shape
        assert ground_truth.shape == (H, W), f"Ground truth shape {ground_truth.shape} must match heatmap spatial dims ({H}, {W})"

        # Cast heatmap to float64 precision for better accuracy
        heatmap = heatmap.astype(dtype=np.float64)
        
        # Step 1: Pool the relevance across the channel dimension
        pooled_heatmap = self.pool_heatmap(heatmap)

        # Step 2: Compute the ratio of relevance mass within ground truth w.r.t the total relevance
        relevance_within_ground_truth = np.sum(pooled_heatmap * np.where(ground_truth, 1.0, 0.0).astype(dtype=np.float64))
        relevance_total = np.sum(pooled_heatmap)
        relevance_mass_accuracy = 1.0 * relevance_within_ground_truth / relevance_total
        assert (0.0 <= relevance_mass_accuracy) and (relevance_mass_accuracy <= 1.0)

        # Step 3: Order pixels by relevance and count how many of the top-N fall in ground truth
        pixels_sorted_by_relevance = np.argsort(np.ravel(pooled_heatmap))[::-1]
        assert pixels_sorted_by_relevance.shape == (H * W,)
        
        gt_flat = np.ravel(ground_truth)
        assert gt_flat.shape == (H * W,)
        
        N = np.sum(gt_flat)
        N_gt = np.sum(gt_flat[pixels_sorted_by_relevance[:int(N)]])
        relevance_rank_accuracy = 1.0 * N_gt / N
        assert (0.0 <= relevance_rank_accuracy) and (relevance_rank_accuracy <= 1.0)
            
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
        
        for i in tqdm(range(n_samples), desc='Evaluating relevance'):
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
            return np.mean(results["mass"])
        elif self.output_type == 'rank':
            return np.mean(results["rank"])
        else:
            return {"mass": np.mean(results["mass"]), "rank": np.mean(results["rank"])}

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
