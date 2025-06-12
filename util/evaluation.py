import torch
from torch import nn
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
        top, c = torch.max(pred, 1)
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
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = pred[0, c]
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

    def evaluate(self, img_batch, exp_batch, batch_size):
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
            preds = self.model(img_batch[i*batch_size:(i+1)*batch_size].cuda()).cpu()
            predictions[i*batch_size:(i+1)*batch_size] = preds
        img_batch = img_batch.cpu().float()
        top = np.argmax(predictions, -1)
        n_steps = (self.img_size*self.img_size + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, self.img_size*self.img_size), axis=1), axis=-1)
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

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(start[j*batch_size:(j+1)*batch_size].cuda())
                preds = preds.cpu().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
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
        
    def __call__(self, img_batch, exp_batch, batch_size):
        """Input batch images and explanations, return AUC of insertion metric.

        Args:
            img_batch (tensor.float32): All Input images. [N, C, H, W]
            exp_batch (_type_): All Input explanations. [N, H, W]
            batch_size (_type_): batch size for evaluation.

        Returns:
            _type_: average AUC of insertion metric for all images in batch.
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
        
    def __call__(self, img_batch, exp_batch, batch_size):
        """Input batch images and explanations, return AUC of deletion metric.

        Args:
            img_batch (tensor.float32): All Input images. [N, C, H, W]
            exp_batch (_type_): All Input explanations. [N, H, W]
            batch_size (_type_): batch size for evaluation.

        Returns:
            _type_: average AUC of deletion metric for all images in batch.
        """
        # Evaluate deletion
        '''
        h = deletion.evaluate(torch.from_numpy(images.astype('float32')), exp, 100)
        scores['del'].append(auc(h.mean(1)))
        '''
        h = self.evaluate(img_batch, exp_batch, batch_size)
        return auc(h.mean(1))
        