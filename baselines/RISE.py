import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
import os
from contextlib import nullcontext

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100, maskspath='masks.npy', N=1000):
        super(RISE, self).__init__()
        self.model = model.eval()
        self.input_size = input_size # (H, W)
        self.gpu_batch = gpu_batch
        if not os.path.isfile(maskspath):
            self.generate_masks(N=N, s=10, p1=0.1, savepath=maskspath)
        else:
            self.load_masks(maskspath)
            print('Masks are loaded in CPU')

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        H, W = self.input_size
        cell_size = np.ceil(np.array(self.input_size) / s).astype(int)
        up_size = ((s + 1) * cell_size[0], (s + 1) * cell_size[1])

        grid = (np.random.rand(N, s, s) < p1).astype('float32')
        masks = np.empty((N, *self.input_size), dtype='float32')

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + H, y:y + W]
            
        masks = masks.reshape(N, 1, H, W)
        np.save(savepath, masks)
        self.masks = torch.from_numpy(masks).float()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float()
        self.N = self.masks.shape[0]
        self.p1 = float(self.masks.mean().item())

    def forward(self, x):
        """
        x: (1, C, H, W) tensor on GPU
        Returns: saliency (CL, H, W) on CPU
        """
        assert x.dim() == 4 and x.size(0) == 1
        device = x.device
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        #stack = torch.mul(self.masks, x.data)
        with torch.no_grad():
            dummy = self.model(x)
            CL = dummy.size(1)
        sal_acc = torch.zeros(CL, H * W, device=device, dtype=x.dtype)
        
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (device.type == 'cuda' and self.use_amp) else nullcontext()

        with torch.no_grad(), amp_ctx:
            for i in range(0, N, self.gpu_batch):
                j = min(i + self.gpu_batch, N)
                m = self.masks[i:j].to(device, non_blocking=True)      # (b,1,H,W)
                x_rep = x.expand(m.size(0), -1, -1, -1)                # (b,C,H,W)
                masked = x_rep * m                                      # (b,C,H,W)
                p = self.model(masked)                                  # (b, CL)
                m_flat = m.view(m.size(0), -1)                          # (b, H*W)
                sal_acc += p.transpose(0, 1) @ m_flat                   # (CL, H*W)
                del m, x_rep, masked, p, m_flat
                torch.cuda.empty_cache()

        sal = sal_acc.view(CL, H, W) / N / self.p1                     # 規範化
        return sal.detach().cpu().numpy()  # shape: (CL, H, W)
    
    
class RISEBatch(RISE):
    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, C, H, W) on GPU
        return: (B, CL, H, W) -> numpy
        """
        assert x.is_cuda
        device = x.device
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        
        dummy_out = self.model(x[:1])     # (1, CL)
        CL = dummy_out.size(1)
        
        #AMP
        use_amp = bool(getattr(self, "use_amp", False))
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (device.type == "cuda" and use_amp)
            else nullcontext()
        )
        
        sal_all = torch.zeros(B, CL, H * W, device=device, dtype=x.dtype)
        
        with amp_ctx:
            for b in range(B):
                xb = x[b:b+1]                             # (1, C, H, W)
                sal_acc = torch.zeros(CL, H * W, device=device, dtype=x.dtype)
                for i in range(0, N, self.gpu_batch):
                    j = min(i + self.gpu_batch, N)
                    m = self.masks[i:j].to(device, non_blocking=True)   # (m, 1, H, W)
                    xb_rep = xb.expand(m.size(0), -1, -1, -1)           # (m, C, H, W)
                    masked = xb_rep * m                                 # (m, C, H, W)
                    p = self.model(masked)                              # (m, CL)
                    # p = p.softmax(dim=1)
                    m_flat = m.view(m.size(0), -1)                      # (m, H*W)
                    sal_acc += p.transpose(0, 1) @ m_flat               # (CL, H*W)
                    del m, xb_rep, masked, p, m_flat
                    torch.cuda.empty_cache()

                sal_all[b] = sal_acc
        
        sal_all = sal_all.view(B, CL, H, W) / N / self.p1
        return sal_all.detach().cpu().numpy()

# To process in batches
# def explain_all_batch(data_loader, explainer):
#     n_batch = len(data_loader)
#     b_size = data_loader.batch_size
#     total = n_batch * b_size
#     # Get all predicted labels first
#     target = np.empty(total, 'int64')
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
#         p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
#         target[i * b_size:(i + 1) * b_size] = c
#     image_size = imgs.shape[-2:]
#
#     # Get saliency maps for all images in val loader
#     explanations = np.empty((total, *image_size))
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
#         saliency_maps = explainer(imgs.cuda())
#         explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
#             range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
#     return explanations
