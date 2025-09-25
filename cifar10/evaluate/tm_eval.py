import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import time
import pandas as pd
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from common.utils import AverageMeter, accuracy
from common.model import WideResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

exp_curve = pd.read_csv('/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/expert/expert_acc_curve_new_new.csv')
p_curve = exp_curve['p']

# exp_low_curve = pd.read_csv('/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/expert/expert_acc_curve_low_new.csv')
# p_low_curve = exp_low_curve['p']

from torch.utils.data import Dataset
from typing import Tuple, Any


import torch.nn as nn
import torch.nn.functional as F
from common.model import WideResNetRevised


import math, torch, numpy as np
from torch.utils.data import Subset, DataLoader

def build_sequence_windows(num_samples, T, stride):
    """
    Pure sliding window: returns T consecutive sample indices for each sequence (list of lists).
    Ensures no out-of-bounds (tail with insufficient T will be discarded).
    """
    starts = list(range(0, num_samples - T + 1, stride))
    seqs = [list(range(s, s + T)) for s in starts]
    return seqs

def split_train_test_ranges(num_samples, train_ratio=0.9):
    """
    Safe split: divide original indices into two non-overlapping intervals to avoid sample leakage.
    Returns (train_range, test_range), both as (start, end) half-open intervals.
    """
    split = int(num_samples * train_ratio)
    return (0, split), (split, num_samples)

def windows_in_range(start_end, T, stride):
    """
    Sliding window only within a certain interval, ensuring each window falls completely within that interval.
    Returns a list of window start points 'relative to the interval start'.
    """
    start, end = start_end
    length = end - start
    starts = list(range(0, length - T + 1, stride))
    # Convert to "global" start points
    return [s + start for s in starts]

def make_augmented_sequence_datasets(
    base_dataset,               # e.g., CIFAR-10 train split with transforms
    T,                          # sequence length
    amp=1,                      # amplification factor (larger => more sequences)
    stride=None,                # if given, ignore amp and use this sliding step size directly
    train_ratio=0.9,            # data split ratio by sample index
    safe_split=True,            # True: split first then sliding window to avoid leakage; False: full sliding window then random split (may leak)
    seed=0
):
    rng = np.random.default_rng(seed)
    N = len(base_dataset)
    assert T > 0 and T <= N, "Invalid T"

    # Derive stride from amp (empirical approach: smaller stride means larger amplification)
    if stride is None:
        # Make sliding window coverage roughly ~ amp: simple approximation stride ≈ max(1, T // amp)
        stride = max(1, T // max(int(amp), 1))

    if safe_split:
        # Safe: first split samples into two segments, then sliding window within each
        train_range, test_range = split_train_test_ranges(N, train_ratio)

        train_starts = windows_in_range(train_range, T, stride)
        test_starts  = windows_in_range(test_range,  T, stride)

        train_indices = []
        for s in train_starts:
            train_indices.extend(range(s, s + T))

        test_indices = []
        for s in test_starts:
            test_indices.extend(range(s, s + T))

        # Use Subset then pass to your IndexedCIFAR10 (it needs "flat index list")
        train_subset = Subset(base_dataset, train_indices)
        test_subset  = Subset(base_dataset, test_indices)
    else:
        # Unsafe: full sliding window then random split by "sequence" units (sequences from different splits may share samples)
        starts = list(range(0, N - T + 1, stride))
        rng.shuffle(starts)
        num_train = int(len(starts) * train_ratio)
        train_starts = starts[:num_train]
        test_starts  = starts[num_train:]

        train_indices = []
        for s in train_starts:
            train_indices.extend(range(s, s + T))
        test_indices = []
        for s in test_starts:
            test_indices.extend(range(s, s + T))

        train_subset = Subset(base_dataset, train_indices)
        test_subset  = Subset(base_dataset, test_indices)

    # Your IndexedCIFAR10 only needs "one dataset" and "T", it will calculate seq_id and t in __getitem__
    train_indexed = IndexedCIFAR10(train_subset, T)
    test_indexed  = IndexedCIFAR10(test_subset,  T)
    return train_indexed, test_indexed

def make_augmented_sequence_loaders(
    base_dataset,
    T,
    amp=1,
    stride=None,
    train_ratio=0.9,
    safe_split=True,
    seed=0,
    batch_seqs=16,         # how many sequences per batch
    num_workers=0,
    pin_memory=True,
    device="cuda"
):
    """
    Convenient to get DataLoader directly.
    Ensures each batch size = batch_seqs * T, convenient for batch training by sequence.
    """
    train_ds, test_ds = make_augmented_sequence_datasets(
        base_dataset, T, amp, stride, train_ratio, safe_split, seed
    )

    # Set batch size to "number of sequences * T"
    train_loader = DataLoader(
        train_ds, batch_size=batch_seqs * T, shuffle=False, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_seqs * T, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"[aug] train sequences: {len(train_ds)//T}, test sequences: {len(test_ds)//T}, T={T}, "
          f"amp={amp}, stride={stride}, safe_split={safe_split}")
    return train_loader, test_loader

# class L2DGRU(nn.Module):
#     def __init__(self, backbone: WideResNetRevised, hidden_dim: int, num_layers: int, n_classes: int, dropout: float = 0.0):
#         super(L2DGRU, self).__init__()
#         self.backbone = backbone
#         self.n_classes = n_classes
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         # We'll dynamically determine the backbone's feature dimension.
#         gru_input_dim = self.backbone.feat_dim + 2 # +1 for h_t-1, +1 for t
        
#         # Define the GRU network
#         self.gru = nn.GRU(gru_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
#         # The final layer combines classifier and deferral outputs
#         self.fc = nn.Linear(hidden_dim, n_classes + 1)

#     def forward(self, x: torch.Tensor, h_prev: torch.Tensor, t: torch.Tensor, hidden_state=None) -> torch.Tensor:
#         # x: (B, C, H, W)
#         batch_size = x.size(0)
        
#         # Get features from the trainable backbone
#         features = self.backbone.forward_features(x)
        
#         # Ensure all inputs for GRU have 3 dimensions: [B, T, D] where T=1
#         # features: [B, feat_dim] -> [B, 1, feat_dim]
#         # h_prev:   [B, 1]       -> [B, 1, 1]
#         # t:        [B, 1]       -> [B, 1, 1]
        
#         # The corrected shapes:
#         t_input = t.unsqueeze(1)

#         gru_input = torch.cat([features, h_prev, t_input], dim=1)
        
#         # GRU forward pass
#         gru_output, hidden = self.gru(gru_input, hidden_state)
#         # print(gru_output.shape)
        
#         # Squeeze the sequence length dimension (which is 1)
#         output = self.fc(gru_output.squeeze(1))
#         # print(output.shape)
        
#         return F.log_softmax(output, dim=1)

INV_LN2 = 1.4426950408889634  # 1/ln(2), maintain consistency with "base-2" measure

def reject_ce_from_logprobs(logp: torch.Tensor,
                            m: torch.Tensor,      # [B] in {0,1}
                            labels: torch.Tensor, # [B] long
                            m2: torch.Tensor,     # [B], alpha or 1
                            n_classes: int) -> torch.Tensor:
    idx = torch.arange(logp.size(0), device=logp.device)
    # -[ m * log p_defer + m2 * log p_label ]
    loss_nat = -(m * logp[idx, n_classes] + m2 * logp[idx, labels])  # base e
    # If you want to keep "log2 scale", multiply by 1/ln(2)
    return (loss_nat * INV_LN2).mean()

class L2DGRU(nn.Module):
    def __init__(self, backbone, hidden_dim, num_layers, n_classes, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Assume backbone.forward_features(x) -> [B, F]
        self.gru_input_dim = self.backbone.feat_dim + 2  # + h_prev + t
        self.gru = nn.GRU(self.gru_input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, n_classes + 1)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, t: torch.Tensor, hidden_state=None):
        # x: [B, C, H, W], h_prev: [B, 1], t: [B, 1]
        feats = self.backbone.forward_features(x)           # [B, F]
        assert feats.dim() == 2, f"features dim must be 2, got {feats.shape}"

        feats   = feats.unsqueeze(1)                        # [B, 1, F]
        h_prev  = h_prev.to(feats.dtype).unsqueeze(1)      # [B, 1, 1]
        t_input = t.to(feats.dtype).unsqueeze(1)            # [B, 1, 1]

        # print(h_prev.shape, t_input.shape, feats.shape)

        gru_input = torch.cat([feats, h_prev, t_input], dim=2)

        # Safety check (can keep for a few training rounds)
        if not torch.isfinite(gru_input).all():
            raise RuntimeError("Non-finite values in GRU input")

        out, _ = self.gru(gru_input, hidden_state)         # [B, 1, H]
        logits = self.fc(out.squeeze(1))                   # [B, K+1]

        log_probs = F.log_softmax(logits, dim=1)
        if not torch.isfinite(log_probs).all():
            print("WARNING: non-finite log_probs; clamping")
            log_probs = torch.nan_to_num(log_probs, neginf=-30.0, posinf=0.0)

        return log_probs

class IndexedCIFAR10(Dataset):
    def __init__(self, original_dataset: Dataset, seq_len: int):
        self.original_dataset = original_dataset
        self.seq_len = seq_len
        self.num_samples = len(original_dataset)
        
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any, int, int]:
        img, label = self.original_dataset[idx]
        seq_id = idx // self.seq_len
        t = idx % self.seq_len
        return img, label, seq_id, t

def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    '''
    The L_{CE} loss implementation for CIFAR
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]  # batch_size
    rc = [n_classes] * batch_size
    outputs = -m * torch.log2(outputs[range(batch_size), rc]) - m2 * torch.log2(
        outputs[range(batch_size), labels])  
    return torch.sum(outputs) / batch_size

def my_CrossEntropyLoss(outputs, labels):
    # Regular Cross entropy loss
    batch_size = outputs.size()[0]  # batch_size
    outputs = - torch.log2(outputs[range(batch_size), labels])  # regular CE
    return torch.sum(outputs) / batch_size


@torch.no_grad()
def eval_seq_loss(val_loader, model, expert_fn, n_classes, T, alpha, prior=0.0):
    """
    Evaluate sequential deferral average loss (consistent with train_reject calculation, just no backprop).
    Returns: scalar float (smaller is better)
    """
    model.eval()
    dev = next(model.parameters()).device
    INV_LN2 = 1.4426950408889634
    total_loss = 0.0
    total_count = 0

    for images, labels, seq_ids, ts in val_loader:
        images = images.to(dev, non_blocking=True).float()
        labels = labels.to(dev, non_blocking=True).long()
        ts     = ts.to(dev, non_blocking=True).long()

        B = images.size(0)
        assert B % T == 0, "val batch_size must be divisible by T (set drop_last=True or adjust batch)"
        S = B // T

        # [S,T,...]
        images_seq = images.view(S, T, *images.shape[1:])
        labels_seq = labels.view(S, T)
        ts_seq     = ts.view(S, T)

        # CNN features
        imgs_flat  = images_seq.reshape(S*T, *images.shape[1:])
        feats_flat = model.backbone.forward_features(imgs_flat)  # [S*T,F]
        Fdim       = feats_flat.size(1)
        feats      = feats_flat.view(S, T, Fdim)                 # [S,T,F]

        # Expert predictions and correctness
        exp_preds_flat = expert_fn(imgs_flat, labels.view(-1), ts.view(-1))  # [S*T]
        exp_preds      = exp_preds_flat.view(S, T)
        exp_correct    = (exp_preds == labels_seq).float()                   # [S,T]

        # h_prev and t
        h_prev = torch.zeros(S, T, 1, device=dev, dtype=feats.dtype)
        h_prev[:, 1:, 0] = exp_correct[:, :-1]
        h_prev[:, 0, 0]  = prior
        t_norm = (ts_seq.float() / (T - 1)).unsqueeze(-1)

        # GRU forward
        gru_in = torch.cat([feats, h_prev, t_norm], dim=2)      # [S,T,F+2]
        out, _ = model.gru(gru_in)                              # [S,T,H]
        logits = model.fc(out)                                  # [S,T,n_classes+1]
        logp   = F.log_softmax(logits, dim=-1)

        # Loss consistent with training
        logp_defer = logp[..., n_classes]                       # [S,T]
        logp_label = logp.gather(dim=2, index=labels_seq.unsqueeze(-1)).squeeze(-1)  # [S,T]
        m  = exp_correct
        m2 = torch.where(m > 0.5, torch.full_like(m, float(alpha)), torch.ones_like(m))

        loss_nat = -(m * logp_defer + m2 * logp_label)          # base e
        loss     = (loss_nat * INV_LN2).mean()                  # average over all S*T samples

        total_loss  += float(loss.detach().cpu()) * (S*T)
        total_count += (S*T)

    return total_loss / max(total_count, 1)

def train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, alpha, T):
    model.train()
    batch_time = AverageMeter(); losses = AverageMeter(); top1 = AverageMeter()
    end = time.time()

    for i, (images, labels, seq_ids, ts) in enumerate(train_loader):
        # Device and dtype
        dev = next(model.parameters()).device
        images = images.to(dev, non_blocking=True).float()
        labels = labels.to(dev, non_blocking=True).long()
        ts     = ts.to(dev, non_blocking=True).long()

        B = images.size(0)
        assert B % T == 0, "batch_size should be divisible by T (DataLoader set drop_last=True & reasonable batch_size)"
        S = B // T  # number of sequences

        # 1) Pre-reshape batch to [S, T, ...]
        images_seq = images.view(S, T, *images.shape[1:])        # [S,T,C,H,W]
        labels_seq = labels.view(S, T)                            # [S,T]
        ts_seq     = ts.view(S, T)                                # [S,T]

        # 2) Extract CNN features for all frames at once, then reshape back to [S,T,F]
        #    Note: backbone.forward_features accepts [B,C,H,W], so flatten first
        imgs_flat = images_seq.reshape(S*T, *images.shape[1:])    # [S*T,C,H,W]
        feats_flat = model.backbone.forward_features(imgs_flat)   # [S*T,F]
        Fdim = feats_flat.size(1)
        feats = feats_flat.view(S, T, Fdim)                       # [S,T,F]

        # 3) Get expert predictions at once (vectorized) and calculate correctness for each step
        #    expert_fn(images_t, labels_t, ts_t) already supports batch; feed it all at once
        exp_preds_flat = expert_fn(imgs_flat, labels.view(-1), ts.view(-1))  # [S*T]
        exp_preds = exp_preds_flat.view(S, T)                                  # [S,T]
        exp_correct = (exp_preds == labels_seq).float()                        # [S,T]

        # 4) Construct h_prev: shift exp_correct "right by one" along time dimension, use prior for t=0 (e.g. 0 or 0.5)
        prior = 0.0  # or 0.5, depending on your setting
        h_prev = torch.zeros(S, T, 1, device=dev, dtype=feats.dtype)
        h_prev[:, 1:, 0] = exp_correct[:, :-1]  # h_prev for t>0 = whether expert was correct in previous step
        h_prev[:, 0, 0]  = prior

        # 5) Construct t_input: normalized time steps, shape [S,T,1]
        t_norm = (ts_seq.float() / (T - 1)).unsqueeze(-1)  # [S,T,1]

        # 6) Concatenate GRU input and forward at once
        gru_in = torch.cat([feats, h_prev, t_norm], dim=2)  # [S,T,F+2]
        # GRU expects [batch, seq, dim], which matches perfectly
        out, _ = model.gru(gru_in)                          # [S,T,H]
        logits = model.fc(out)                               # [S,T,n_classes+1]
        logp = F.log_softmax(logits, dim=-1)                 # [S,T,K+1]

        # 7) Calculate loss (average over T)
        idx_s = torch.arange(S, device=dev).unsqueeze(1).expand(S,T)  # [S,T]
        idx_t = torch.arange(T, device=dev).unsqueeze(0).expand(S,T)  # [S,T]
        # defer class index = n_classes
        logp_defer = logp[idx_s, idx_t, n_classes]                   # [S,T]
        logp_label = logp.gather(dim=2, index=labels_seq.unsqueeze(-1)).squeeze(-1)  # [S,T]

        # m: whether expert is correct in this step
        m = exp_correct                                             # [S,T]
        # m2: use alpha for hits, otherwise 1
        m2 = torch.where(m > 0.5, torch.full_like(m, float(alpha)), torch.ones_like(m))

        loss_nat = -(m * logp_defer + m2 * logp_label)              # base e
        INV_LN2 = 1.4426950408889634
        loss = (loss_nat * INV_LN2).mean()                           # mean over S*T

        # 8) Backprop & update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Optional: gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        # 9) Statistics accuracy (argmax of non-defer classes as alone; full system argmax)
        pred_all  = logp.argmax(dim=-1)                              # [S,T]
        pred_alone = logp[..., :n_classes].argmax(dim=-1)            # [S,T]
        top1_batch = (pred_all == labels_seq).float().mean() * 100.0

        losses.update(float(loss.detach().cpu()), S*T)
        top1.update(float(top1_batch.detach().cpu()), S*T)

        batch_time.update(time.time() - end); end = time.time()
        if i % 10 == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})")

# def train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, alpha, T):
#     """
#     Train for one epoch on the training set with L2D-GRU deferral.
    
#     Args:
#         train_loader: DataLoader providing (images, labels, seq_ids, ts).
#         model: The L2DGRU model.
#         optimizer: The optimizer for the model.
#         scheduler: The learning rate scheduler.
#         epoch: The current epoch number.
#         expert_fn: The function to get expert predictions.
#         n_classes: The number of classification classes.
#         alpha: The alpha parameter for the loss function.
#         T: The sequence length.
#     """
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()

#     # switch to train mode
#     model.train()

#     end = time.time()
#     for i, (images, labels, seq_ids, ts) in enumerate(train_loader):
#         images, labels, ts = images.to(device), labels.to(device), ts.to(device)
#         batch_size = images.size(0)
        
#         # Since we use drop_last=True in the DataLoader, we can assume a clean batch.
#         num_sequences = batch_size // T
        
#         # Initialize h_prev (previous expert correctness)
#         h_prev = torch.zeros(num_sequences, 1).to(device)

#         total_loss = 0
        
#         # Process the sequence step-by-step
#         for t_step in range(T):
#             # Get the input for the current time step from the batch
#             start_idx = t_step * num_sequences
#             end_idx = start_idx + num_sequences
            
#             images_t = images[start_idx:end_idx]
#             labels_t = labels[start_idx:end_idx]
#             ts_t = ts[start_idx:end_idx]
            
#             # Normalize t_step to [0, 1] for the GRU input
#             t_input = ts_t.float() / (T - 1)

#             t_input = t_input.unsqueeze(1)
            
#             # Get expert predictions for this time step
#             expert_preds_t = expert_fn(images_t, labels_t, ts_t)
            
#             # Get model outputs (log-probabilities)
#             log_outputs = model(images_t, h_prev, t_input)


#             m = (expert_preds_t == labels_t).float()
#             m2 = torch.where(m > 0.5, torch.full_like(m, float(alpha)),        # [B]
#                  torch.ones_like(m))
#             loss = reject_ce_from_logprobs(log_outputs, m, labels_t, m2, n_classes)
#             total_loss += loss

#             # Update h_prev based on expert correctness
#             is_expert_correct = (expert_preds_t == labels_t).float().view(-1, 1).to(device)
#             h_prev = is_expert_correct
        
#         # Backpropagate the total loss for the whole sequence
#         mean_loss = total_loss / T
#         optimizer.zero_grad()
#         mean_loss.backward()
#         optimizer.step()
#         scheduler.step()

#         # Update and print metrics
#         prec1 = accuracy(torch.exp(log_outputs).data, labels_t, topk=(1,))[0]
#         losses.update(mean_loss.item(), batch_size) # Average loss over T steps
#         top1.update(prec1, images.size(0))

#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % 10 == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                 epoch, i, len(train_loader), batch_time=batch_time,
#                 loss=losses, top1=top1))
 
@torch.no_grad()
def metrics_print(net, expert_fn, n_classes, loader, T, prior_correct=0.5, save_metrics_csv=None):
    """
    Evaluate L2D-GRU sequentially with per-sequence h_prev carried across timesteps.
    Defer class index = n_classes.
    """
    net.eval()

    step_metrics = [{
        "correct_sys": 0, "exp": 0, "correct_cls": 0, "alone_correct": 0,
        "real_total": 0, "exp_total": 0, "cls_total": 0
    } for _ in range(T)]

    overall = {
        "correct_sys": 0, "exp": 0, "correct_cls": 0, "alone_correct": 0,
        "real_total": 0, "exp_total": 0, "cls_total": 0
    }

    for images, labels, seq_ids, ts in loader:
        images = images.to(device)
        labels = labels.to(device)

        seq_ids_np = seq_ids.cpu().numpy()   # CPU numpy
        ts_np      = ts.cpu().numpy()        # CPU numpy

        # Process each sequence individually
        for sid in np.unique(seq_ids_np):
            idxs_cpu = np.nonzero(seq_ids_np == sid)[0]        # CPU numpy indices
            if idxs_cpu.size == 0:
                continue
            # Sort by t ascending
            order = np.argsort(ts_np[idxs_cpu])                # CPU numpy order
            idxs_cpu = idxs_cpu[order]                         # CPU numpy sorted indices

            # Convert to tensor; move to CUDA when indexing CUDA tensors
            idxs = torch.from_numpy(idxs_cpu).long()
            # Initialize h_prev for this sequence
            h_prev = torch.full((1, 1), float(prior_correct), device=device)

            for ii_cpu in idxs:
                ii = ii_cpu.item()

                # Get single sample to CUDA: move index to same device
                ii_cuda = torch.tensor([ii], device=device, dtype=torch.long)
                img_t   = images.index_select(0, ii_cuda)        # [1,3,32,32]
                lab_t   = labels.index_select(0, ii_cuda)        # [1]
                t_val   = float(ts_np[ii])                       # scalar (python float)
                t_norm  = torch.tensor([[t_val / (T - 1)]], device=device)

                # Model output (log-probs)
                logp = net(img_t, h_prev, t_norm)                # [1, K+1]
                pred_all = logp.argmax(dim=1)                    # [1]
                is_defer = (pred_all.item() == n_classes)

                alone_pred = logp[:, :n_classes].argmax(dim=1)   # [1]

                # Expert prediction (according to your current signature: images, labels, ts)
                exp_pred = expert_fn(img_t, lab_t, torch.tensor([t_val], device=device))

                # Accumulate to corresponding time step
                t_idx = int(t_val)
                m = step_metrics[t_idx]

                is_alone_correct = int((alone_pred[0] == lab_t[0]).item())
                m["alone_correct"] += is_alone_correct
                overall["alone_correct"] += is_alone_correct

                if not is_defer:
                    is_cls_correct = int((pred_all[0] == lab_t[0]).item())
                    m["correct_cls"] += is_cls_correct
                    m["cls_total"]   += 1
                    m["correct_sys"] += is_cls_correct

                    overall["correct_cls"] += is_cls_correct
                    overall["cls_total"]   += 1
                    overall["correct_sys"] += is_cls_correct
                else:
                    is_exp_correct = int((exp_pred[0] == lab_t[0].item()))
                    m["exp"]       += is_exp_correct
                    m["exp_total"] += 1
                    m["correct_sys"] += is_exp_correct

                    overall["exp"]       += is_exp_correct
                    overall["exp_total"] += 1
                    overall["correct_sys"] += is_exp_correct

                m["real_total"]       += 1
                overall["real_total"] += 1

                # Update h_prev to "whether expert is correct" for this step
                h_prev = torch.tensor([[1.0 if (exp_pred[0] == lab_t[0].item()) else 0.0]],
                                      device=device, dtype=torch.float32)

    save_metrics = []
    # Output for each time step
    print("\n--- Per-Time-Step Metrics ---")
    for t in range(T):
        m = step_metrics[t]
        if m["real_total"] > 0:
            coverage         = m["cls_total"] / m["real_total"]
            system_acc       = 100.0 * m["correct_sys"] / m["real_total"]
            expert_acc       = 100.0 * m["exp"] / (m["exp_total"] + 1e-9)
            classifier_acc   = 100.0 * m["correct_cls"] / (m["cls_total"] + 1e-9)
            alone_classifier = 100.0 * m["alone_correct"] / m["real_total"]

            print(f"\n--- Metrics for Time Step {t} ---")
            print(f"coverage: {coverage:.4f}")
            print(f"system accuracy: {system_acc:.3f}")
            print(f"expert accuracy: {expert_acc:.3f}")
            print(f"classifier accuracy: {classifier_acc:.3f}")
            print(f"alone classifier: {alone_classifier:.3f}")
        if save_metrics_csv is not None:
            save_metrics.append({
                "time_step": t,
                "coverage": coverage,
                "system_acc": system_acc,
                "expert_acc": expert_acc,
                "classifier_acc": classifier_acc,
                "alone_classifier": alone_classifier
            })

    # Global output
    print("\n--- Overall Metrics (across all seen timesteps) ---")
    if overall["real_total"] > 0:
        overall_coverage         = overall["cls_total"] / overall["real_total"]
        overall_system_acc       = 100.0 * overall["correct_sys"] / overall["real_total"]
        overall_expert_acc       = 100.0 * overall["exp"] / (overall["exp_total"] + 1e-9)
        overall_classifier_acc   = 100.0 * overall["correct_cls"] / (overall["cls_total"] + 1e-9)
        overall_alone_classifier = 100.0 * overall["alone_correct"] / overall["real_total"]

        print({
            "coverage": f"{overall['cls_total']} out of {overall['real_total']} "
                        f"({overall_coverage:.4f})",
            "system accuracy": overall_system_acc,
            "expert accuracy": overall_expert_acc,
            "classifier accuracy": overall_classifier_acc,
            "alone classifier": overall_alone_classifier
        })
        if save_metrics_csv is not None:
            save_metrics.append({
                "time_step": "overall",
                "coverage": overall_coverage,
                "system_acc": overall_system_acc,
                "expert_acc": overall_expert_acc,
                "classifier_acc": overall_classifier_acc,
                "alone_classifier": overall_alone_classifier
            })
            pd.DataFrame(save_metrics).to_csv(save_metrics_csv, index=False)
# def metrics_print(net, expert_fn, n_classes, loader, T):
#     """
#     Evaluates model performance on a sequential dataset.

#     Args:
#         net: The neural network model (e.g., WideResNet).
#         expert_fn: The function that generates expert predictions.
#         n_classes: The number of classification classes.
#         loader: The data loader, which provides (images, labels, seq_ids, ts).
#         T: The length of each sequence.
#     """
#     net.eval()
    
#     # Store metrics for each time step 't'
#     step_metrics = [{
#         "correct_sys": 0, "exp": 0, "correct_cls": 0, "alone_correct": 0,
#         "real_total": 0, "exp_total": 0, "cls_total": 0
#     } for _ in range(T)]

#     # Store overall metrics
#     overall_metrics = {
#         "correct_sys": 0, "exp": 0, "correct_cls": 0, "alone_correct": 0,
#         "real_total": 0, "exp_total": 0, "cls_total": 0
#     }

#     with torch.no_grad():
#         for images, labels, seq_ids, ts in loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             ts = ts.to(device)
#             batch_size = images.size(0)

#             # Get model and expert outputs
#             outputs = net(images)
#             exp_predictions = expert_fn(images, labels, ts)
#             _, predicted = torch.max(outputs.data, 1)

#             # Process each sample in the batch
#             for i in range(batch_size):
#                 t = ts[i].item() # Get the time step for the current sample
                
#                 # Check if the model has a deferral class (n_classes)
#                 # Your old code checks for `predicted[i] == 10`.
#                 # If your WideResNet has n_classes + 1 outputs, `10` is the deferral class.
#                 is_deferred = (predicted[i].item() == n_classes)
                
#                 # Get the classifier's prediction (even if it defers)
#                 alone_pred = predicted[i]
#                 if is_deferred:
#                     # If deferred, the classifier's 'final' prediction
#                     # is the second highest confidence among the classification classes.
#                     # Your original logic is flawed here; a simpler approach is to
#                     # treat the highest confidence among the n_classes as the prediction.
#                     # For simplicity and correctness with WideResNet, let's just get
#                     # the argmax over the classification classes (excluding the deferral).
#                     alone_pred = torch.max(outputs.data[i, :n_classes], 0)[1]

#                 # Accumulate metrics for the current time step and overall
#                 metrics = step_metrics[t]
                
#                 # alone_correct: The accuracy of the classifier alone (never defers)
#                 is_alone_correct = (alone_pred == labels[i]).item()
#                 metrics["alone_correct"] += is_alone_correct
#                 overall_metrics["alone_correct"] += is_alone_correct

#                 if not is_deferred:
#                     # Classifier's decision
#                     is_cls_correct = (predicted[i] == labels[i]).item()
#                     metrics["correct_cls"] += is_cls_correct
#                     metrics["cls_total"] += 1
#                     metrics["correct_sys"] += is_cls_correct
                    
#                     overall_metrics["correct_cls"] += is_cls_correct
#                     overall_metrics["cls_total"] += 1
#                     overall_metrics["correct_sys"] += is_cls_correct
#                 else:
#                     # Expert's decision
#                     is_exp_correct = (exp_predictions[i] == labels[i].item())
#                     metrics["exp"] += is_exp_correct
#                     metrics["exp_total"] += 1
#                     metrics["correct_sys"] += is_exp_correct
                    
#                     overall_metrics["exp"] += is_exp_correct
#                     overall_metrics["exp_total"] += 1
#                     overall_metrics["correct_sys"] += is_exp_correct
                    
#                 metrics["real_total"] += 1
#                 overall_metrics["real_total"] += 1

#     # Print per-time-step metrics
#     print("\n--- Per-Time-Step Metrics ---")
#     for t in range(T):
#         metrics = step_metrics[t]
#         if metrics["real_total"] > 0:
#             coverage = metrics["cls_total"] / metrics["real_total"]
#             system_acc = 100 * metrics["correct_sys"] / metrics["real_total"]
#             expert_acc = 100 * metrics["exp"] / (metrics["exp_total"] + 1e-5)
#             classifier_acc = 100 * metrics["correct_cls"] / (metrics["cls_total"] + 1e-5)
#             alone_classifier_acc = 100 * metrics["alone_correct"] / metrics["real_total"]
            
#             print(f"\n--- Metrics for Time Step {t+1} ---")
#             print(f"coverage: {coverage:.4f}")
#             print(f"system accuracy: {system_acc:.3f}")
#             print(f"expert accuracy: {expert_acc:.3f}")
#             print(f"classifier accuracy: {classifier_acc:.3f}")
#             print(f"alone classifier: {alone_classifier_acc:.3f}")
    
#     # Print overall metrics
#     print("\n--- Overall Metrics (Average over all time steps) ---")
#     if overall_metrics["real_total"] > 0:
#         overall_coverage = overall_metrics["cls_total"] / overall_metrics["real_total"]
#         overall_system_acc = 100 * overall_metrics["correct_sys"] / overall_metrics["real_total"]
#         overall_expert_acc = 100 * overall_metrics["exp"] / (overall_metrics["exp_total"] + 1e-5)
#         overall_classifier_acc = 100 * overall_metrics["correct_cls"] / (overall_metrics["cls_total"] + 1e-5)
#         overall_alone_classifier_acc = 100 * overall_metrics["alone_correct"] / overall_metrics["real_total"]
        
#         to_print = {
#             "coverage": f"{overall_metrics['cls_total']} out of {overall_metrics['real_total']}",
#             "system accuracy": overall_system_acc,
#             "expert accuracy": overall_expert_acc,
#             "classifier accuracy": overall_classifier_acc,
#             "alone classifier": overall_alone_classifier_acc
#         }
#         print(to_print)
# def validate_reject(val_loader, model, epoch, expert_fn, n_classes):
#     """Perform validation on the validation set with deferral"""
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()

#     end = time.time()
#     for i, (input, target) in enumerate(val_loader):
#         target = target.to(device)
#         input = input.to(device)

#         # compute output
#         with torch.no_grad():
#             output = model(input)
#         # expert prediction
#         batch_size = output.size()[0]  # batch_size
#         m = expert_fn(input, target)
#         alpha = 1
#         m2 = [0] * batch_size
#         for j in range(0, batch_size):
#             if m[j] == target[j].item():
#                 m[j] = 1
#                 m2[j] = alpha
#             else:
#                 m[j] = 0
#                 m2[j] = 1
#         m = torch.tensor(m)
#         m2 = torch.tensor(m2)
#         m = m.to(device)
#         m2 = m2.to(device)
#         # compute loss
#         loss = reject_CrossEntropyLoss(output, m, target, m2, n_classes)

#         # measure accuracy and record loss
#         prec1 = accuracy(output.data, target, topk=(1,))[0]
#         losses.update(loss.data.item(), input.size(0))
#         top1.update(prec1.item(), input.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % 10 == 0:
#             print('Test: [{0}/{1}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                 i, len(val_loader), batch_time=batch_time, loss=losses,
#                 top1=top1))

#     print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

#     return top1.avg
def run_reject(model, data_aug, n_dataset, expert_fn, epochs, alpha, T):
    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if n_dataset == 10:
        dataset = 'cifar10'
    elif n_dataset == 100:
        dataset = 'cifar100'

    kwargs = {'num_workers': 0, 'pin_memory': True}

    # Load the full CIFAR-10 dataset
    train_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=True, download=True,
                                                           transform=transform_train)
    
    # Calculate sizes based on the number of full sequences
    num_total_samples = len(train_dataset_all)
    num_total_sequences = num_total_samples // T
    
    train_seq_size = int(0.90 * num_total_sequences)
    test_seq_size = num_total_sequences - train_seq_size

    # Randomly split the dataset based on sequence chunks
    # This ensures that all images within a sequence stay together
    indices = torch.randperm(num_total_sequences)
    train_indices_seq = indices[:train_seq_size]
    test_indices_seq = indices[train_seq_size:]

    # Create the training and testing datasets with proper indices
    train_indices = []
    for seq_idx in train_indices_seq:
        train_indices.extend(range(seq_idx * T, (seq_idx + 1) * T))
    
    test_indices = []
    for seq_idx in test_indices_seq:
        test_indices.extend(range(seq_idx * T, (seq_idx + 1) * T))

    # Create subsets from the original dataset using the calculated indices
    train_subset = torch.utils.data.Subset(train_dataset_all, train_indices)
    test_subset = torch.utils.data.Subset(train_dataset_all, test_indices)

    # Wrap the subsets with IndexedCIFAR10
    train_dataset_indexed = IndexedCIFAR10(train_subset, T)
    test_dataset_indexed = IndexedCIFAR10(test_subset, T)

    # Create DataLoaders with shuffle=False to maintain sequential order
    train_loader = torch.utils.data.DataLoader(train_dataset_indexed,
                                               batch_size=1000, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset_indexed,
                                               batch_size=1000, shuffle=False, **kwargs)
    # amp = 5
    # train_loader, test_loader = make_augmented_sequence_loaders(
    #     base_dataset=train_dataset_all ,
    #     T=T,
    #     amp=amp,
    #     stride=None,
    #     train_ratio=0.9,
    #     safe_split=True, 
    #     seed=42,
    #     batch_seqs=16,
    #     num_workers=0,
    #     pin_memory=True
    # )
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print(len(train_loader))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)

    # optionally resume from a checkpoint

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)
    out_path = os.path.join(save_root, f"general_model_curve_mixed_seed{seed}.pth")
    patience = 30

    for epoch in range(epochs):
        train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_dataset, alpha, T)
        val_loss = eval_seq_loss(test_loader, model, expert_fn, n_dataset, T, alpha, prior=0.0)
        print(f"[epoch {epoch}] val_loss = {val_loss:.6f}")
        best_val = val_loss
        no_improve = 0
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), out_path)
            print(f"new best! saved to {out_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch = {best_epoch}, best val_loss = {best_val:.6f})")
                break

        if epoch % 10 == 0:
            metrics_print(model, expert_fn, n_dataset, test_loader, T)

    if os.path.exists(out_path):
        state = torch.load(out_path, map_location=device)
        model.load_state_dict(state)
        print(f"[restore] loaded best model from epoch {best_epoch} (val_loss={best_val:.6f})")

    metrics_print(model, expert_fn, n_dataset, test_loader, T,
                  save_metrics_csv=os.path.join(result_root, f"general_model_curve_mixed_seed{seed}.csv"))


def eval_reject(model, data_aug, n_dataset, expert_fn, epochs, alpha, T):
    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if n_dataset == 10:
        dataset = 'cifar10'
    elif n_dataset == 100:
        dataset = 'cifar100'

    kwargs = {'num_workers': 0, 'pin_memory': True}

    test_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=False, download=True, transform=transform_test)
    test_dataset_indexed = IndexedCIFAR10(test_dataset_all, T)
    test_loader = torch.utils.data.DataLoader(test_dataset_indexed,
                                               batch_size=128, shuffle=True, **kwargs)
    model = model.to(device)

    cudnn.benchmark = True

    metrics_print(model, expert_fn, n_dataset, test_loader, T,
                  save_metrics_csv=os.path.join(result_root, f"general_model_curve_mixed_seed{seed}.csv"))

n_dataset = 10

class SeqExpert:
    def __init__(
        self,
        n_classes: int,
        acc_curve,
        other_acc,
        k: int,
        cycle: bool = True,
        seed: int = 66,
    ):
        self.n_classes = int(n_classes)
        self.k = int(k)
        self.acc_curve = np.asarray(acc_curve, dtype=float).clip(0.0, 1.0)
        self.T = len(self.acc_curve)
        self.cycle = cycle
        
        if other_acc is None:
            self.other_acc_curve = np.full(self.T, 1.0 / self.n_classes, dtype=float)
        else:
            arr = np.asarray(other_acc, dtype=float)
            if arr.ndim == 0:
                self.other_acc_curve = np.full(self.T, float(arr), dtype=float)
            else:
                assert len(arr) == self.T, "other_acc_curve length must match acc_curve"
                self.other_acc_curve = arr.clip(0.0, 1.0)

        self.rng = np.random.default_rng(seed)


    @torch.no_grad()
    def predict(self, inputs, labels: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        device = labels.device
        B = labels.size(0)
        
        # Use values in t_tensor as indices
        idx = t_tensor.long().cpu().numpy()

        p_master = self.acc_curve[idx]            # (B,)
        p_other  = self.other_acc_curve[idx]      # (B,)

        y_true = labels.detach().long().cpu().numpy()  # (B,)
        mask_master = (y_true < self.k)

        u = self.rng.random(B)
        hit = np.empty(B, dtype=bool)
        hit[mask_master] = (u[mask_master] < p_master[mask_master])
        hit[~mask_master] = (u[~mask_master] < p_other[~mask_master])

        def sample_any(y_np):
            any_class = self.rng.integers(0, self.n_classes, size=y_np.shape)
            return any_class

        preds = np.empty(B, dtype=np.int64)
        preds[hit]  = y_true[hit]
        preds[~hit] = sample_any(y_true[~hit])

        return torch.from_numpy(preds).to(device=device, dtype=torch.long)

import os, random, numpy as np, torch

run_seeds = [42, 43, 44, 45, 46]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

alpha = 1
epochs = 200
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_root = f"models/gru_curve_ft/20250923_235549"
result_root = f"results/gru_curve_ft_test/{timestamp}"
os.makedirs(save_root, exist_ok=True)
os.makedirs(result_root, exist_ok=True)

T = 10
gap = len(p_curve) // T
p_curve_new = []
for i in range(T):
    start = i * gap
    end = min((i + 1) * gap, len(p_curve))
    segment = p_curve[start:end]
    avg = np.mean(segment)
    p_curve_new.append(avg)

p_curve = np.array(p_curve_new)
p_low_curve = np.full(T, 0.1)
print(len(p_curve))
print(len(p_low_curve))

# gap = len(p_curve) // T
# start = gap // 2 - 1
# p_curve_new = []
# for i in range(T):
#     p_curve_new.append(p_curve[start + i * gap])
# p_low_curve = np.full(T, 0.1)
# p_curve = np.array(p_curve_new)


for seed in run_seeds:
    print(f"\n================ SEED {seed} ================\n")
    set_seed(seed)

    expert = SeqExpert(n_dataset, p_curve, p_low_curve, k=7, seed=seed)

    backbone = WideResNetRevised(28, n_dataset, 4, dropRate=0.0)
    model = L2DGRU(backbone, hidden_dim=512, num_layers=1, n_classes=n_dataset, dropout=0.0)

    model = torch.load(os.path.join(save_root, f"general_model_curve_mixed_seed{seed}.pth"), map_location=device, weights_only=False)

    eval_reject(model, False, n_dataset, expert.predict, epochs, alpha, T)
    


    