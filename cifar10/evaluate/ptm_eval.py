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

exp_curve = pd.read_csv('/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/expert/expert_acc_curve_new_new.csv')
p_curve = exp_curve['p']

exp_low_curve = pd.read_csv('/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/expert/expert_acc_curve_low_new.csv')
p_low_curve = exp_low_curve['p']

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

def train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, alpha):
    """Train for one epoch on the training set with deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model = model.to(device)
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)
        output = model(input)
        batch_size = output.size()[0]  # batch_size
        m = expert_fn(input, target)
        m2 = [0] * batch_size
        for j in range(0, batch_size):
            if m[j] == target[j].item():
                m[j] = 1
                m2[j] = alpha
            else:
                m[j] = 0
                m2[j] = 1
        m = torch.tensor(m)
        m2 = torch.tensor(m2)
        m = m.to(device)
        m2 = m2.to(device)
        # done getting expert predictions and costs 
        # compute loss
        criterion = nn.CrossEntropyLoss()
        loss = reject_CrossEntropyLoss(output, m, target, m2, n_classes)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

def train_reject_class(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, alpha):
    """Train for one epoch on the training set without deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)

        # compute loss
        loss = my_CrossEntropyLoss(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))


def validate_reject(val_loader, model, epoch, expert_fn, n_classes):
    """Perform validation on the validation set with deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output = model(input)
        # expert prediction
        batch_size = output.size()[0]  # batch_size
        m = expert_fn(input, target)
        alpha = 1
        m2 = [0] * batch_size
        for j in range(0, batch_size):
            if m[j] == target[j].item():
                m[j] = 1
                m2[j] = alpha
            else:
                m[j] = 0
                m2[j] = 1
        m = torch.tensor(m)
        m2 = torch.tensor(m2)
        m = m.to(device)
        m2 = m2.to(device)
        # compute loss
        loss = reject_CrossEntropyLoss(output, m, target, m2, n_classes)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


# def metrics_print(net, expert_fn, n_classes, loader, result_save_dir, timestep):
#     net = net.to(device)          # ensure model is on correct device
#     net.eval()

#     correct = 0
#     correct_sys = 0
#     exp = 0
#     exp_total = 0
#     total = 0
#     real_total = 0
#     alone_correct = 0

#     with torch.no_grad():
#         for images, labels in loader:
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)

#             outputs = net(images)   # now inputs and weights are on same device
#             _, predicted = torch.max(outputs, 1)

#             batch_size = outputs.size(0)
#             exp_prediction = expert_fn(images, labels)  # if expert_fn only uses labels, can put on CPU; if uses images, need device consistency

#             for i in range(batch_size):
#                 r = (predicted[i].item() == n_classes)  # n_classes as "reject class" index
#                 prediction = predicted[i]
#                 if predicted[i] == 10:  # note: if n_classes is not 10, this should be changed to n_classes
#                     max_idx = 0
#                     for j in range(0, 10):  # same as above, recommend using n_classes instead of hardcoded 10
#                         if outputs[i][j] >= outputs[i][max_idx]:
#                             max_idx = j
#                     prediction = max_idx
#                 else:
#                     prediction = predicted[i]

#                 alone_correct += (prediction == labels[i]).item()
#                 if r == 0:
#                     total += 1
#                     correct += (predicted[i] == labels[i]).item()
#                     correct_sys += (predicted[i] == labels[i]).item()
#                 else:
#                     exp += (exp_prediction[i] == labels[i].item())
#                     correct_sys += (exp_prediction[i] == labels[i].item())
#                     exp_total += 1
#                 real_total += 1

#     cov = f"{total} out of {real_total}"
#     to_print = {
#         "coverage": cov,
#         "system accuracy": 100 * correct_sys / real_total,
#         "expert accuracy": 100 * exp / (exp_total + 1e-4),
#         "classifier accuracy": 100 * correct / (total + 1e-4),
#         "alone classifier": 100 * alone_correct / real_total
#     }
#     print(to_print)

import os, csv
import torch

def metrics_print(net, expert_fn, n_classes, loader, result_save_dir, timestep):
    """
    Evaluate once on data slice loader for given time step, and write one row of results to result_save_dir/results.csv
    Expect loader to only contain samples of that t (like (images, labels))
    """
    os.makedirs(result_save_dir, exist_ok=True)
    csv_path = os.path.join(result_save_dir, "results.csv")

    net = net.to(device)
    net.eval()

    # Counters
    correct_sys = 0          # whether system (model or expert) is finally correct
    correct_cls = 0          # whether model is correct when making decisions
    alone_correct = 0        # whether model prediction alone (ignoring defer) is correct
    exp_correct = 0          # whether expert is correct when making decisions
    cls_total = 0            # number of samples handled by model
    exp_total = 0            # number of samples handled by expert
    real_total = 0           # total samples (this t)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Model output; assume last dimension size is n_classes+1, last index is defer class
            outputs = net(images)                 # [B, n_classes+1] or [B, K] (if no defer then need corresponding adjustment)
            pred_all = outputs.argmax(dim=1)      # [B]
            # Model "alone prediction" without defer: take argmax of first n_classes
            pred_alone = outputs[:, :n_classes].argmax(dim=1)  # [B]

            # Expert prediction
            exp_prediction = expert_fn(images, labels)  # should return integer classification results of length B (can be on CPU/GPU)

            B = labels.size(0)
            for i in range(B):
                is_defer = (pred_all[i].item() == n_classes)

                # Statistics for alone (model correctness without considering defer)
                alone_correct += int(pred_alone[i] == labels[i])

                if not is_defer:
                    # Model takes over
                    cls_total += 1
                    is_cls_correct = int(pred_all[i] == labels[i])
                    correct_cls += is_cls_correct
                    correct_sys += is_cls_correct
                else:
                    # Expert takes over
                    exp_total += 1
                    is_exp_correct = int(exp_prediction[i] == labels[i].item())
                    exp_correct += is_exp_correct
                    correct_sys += is_exp_correct

                real_total += 1

    # Calculate metrics
    coverage = (cls_total / real_total) if real_total > 0 else 0.0
    system_acc = 100.0 * correct_sys / real_total if real_total > 0 else 0.0
    expert_acc = 100.0 * exp_correct / exp_total if exp_total > 0 else 0.0
    classifier_acc = 100.0 * correct_cls / cls_total if cls_total > 0 else 0.0
    alone_classifier = 100.0 * alone_correct / real_total if real_total > 0 else 0.0

    row = {
        "t": int(timestep),
        "total": int(real_total),          # total samples for this t
        "coverage": float(coverage),
        "system_acc": float(system_acc),
        "expert_acc": float(expert_acc),
        "classifier_acc": float(classifier_acc),
        "alone_classifier": float(alone_classifier),
        "cls_total": int(cls_total),
        "exp_total": int(exp_total),
    }

    # Append to CSV (write header if not exists)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # Print a copy to console
    print(row)

def run_reject(model, data_aug, n_dataset, expert_fn, epochs, alpha):
    '''
    model: WideResNet model
    data_aug: boolean to use data augmentation in training
    n_dataset: number of classes
    expert_fn: expert model
    epochs: number of epochs to train
    alpha: alpha parameter in L_{CE}^{\alpha}
    '''
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


    train_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=True, download=True,
                                                           transform=transform_train)
    train_size = int(0.90 * len(train_dataset_all))
    test_size = len(train_dataset_all) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset_all, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=128, shuffle=True, **kwargs)
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

    for epoch in range(0, epochs):
        # train for one epoch
        train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_dataset, alpha)
        if epoch % 10 == 0:
            metrics_print(model, expert_fn, n_dataset, test_loader)



def eval_reject(model, data_aug, n_dataset, expert_fn, epochs, alpha, result_save_dir, timestep):
    '''
    model: WideResNet model
    data_aug: boolean to use data augmentation in training
    n_dataset: number of classes
    expert_fn: expert model
    epochs: number of epochs to train
    alpha: alpha parameter in L_{CE}^{\alpha}
    '''
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


    train_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=True, download=True,
                                                           transform=transform_train)
    test_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=False, download=True, transform=transform_test)
    # train_size = int(0.90 * len(train_dataset_all))
    # test_size = len(train_dataset_all) - train_size

    # train_dataset, test_dataset = torch.utils.data.random_split(train_dataset_all, [train_size, test_size])
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=128, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                            batch_size=128, shuffle=True, **kwargs)
    # get the number of model parameters
    test_loader = torch.utils.data.DataLoader(test_dataset_all,
                                               batch_size=128, shuffle=True, **kwargs)

    model = model.to(device)

    # optionally resume from a checkpoint

    cudnn.benchmark = True

    metrics_print(model, expert_fn, n_dataset, test_loader, result_save_dir, timestep)


n_dataset = 10  # cifar-10

class synth_expert:
    '''
    simple class to describe our synthetic expert on CIFAR-10
    ----
    k: number of classes expert can predict
    n_classes: number of classes (10+1 for CIFAR-10)
    '''
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() <= self.k:
                outs[i] = labels[i].item()
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs
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
        self.t = 0
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

    def _next_indices(self, batch_size: int):
        if self.cycle:
            idx = (np.arange(batch_size) + self.t) % self.T
            self.t = (self.t + batch_size) % self.T
        else:
            idx = np.minimum(np.arange(batch_size) + self.t, self.T - 1)
            self.t = min(self.t + batch_size, self.T - 1)
        return idx

    @torch.no_grad()
    def predict(self, inputs, labels: torch.Tensor) -> torch.Tensor:
        # labels: (B,) long on any device
        device = labels.device
        B = labels.size(0)
        idx = self._next_indices(B)

        p_master = self.acc_curve[idx]            # (B,)
        p_other  = self.other_acc_curve[idx]      # (B,)

        y_true = labels.detach().long().cpu().numpy()  # (B,)
        mask_master = (y_true < self.k)

        u = self.rng.random(B)
        hit = np.empty(B, dtype=bool)
        hit[mask_master] = (u[mask_master] < p_master[mask_master])
        hit[~mask_master] = (u[~mask_master] < p_other[~mask_master])

        def sample_wrong(y_np):
            r = self.rng.integers(0, self.n_classes - 1, size=y_np.shape)
            wrong = r + (r >= y_np)
            return wrong

        preds = np.empty(B, dtype=np.int64)
        preds[hit]  = y_true[hit]
        preds[~hit] = sample_wrong(y_true[~hit])

        return torch.from_numpy(preds).to(device=device, dtype=torch.long)

def build_window_experts(p_curve, p_low_curve, k=5, n_classes=10, num_windows=10, seed_base=66):
    pc = np.asarray(p_curve, dtype=float)
    pl = np.asarray(p_low_curve, dtype=float)
    L = len(pc)

    experts = []
    for w in range(num_windows):
        start = int(np.floor(w * L / num_windows))
        end   = int(np.floor((w + 1) * L / num_windows)) if w < num_windows - 1 else L

        acc_win = pc[start:end]
        oth_win = pl[start:end]

        exp_w = SeqExpert(
            n_classes=n_classes,
            acc_curve=acc_win,
            other_acc=oth_win,
            k=k,
            cycle=True,
            seed=seed_base + w,
        )
        experts.append(exp_w)
    return experts


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--seeds", type=int, default=None)
args = parser.parse_args()

run_seeds = []

if args.seeds is not None:
    run_seeds = [int(s) for s in args.seeds.split(",") if s.strip()!=""]
elif args.seed is not None:
    run_seeds = [args.seed]
else:
    run_seeds = [42, 43, 44]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_windows = 10
epochs_each = 200
alpha = 1
n_classes = n_dataset

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

base_save_dir = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/models/perstep_curve_1109"
os.makedirs(base_save_dir, exist_ok=True)

for seed in run_seeds:
    print(f"\n================ SEED {seed} ================\n")
    set_seed(seed)
    try:
        experts = build_window_experts(
            p_curve, p_low_curve, k=7, n_classes=n_classes,
            num_windows=num_windows, seed=seed
        )
    except TypeError:
        experts = build_window_experts(
            p_curve, p_low_curve, k=7, n_classes=n_classes,
            num_windows=num_windows
        )

    save_dir = os.path.join(base_save_dir, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    base_result_dir = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/results/perstep_curve_test"

    result_save_dir = os.path.join(base_result_dir, f"seed_{seed}")

    timestep = 0

    for w_id, expert in enumerate(experts):
        print(f"\n======== Training model for window {w_id} (seed={seed}) ========")
        model = WideResNet(28, n_classes + 1, 4, dropRate=0.0)
        model = torch.load(os.path.join(save_dir, f"l2d_window_{w_id:02d}.pth"), map_location=device, weights_only=False)
        eval_reject(model, False, n_classes, expert.predict, epochs_each, alpha, result_save_dir, timestep)
        timestep += 1
