from common.utils import AverageMeter, accuracy
import time
import torch
from common.model import WideResNet
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def my_CrossEntropyLoss(outputs, labels):
    batch_size = outputs.size()[0]            # batch_size
    outputs =  - torch.log2(outputs[range(batch_size), labels]+0.00001)   # pick the values corresponding to the labels
    return torch.sum(outputs)/batch_size

def train_expert(train_loader_exp, model, optimizer, scheduler, epoch, dataset_expert, train_loader):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader_exp):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        # compute new target
        batch_size = output.size()[0]            # batch_size
        m = [0] * batch_size
        for j in range (0,batch_size):
            m[j] = dataset_expert[str(input[j].cpu().numpy())]
        m = torch.tensor(m)
        m = m.to(device)
        # compute loss
        loss = my_CrossEntropyLoss(output, m)


        # measure accuracy and record loss
        prec1 = accuracy(output.data, m, topk=(1,))[0]
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


def validate_expert(val_loader_exp, model, val_loader, dataset_expert):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader_exp):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output = model(input)
        # expert prediction
        batch_size = output.size()[0]            # batch_size
        m = [0] * batch_size
        for j in range (0,batch_size):
            m[j] = dataset_expert[str(input[j].cpu().numpy())]

        m = torch.tensor(m)
        m = m.to(device)
        # compute loss
        loss = my_CrossEntropyLoss(output, m)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, m, topk=(1,))[0]
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
best_prec1 = 0
def run_expert(model, data_aug, n_dataset, expert_fn, epochs, dataset_expert, train_loader, val_loader, val_loader_rej):
    global best_prec1
    # Data loading code
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)

    # optionally resume from a checkpoint
    

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9, nesterov = True,
                                weight_decay=5e-4)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*200)

    for epoch in range(0, epochs):
        # train for one epoch
        train_expert(val_loader, model, optimizer, scheduler, epoch, expert_fn, n_dataset)

        # evaluate on validation set
        prec1 = validate_expert(val_loader_rej, model, epoch, expert_fn, n_dataset)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

    print('Best accuracy: ', best_prec1)



def metrics_print_confid_cifar10h(net_mod, net_exp, dataset_expert_probs, n_classes, loader):
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs_mod = net_mod(images)
            outputs_exp = net_exp(images)
            _, predicted = torch.max(outputs_mod.data, 1)
            _, predicted_exp = torch.max(outputs_exp.data, 1)
            batch_size = outputs_mod.size()[0]            # batch_size
            for i in range(0,batch_size):
                r_score = 1 - outputs_mod.data[i][predicted[i].item()].item()
                r_score = r_score - outputs_exp.data[i][1].item()
                r = 0
                if r_score >= 0:
                    r = 1
                else:
                    r =  0
                if r==0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r==1:
                    exp_prediction = np.argmax(np.random.multinomial(1, dataset_expert_probs[str(images[i].cpu().numpy())]))
                    exp += (exp_prediction == labels[i].item())
                    correct_sys +=(exp_prediction == labels[i].item())
                    exp_total+=1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001) }
    print(to_print)
    return [100*total/real_total,  100*correct_sys/real_total, 100* exp/(exp_total+0.0002),100*correct/(total+0.0001) ]
