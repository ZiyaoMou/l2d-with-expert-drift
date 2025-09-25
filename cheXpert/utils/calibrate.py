import torch
import torch.nn as nn
import torch.optim as optim

def CrossEntropyLoss(outputs, labels, weights):
    # m: expert costs, labels: ground truth, n_classes: number of classes
    '''
    outputs: model outputs
    labels: target vector
    weights: uncertainty flagging
    '''
    batch_size = outputs.size()[0]  # batch_size
    total_loss = 0
    j = 0
    for i in range(14):
        out_softmax = torch.nn.functional.softmax(outputs[range(batch_size),j:j+2])
        j += 2 # index variable update, for CE use 2, for LCE use 3
        loss_class = - weights[range(batch_size),i] * torch.log2(out_softmax[range(batch_size), labels[range(batch_size),i].long()])   
        loss_class = torch.sum(loss_class)/(torch.sum(weights[range(batch_size),i] ) +0.000000001)  # average loss
        total_loss += loss_class # total loss is the sum of all class losses
    total_loss /= 14
    return total_loss
    

class ModelWithTemperature(nn.Module):
    """
    Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(14) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        batch_size = logits.size()[0]  # batch_size
        j = 0
        logits_temp = logits.clone()
        for i in range(14):
            logits_temp[range(batch_size),j:j+2] = logits_temp[range(batch_size),j:j+2].clone()/self.temperature[i]
            j += 2 # index variable update, for CE use 2, for LCE use 3
        return logits_temp

    def set_temp(self, valid_loader):
        self.cuda()
        self.model.eval()
        logits_list = []
        logits_temp_list = []
        labels_list = []
        weights_list = []
        with torch.no_grad():
            for i, (input, target, rad1, rad2, rad3, weights) in enumerate(valid_loader):
                target = target.cuda()
                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
                out = self.model(varInput)
                out_temp = self.temperature_scale(out)
                weights_list.append(weights.cuda(non_blocking=True))
                logits_list.append(out.cuda())
                labels_list.append(target)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            weights = torch.cat(weights_list).cuda()
        before_temperature_nll = CrossEntropyLoss(logits, labels, weights).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=1000)
        def eval():
            loss = CrossEntropyLoss(self.temperature_scale(logits), labels, weights)
            optimizer.zero_grad()
            loss.backward(retain_graph=True )
            return loss
        optimizer.step(eval)
        after_temperature_nll = CrossEntropyLoss(self.temperature_scale(logits), labels, weights).item()
        print('After temperature - NLL: %.3f' % (after_temperature_nll))
        print(f'temperatures are {self.temperature}')


class ModelWithTemperature_rad(nn.Module):
    """
    Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature_rad, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(14) * 1)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        batch_size = logits.size()[0]  # batch_size
        j = 0
        logits_temp = logits.clone()
        for i in range(14):
            logits_temp[range(batch_size),j:j+2] = logits_temp[range(batch_size),j:j+2].clone()/self.temperature[i]
            j += 2 # index variable update, for CE use 2, for LCE use 3
        return logits_temp

    def set_temp(self, valid_loader):
        self.cuda()
        self.model.eval()
        logits_list = []
        logits_temp_list = []
        labels_list = []
        weights_list = []
        with torch.no_grad():
            for i, (input, target, rad1, rad2, rad3, weights) in enumerate(valid_loader):
                target = target.cuda()
                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
                out = self.model(varInput)
                out_temp = self.temperature_scale(out)
                weights_list.append(weights.cuda(non_blocking=True))
                logits_list.append(out.cuda())
                labels_list.append(target == rad1.cuda())
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            weights = torch.cat(weights_list).cuda()
        before_temperature_nll = CrossEntropyLoss(logits, labels, weights).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=1000)
        def eval():
            loss = CrossEntropyLoss(self.temperature_scale(logits), labels, weights)
            optimizer.zero_grad()
            loss.backward(retain_graph=True )
            return loss
        optimizer.step(eval)
        after_temperature_nll = CrossEntropyLoss(self.temperature_scale(logits), labels, weights).item()
        print('After temperature - NLL: %.3f' % (after_temperature_nll))
        print(f'temperatures are {self.temperature}')
