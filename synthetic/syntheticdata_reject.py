from __future__ import print_function
import torch
import torch.nn.functional as F
from scipy.stats import multivariate_normal
import math
import numpy as np, scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
def sample(mu, var, nb_samples=500):
    """
    sample guassian random variable
    :param mu: torch.Tensor (features)
    :param var: torch.Tensor (features) (note: zero covariance)
    :return: torch.Tensor (nb_samples, features)
    """
    out = []
    for i in range(nb_samples):
        out += [
            torch.normal(mu, var.sqrt())
        ]
    return torch.stack(out, dim=0)
class Linear_net(nn.Module):
    '''
    Linear multiclass classifier with unit init
    '''
    def __init__(self, input_dim, out_dim):
        super(Linear_net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, out_dim)
        torch.nn.init.ones_(self.fc1.weight)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.softmax(x)
        return x

    
class Linear_net_sig(nn.Module):
    '''
    Linear binary classifier with unit init
    '''
    def __init__(self, input_dim, out_dim = 1):
        super(Linear_net_sig, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, 1)
        torch.nn.init.ones_(self.fc1.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x    


def run_classifier(net, data_x, data_y):
    '''
    trains multiclass classifier using SGD
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1, momentum=0)
    for epoch in range(50):  # loop over the dataset multiple times

        running_loss = 0.0
        # get the inputs; data is a list of [inputs, labels]
        inputs = data_x
        labels = data_y
        order = np.array(range(len(data_x)))
        np.random.shuffle(order)
        # in-place changing of values
        inputs[np.array(range(len(data_x)))] = inputs[order]
        labels[np.array(range(len(data_x)))] = labels[order]

        # zero the parameter gradients
        optimizer.zero_grad()
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(inputs2)*100)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        running_loss += loss.item()
        #print("loss " + str(loss.item()))

    
# def run_classifier_sig(net, data_x, data_y):
#     '''
#     trains binary classifier using SGD
#     '''
#     BCE = torch.nn.BCELoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0)
#     for epoch in range(100):  # loop over the dataset multiple times

#         running_loss = 0.0
#         # get the inputs; data is a list of [inputs, labels]
#         inputs = data_x
#         labels = data_y
#         order = np.array(range(len(data_x)))
#         np.random.shuffle(order)
#         # in-place changing of values
#         inputs[np.array(range(len(data_x)))] = inputs[order]
#         labels[np.array(range(len(data_x)))] = labels[order]

#         # zero the parameter gradients
#         #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(inputs)*100)

#         # forward + backward + optimize
#         outputs = net(inputs)
#         #loss = -labels*torch.log2(outputs) - (1-labels)*torch.log2(1-outputs) #BCE(outputs, labels)
#         #loss = torch.sum(loss)/ len(inputs)
#         loss = BCE(outputs, labels) 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         #scheduler.step()
#         running_loss += loss.item()
#         print("loss " + str(loss.item()))

#     #print('Finished Training')

def test_classifier(net, data_x, data_y):
    '''
    tests multiclass classifier and prints accuracy
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        inputs =  data_x
        labels = data_y
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the  test examples: %d %%' % (
        100 * correct / total))
# def test_classifier_sig(net, data_x, data_y):
#     '''
#     tests binary classifier and prints accuracy
#     '''
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         inputs =  data_x
#         labels = data_y
#         outputs = net(inputs)
#         predicted = torch.round(outputs.data)
#         total = labels.size(0)
#         for i in range(total):
#             correct += predicted[i].item() == labels[i].item()
#         #correct = (predicted == labels).sum()
#     print('Accuracy of the network on the  test examples: %d %%' % (
#         100 * correct / total))

class Linear_net_rej(nn.Module):
    '''
    Linear Classifier to be used for the L_CE loss
    '''
    def __init__(self, input_dim, out_dim):
        super(Linear_net_rej, self).__init__()
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(input_dim, out_dim+1)
        self.fc_rej = nn.Linear(input_dim, 1)
        torch.nn.init.ones_(self.fc.weight)
        torch.nn.init.ones_(self.fc_rej.weight)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.fc(x)
        rej = self.fc_rej(x)
        #out = torch.cat([out,rej],1)
        out = self.softmax(out)
        return out

def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    '''
    Implmentation of L_{CE}^{\alpha}
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting (I_{m =y})
        labels: target
        m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
        n_classes: number of classes
    '''    
    batch_size = outputs.size()[0]            # batch_size
    rc = [n_classes] * batch_size
    rc = torch.tensor(rc)
    outputs =  -m*torch.log2( outputs[range(batch_size), rc]) - m2*torch.log2(outputs[range(batch_size), labels])   # pick the values corresponding to the labels
    return torch.sum(outputs)/batch_size
def run_classifier_rej(net, net_exp, data_x, data_y, alpha):
    '''
    training script for L_{CE}
        net: classifier and rejector model
        net_exp: expert model
        data_x: numpy x data
        data_y: numpy y data
        alpha: hyperparam alpha for loss L_CE^{\alpha}
    '''
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data_x)*50)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        inputs = data_x
        labels = data_y
        order = np.array(range(len(data_x)))
        np.random.shuffle(order)
        # in-place changing of values
        inputs[np.array(range(len(data_x)))] = inputs[order]
        labels[np.array(range(len(data_x)))] = labels[order]
        x_batches = torch.split(inputs,64)
        y_batches = torch.split(labels,64)  
        for inputs, labels in zip(x_batches, y_batches):
            # get the inputs; data is a list of [inputs, labels]


            #order = np.array(range(len(data_x)))
            #np.random.shuffle(order)
            # in-place changing of values
            #inputs[np.array(range(len(data_x)))] = inputs[order]
            #labels[np.array(range(len(data_x)))] = labels[order]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            m = net_exp(inputs)
            _, predicted = torch.max(m.data, 1)
            m = (predicted==labels)*1
            m2 = [0] * len(inputs)
            for j in range (0,len(inputs)):
                if m[j]:
                    m2[j] = alpha
                else:
                    m2[j] = 1
            m = torch.tensor(m)
            m2 = torch.tensor(m2)
            outputs = net(inputs)
            loss = reject_CrossEntropyLoss(outputs, m, labels, m2, 2)
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            #print("loss " + str(loss.item()))

    #print('Finished Training')

def test_classifier_rej(net, net_exp, data_x, data_y):
    '''
    Testing script for L_{CE} loss
    '''
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    with torch.no_grad():
        inputs =  data_x
        labels = data_y
        m = net_exp(inputs)
        _, predicted_exp = torch.max(m.data, 1)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(inputs)):
            r = (predicted[i] == 2).item()
            if r:
                exp += (predicted_exp[i] == labels[i]).item()
                correct_sys += (predicted_exp[i] == labels[i]).item()
                exp_total += 1
            else:
                correct += (predicted[i] == labels[i]).item() 
                correct_sys += (predicted[i] == labels[i]).item()
                total += 1
        real_total += labels.size(0)
    cov = str(total) + str(" out of") + str(real_total)
    to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001), "alone classifier": 100*alone_correct/real_total }
    #print(to_print)
    return [100*total/real_total,  100*correct_sys/real_total, 100* exp/(exp_total+0.0002),100*correct/(total+0.0001) ]


# def test_confidence(net, net_exp, data_x, data_y):
#     '''
#     Confidence baseline, compares confidence of net and net_exp and rejects accordingly
#     '''
#     correct = 0
#     correct_sys = 0
#     exp = 0
#     exp_total = 0
#     total = 0
#     real_total = 0
#     alone_correct = 0
#     with torch.no_grad():
#         inputs =  data_x
#         labels = data_y
#         m = net_exp(inputs)
#         _, predicted_exp = torch.max(m.data, 1)
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         for i in range(len(inputs)):
#             r_score = 1 - outputs.data[i][predicted[i].item()].item()
#             r_score = r_score - (1-m.data[i][predicted_exp[i].item()].item())
#             r = 0
#             if r_score >= 0:
#                 r = 1
#             else:
#                 r =  0
#             if r:
#                 exp += (predicted_exp[i] == labels[i]).item()
#                 correct_sys += (predicted_exp[i] == labels[i]).item()
#                 exp_total += 1
#             else:
#                 correct += (predicted[i] == labels[i]).item() 
#                 correct_sys += (predicted[i] == labels[i]).item()
#                 total += 1
#         real_total += labels.size(0)
#     cov = str(total) + str(" out of") + str(real_total)
#     to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001), "alone classifier": 100*alone_correct/real_total }
#     #print(to_print)
#     return [100*total/real_total,  100*correct_sys/real_total, 100* exp/(exp_total+0.0002),100*correct/(total+0.0001) ]


# def test_oracle(net_class, net_exp, net_rej, data_x, data_y):
#     '''
#     Baseline: 2 models, classifier as net_class, rejector as net_rej.
#     '''
#     correct = 0
#     correct_sys = 0
#     exp = 0
#     exp_total = 0
#     total = 0
#     real_total = 0
#     alone_correct = 0
#     with torch.no_grad():
#         inputs =  data_x
#         labels = data_y
#         m = net_exp(inputs)
#         _, predicted_exp = torch.max(m.data, 1)
#         outputs = net_class(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         outputs_rej = net_rej(inputs)
#         _, predicted_rej = torch.max(outputs_rej.data, 1)
#         for i in range(len(inputs)):
#             r = (predicted_rej[i] == 1).item()
#             if r:
#                 exp += (predicted_exp[i] == labels[i]).item()
#                 correct_sys += (predicted_exp[i] == labels[i]).item()
#                 exp_total += 1
#             else:
#                 correct += (predicted[i] == labels[i]).item() 
#                 correct_sys += (predicted[i] == labels[i]).item()
#                 total += 1
#         real_total += labels.size(0)
#     cov = str(total) + str(" out of") + str(real_total)
#     to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001), "alone classifier": 100*alone_correct/real_total }
#     #print(to_print)
#     return [100*total/real_total,  100*correct_sys/real_total, 100* exp/(exp_total+0.0002),100*correct/(total+0.0001) ]

# def sample_gumbel(shape, eps=1e-20):
#     U = torch.rand(shape)
#     #U = U.to(device)
#     return -Variable(torch.log(-torch.log(U + eps) + eps))

# def gumbel_binary_sample(logits, t=0.5,eps=1e-20):
#     """ Draw a sample from the Gumbel-Softmax distribution"""
#     gumbel_noise_on = sample_gumbel(logits.size())
#     gumbel_noise_off = sample_gumbel(logits.size())
#     concrete_on = (torch.log2(logits + eps) + gumbel_noise_on) / t
#     concrete_off = (torch.log2(1 - logits + eps) + gumbel_noise_off) / t
#     concrete_softmax = torch.div(torch.exp(concrete_on), torch.exp(concrete_on) + torch.exp(concrete_off))
#     return concrete_softmax


BCE = torch.nn.BCELoss()

# def madras_loss_original(outputs, rej, labels, expert,eps = 10e-12):
#     # m: expert costs, labels: ground truth, n_classes: number of classes
#     batch_size = outputs.size()[0]
#     net_loss = -torch.log2(outputs[range(batch_size),labels]+eps)
#     exp_loss = -torch.log2(expert[range(batch_size),labels]+eps)
#     #exp_loss = #BCE(expert,labels)
#     gumbel_rej = gumbel_binary_sample(rej)
#     system_loss =  (rej[range(batch_size),0])  *  net_loss + rej[range(batch_size),1]  * exp_loss
#     return torch.sum(system_loss)/batch_size
                           
# def run_classifier_madras_original(net_class, net_rej, net_exp, data_x, data_y):
#     optimizer_class = optim.SGD(net_class.parameters(), lr=0.1)
#     optimizer_rej = optim.SGD(net_rej.parameters(), lr=0.1)

#     for epoch in range(30):  # loop over the dataset multiple times
#         running_loss = 0.0
#         inputs = data_x
#         labels = data_y
#         order = np.array(range(len(data_x)))
#         np.random.shuffle(order)
#         # in-place changing of values
#         inputs[np.array(range(len(data_x)))] = inputs[order]
#         labels[np.array(range(len(data_x)))] = labels[order]
#         x_batches = torch.split(inputs,64)
#         y_batches = torch.split(labels,64)  
#         for inputs, labels in zip(x_batches, y_batches):

#             optimizer_class.zero_grad()
#             optimizer_rej.zero_grad()
            
#             # forward + backward + optimize
#             expert_prediction = net_exp(inputs)
#             outputs = net_class(inputs)
#             outputs_no_grad = outputs.detach()
#             batch_size = outputs.size()[0]

#             rej = net_rej(inputs,outputs_no_grad)
            
#             loss_rej = madras_loss_original(outputs_no_grad, rej, labels, expert_prediction)
#             loss_rej.backward()
            
#             loss_class = -torch.log2(outputs[range(batch_size),labels]+10e-12)
#             loss_class = torch.sum(loss_class)/ batch_size
#             loss_class.backward()
            
#             optimizer_class.step()
#             optimizer_rej.step()
            
#             running_loss += loss_rej.item()
#             #print("loss " + str(loss.item()))

#     #print('Finished Training')

# class Linear_net_madras_class(nn.Module):
#     def __init__(self, input_dim, out_dim=1):
#         super(Linear_net_madras_class, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 2)
#         self.sigmoid = nn.Softmax()
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.sigmoid(out)
#         return out

# class Linear_net_madras_rej(nn.Module):
#     def __init__(self, input_dim, out_dim):
#         super(Linear_net_madras_rej, self).__init__()
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(input_dim + 2, 2)
#         self.sigmoid = nn.Softmax()
#     def forward(self, x, y_hat):
#         rej_input = torch.cat((x,y_hat), 1)
#         rej = self.fc1(rej_input)
#         rej = self.sigmoid(rej)
#         return rej
    
def test_classifier_madras_original(net_class, net_rej, net_exp, data_x, data_y):
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    with torch.no_grad():
        inputs =  data_x
        labels = data_y
        m = net_exp(inputs)
        _, predicted_exp = torch.max(m.data, 1)
        outputs = net_class(inputs)
        _, predicted = torch.max(outputs.data,1)
        rej = net_rej(inputs, outputs)
        for i in range(len(inputs)):
            r =  (rej[i][1].item()>=0.5)
            if r:
                exp += (predicted_exp[i] == labels[i]).item()
                correct_sys += (predicted_exp[i] == labels[i]).item()
                exp_total += 1
            else:
                correct += (predicted[i] == labels[i]).item() 
                correct_sys += (predicted[i] == labels[i]).item()
                total += 1
            alone_correct += (predicted[i] == labels[i]).item()  
        real_total += labels.size(0)
    cov = str(total) + str(" out of") + str(real_total)
    to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001), "alone classifier": 100*alone_correct/real_total }
    print(to_print)
    return [100*total/real_total,  100*correct_sys/real_total, 100* exp/(exp_total+0.0002),100*correct/(total+0.0001) ]

experimental_data_rej1 = []
experimental_data_rej5 = []
experimental_data_rej0 = []
# experimental_data_madras = []
# experimental_data_ora = []
# experimental_data_conf = []
trials = 1
TO_PRINT = False
for exp in tqdm(range(0,trials)):
    d = 10
    total_samples = 1000
    group_proportion = np.random.uniform()
    if group_proportion <= 0.02:
        group_proportion = 0.02
    if group_proportion >= 0.98:
        group_proportion = 0.98
    #group_proportion = 0.4
    cluster1_mean = torch.rand(d)*d
    cluster1_var = torch.rand(d)*d
    cluster1 = sample(
        cluster1_mean,
        cluster1_var,
        nb_samples= math.floor(total_samples * group_proportion * 0.5 )
    )
    cluster1_labels = torch.ones([math.floor(total_samples * group_proportion * 0.5 )], dtype=torch.long)
    cluster2_mean = torch.rand(d)*d
    cluster2_var = torch.rand(d)*d
    cluster2 = sample(
        cluster2_mean,
        cluster2_var,
        nb_samples= math.floor(total_samples * group_proportion * 0.5 )
    )
    cluster2_labels = torch.zeros([math.floor(total_samples * group_proportion * 0.5 )], dtype=torch.long)
    cluster3_mean = torch.rand(d)*d
    cluster3_var = torch.rand(d)*d
    cluster3 = sample(
        cluster3_mean,
        cluster3_var,
        nb_samples= math.floor(total_samples * (1-group_proportion) * 0.5 )
    )
    cluster3_labels = torch.ones([math.floor(total_samples * (1-group_proportion) * 0.5 )], dtype=torch.long)
    
    cluster4_mean = torch.rand(d)*d
    cluster4_var = torch.rand(d)*d
    cluster4 = sample(
        cluster4_mean,
        cluster4_var,
        nb_samples= math.floor(total_samples * (1-group_proportion) * 0.5 )
    )
    cluster4_labels = torch.zeros([math.floor(total_samples * (1-group_proportion) * 0.5 )], dtype=torch.long)
    
    # test data
    cluster1_test = sample(
        cluster1_mean,
        cluster1_var,
        nb_samples= math.floor(total_samples * group_proportion * 0.5 )
    )
    cluster1_labels_test = torch.ones([math.floor(total_samples * group_proportion * 0.5 )], dtype=torch.long)
    
    cluster2_test = sample(
        cluster2_mean,
        cluster2_var,
        nb_samples= math.floor(total_samples * group_proportion * 0.5 )
    )
    cluster2_labels_test = torch.zeros([math.floor(total_samples * group_proportion * 0.5 )], dtype=torch.long)

    cluster3_test = sample(
        cluster3_mean,
        cluster3_var,
        nb_samples= math.floor(total_samples * (1-group_proportion) * 0.5 )
    )
    cluster3_labels_test = torch.ones([math.floor(total_samples * (1-group_proportion) * 0.5 )], dtype=torch.long)
    
    cluster4_test = sample(
        cluster4_mean,
        cluster4_var,
        nb_samples= math.floor(total_samples * (1-group_proportion) * 0.5 )
    )
    cluster4_labels_test = torch.zeros([math.floor(total_samples * (1-group_proportion) * 0.5 )], dtype=torch.long)
    data_x_test = torch.cat([cluster1_test, cluster2_test, cluster3_test, cluster4_test])
    data_y_test = torch.cat([cluster1_labels_test, cluster2_labels_test, cluster3_labels_test, cluster4_labels_test])
    # expert model
    net_exp = Linear_net(d,2)
    data_x = torch.cat([cluster3, cluster4])
    data_y = torch.cat([cluster3_labels, cluster4_labels])
    run_classifier(net_exp, data_x, data_y)
    if TO_PRINT:
        print("EXPERT")
        print(test_classifier(net_exp,data_x,data_y))
    #reject
    data_x = torch.cat([cluster1, cluster2, cluster3, cluster4])
    data_y = torch.cat([cluster1_labels, cluster2_labels, cluster3_labels, cluster4_labels])
    net_rej = Linear_net_rej(d,2)
    alpha = 0
    run_classifier_rej(net_rej, net_exp, data_x, data_y, alpha)
    batch_data = test_classifier_rej(net_rej, net_exp, data_x_test, data_y_test)
    experimental_data_rej0.append(batch_data)
    net_rej = Linear_net_rej(d,2)
    alpha = 0.5
    run_classifier_rej(net_rej, net_exp, data_x, data_y, alpha)
    batch_data = test_classifier_rej(net_rej, net_exp, data_x_test, data_y_test)
    experimental_data_rej5.append(batch_data)
    net_rej = Linear_net_rej(d,2)
    alpha = 1
    run_classifier_rej(net_rej, net_exp, data_x, data_y, alpha)
    batch_data = test_classifier_rej(net_rej, net_exp, data_x_test, data_y_test)
    experimental_data_rej1.append(batch_data)
    # # confidence
    # net_class = Linear_net(d,2)
    # data_x = torch.cat([cluster1, cluster2, cluster3, cluster4])
    # data_y = torch.cat([cluster1_labels, cluster2_labels, cluster3_labels, cluster4_labels])
    # run_classifier(net_class, data_x, data_y)
    # if TO_PRINT:
    #     print("Classifier on all")
    #     print(test_classifier(net_exp,data_x,data_y))
    # batch_data = test_confidence(net_class, net_exp, data_x_test, data_y_test)
    # experimental_data_conf.append(batch_data)
    
    # # oracle
    # net_class = Linear_net(d,2)
    # data_x = torch.cat([cluster1, cluster2])
    # data_y = torch.cat([cluster1_labels, cluster2_labels])
    # run_classifier(net_class, data_x, data_y)
    # net_rej = Linear_net(d,2)
    # data_x = torch.cat([cluster1, cluster2, cluster3, cluster4])
    # data_y = torch.cat([torch.zeros([len(torch.cat([cluster1,cluster2]))], dtype=torch.long), torch.ones([len(torch.cat([cluster3,cluster4]))], dtype=torch.long)])
    # run_classifier(net_rej, data_x, data_y)
    # data_x = torch.cat([cluster1, cluster2, cluster3, cluster4])
    # data_y = torch.cat([cluster1_labels, cluster2_labels, cluster3_labels, cluster4_labels])
    # batch_data = test_oracle(net_class, net_exp, net_rej, data_x_test, data_y_test)
    # experimental_data_ora.append(batch_data)
    # # madras
    
    # data_x = torch.cat([cluster1, cluster2, cluster3, cluster4])
    # data_y = torch.cat([cluster1_labels, cluster2_labels, cluster3_labels, cluster4_labels])
    # net_class_madras = Linear_net_madras_class(d,1)
    # net_rej_madras = Linear_net_madras_rej(d,1)
    # run_classifier_madras_original(net_class_madras, net_rej_madras, net_exp, data_x, data_y)
    # batch_data = test_classifier_madras_original(net_class_madras, net_rej_madras, net_exp, data_x_test, data_y_test)
    # experimental_data_madras.append(batch_data)

metrics_class = [" coverage", "system accuracy", "expert accuracy", "classifier accuracy"]

for i in range(0,4):
    print("For " + metrics_class[i])
    print("when alpha = 0")
    print(experimental_data_rej0[0][i])
    print("when alpha = 0.5")
    print(experimental_data_rej5[0][i])
    print("when alpha = 1")
    print(experimental_data_rej1[0][i])
print("#############################")

# print("Results as differences in metrics between our method L_{CE}^0 and Confidence ")
# for i in range(0,4):
#     print("----")
#     print("For " + metrics_class[i])
#     arr = [0] * (trials-2)
#     for j in range(0,trials-2):
#         arr[j] = experimental_data_rej0[j][i] - experimental_data_conf[j][i]
#     print("average: " +str(np.average(arr)))
#     print("std: " + str(np.std(arr)))
#     print("95 confidence interval: " + str(st.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))))
# print("#############################")


# print("Results as differences in metrics between our method L_{CE}^0 and Oracle ")
# for i in range(0,4):
#     print("----")
#     print("For " + metrics_class[i])
#     arr = [0] * (trials-2)
#     for j in range(0,trials-2):
#         arr[j] = experimental_data_rej0[j][i] - experimental_data_ora[j][i]
#     print("average: " +str(np.average(arr)))
#     print("std: " + str(np.std(arr)))
#     print("95 confidence interval: " + str(st.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))))
# print("#############################")

# print("Results as differences in metrics between our method L_{CE}^0 and Madras et al. 2018 ")
# for i in range(0,4):
#     print("----")
#     print("For " + metrics_class[i])
#     arr = [0] * (trials-2)
#     for j in range(0,trials-2):
#         arr[j] = experimental_data_rej0[j][i] - experimental_data_madras[j][i]
#     print("average: " +str(np.average(arr)))
#     print("std: " + str(np.std(arr)))
#     print("95 confidence interval: " + str(st.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))))
# print("#############################")