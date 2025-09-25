import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Pneumoperitoneum', 'Support Devices']

def test_defer_metrics(model, data_loader):
    model.eval()
    # now only for rad_1
    model_out = [[] for temp_con in range(14)] # model predictions
    model_class = [[] for temp_con in range(14)] # model predictions
    target_all = [[] for temp_con in range(14)] # target as cpu array
    rad_1 = [[] for temp_con in range(14)] # radiologist 1 predictions
    rad_2 = [[] for temp_con in range(14)]
    rad_3 = [[] for temp_con in range(14)]
    coverages = [0]*14
    # get predictions
    for i, (input, target, rad1, rad2, rad3, weights) in enumerate(data_loader):
        
        target = target.cuda()
        bs, c, h, w = input.size()
        varInput = input.view(-1, c, h, w)

        out = model(varInput)
        for batch in range(0,len(rad1)):
            j = 0
            for cls in range(14):
                is_uncertain = weights[batch][cls].cpu().numpy().item()
                if is_uncertain == 1:
                    out_softmax = torch.nn.functional.softmax(out[batch][j:j+3])
                    out_softmax_class = torch.nn.functional.softmax(out[batch][j:j+2])
                    _, predicted = torch.max(out_softmax.data, 0)
                    model_class[cls].append(out_softmax_class[1].detach().cpu().numpy().item())
                    if predicted == 2:
                        model_out[cls].append(rad1[batch][cls].cpu().numpy().item())
                    else:
                        model_out[cls].append(out_softmax_class[1].detach().cpu().numpy().item())
                        coverages[cls] += 1
                    rad_1[cls].append(rad1[batch][cls].cpu().numpy().item())
                    rad_2[cls].append(rad2[batch][cls].cpu().numpy().item())
                    rad_3[cls].append(rad3[batch][cls].cpu().numpy().item())
                    target_all[cls].append(target[batch][cls].cpu().numpy().item())
                j += 3

    for i in range(14):
        print(f'################## \n Class {class_names[i]}')
        print(f'Coverage {coverages[i]/len(rad1):.3f}, Defer {roc_auc_score(target_all[i], model_out[i]):.3f}')
        print(f'Radiologist {roc_auc_score(target_all[i], rad_1[i]):.3f},  Model alone {roc_auc_score(target_all[i], model_class[i]):.3f}')


def test_defer_metrics_class(model, data_loader):
    model.eval()
    # now only for rad_1
    model_out = [[] for temp_con in range(14)] # model predictions
    target_all = [[] for temp_con in range(14)] # target as cpu array
    rad_1 = [[] for temp_con in range(14)] # radiologist 1 predictions
    rad_2 = [[] for temp_con in range(14)]
    rad_3 = [[] for temp_con in range(14)]
    coverages = [0]*14
    # get predictions
    for i, (input, target, rad1, rad2, rad3, weight) in enumerate(data_loader):
        
        target = target.cuda()
        bs, c, h, w = input.size()
        varInput = input.view(-1, c, h, w)

        out = model(varInput)
        for batch in range(0,len(rad1)):
            j = 0
            for cls in range(14):
                out_softmax = torch.nn.functional.softmax(out[batch][j:j+3])
                out_softmax_class = torch.nn.functional.softmax(out[batch][j:j+2])
                _, predicted = torch.max(out_softmax.data, 0)  
                #if False: #predicted == 2:
                #    model_out[cls].append(rad1[batch][cls].cpu().numpy().item())
                #else:
                model_out[cls].append(out_softmax_class[1].detach().cpu().numpy().item())#
                coverages[cls] += 1#
                rad_1[cls].append(rad1[batch][cls].cpu().numpy().item())
                rad_2[cls].append(rad2[batch][cls].cpu().numpy().item())
                rad_3[cls].append(rad3[batch][cls].cpu().numpy().item())
                target_all[cls].append(target[batch][cls].cpu().numpy().item())
                #acc[cls] += (target[batch][cls] == rad1[batch][cls])
                j += 3

    for i in range(14):
        print(coverages[i]/len(rad_1[0]))
        print(roc_auc_score(target_all[i], model_out[i]))
        print(roc_auc_score(target_all[i], rad_1[i]))
        #print(roc_auc_score(target_all[i], rad_2[i]))
        #print(roc_auc_score(target_all[i], rad_3[i]))

