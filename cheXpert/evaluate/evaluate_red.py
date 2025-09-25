#outGT1, outPRED1 = CheXpertTrainer.test(model_sig, dataLoaderTest, nnClassCount, "m-epoch5-13052020-174143_min.pth.tar", class_names)
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics

class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Pneumoperitoneum', 'Support Devices']

nnClassCount = 14

def test_model(model, dataloader):
    model.eval()
    model_out = [[] for temp_con in range(14)] # model predictions
    target_all = [[] for temp_con in range(14)] # target as cpu array
    rad_1 = [[] for temp_con in range(14)] # radiologist 1 predictions
    rad_2 = [[] for temp_con in range(14)]
    rad_3 = [[] for temp_con in range(14)]
    # get predictions
    for i, (input, target, rad1, rad2, rad3, weights) in enumerate(dataloader):
        target = target.cuda()
        
        bs, c, h, w = input.size()
        varInput = input.view(-1, c, h, w)

        out = model(varInput)
        for batch in range(0,len(rad1)):
            j = 0
            for cls in range(14):
                is_uncertain = weights[batch][cls].cpu().numpy().item()
                if is_uncertain == 1:
                    out_softmax_class = torch.nn.functional.softmax(out[batch][j:j+2])
                    model_out[cls].append(out_softmax_class[1].detach().cpu().numpy().item())
                    rad_1[cls].append(rad1[batch][cls].cpu().numpy().item())
                    rad_2[cls].append(rad2[batch][cls].cpu().numpy().item())
                    rad_3[cls].append(rad3[batch][cls].cpu().numpy().item())
                    target_all[cls].append(target[batch][cls].cpu().numpy().item())
                j += 2
                    
    j = 0
    for i in range(nnClassCount):
        if i != 2 and i!= 5 and i != 6 and i != 8 and i!= 10:
            continue
        
        fpr, tpr, threshold = metrics.roc_curve(target_all[i], model_out[i])
        roc_auc = metrics.auc(fpr, tpr)
        f = plt.subplot(1, 5, j+1)
        j +=1

        plt.title('ROC for: ' + class_names[i])
        plt.plot(fpr, tpr, label = 'classifier : AUC = %0.2f' % roc_auc)

        fpr, tpr, threshold = metrics.roc_curve(target_all[i], rad_1[i])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr[1], tpr[1],marker='o', markersize=3,  label = 'rad1: AUC = %0.2f' % roc_auc)
        fpr, tpr, threshold = metrics.roc_curve(target_all[i], rad_2[i])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr[1], tpr[1],marker='o', markersize=3,  label = 'rad2: AUC = %0.2f' % roc_auc)
        fpr, tpr, threshold = metrics.roc_curve(target_all[i], rad_3[i])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr[1], tpr[1],marker='o', markersize=3,  label = 'rad3: AUC = %0.2f' % roc_auc)


        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 20
    fig_size[1] = 2
    plt.rcParams["figure.figsize"] = fig_size

    plt.savefig("roc_modelrad_val.png", dpi=1000)
    plt.show()