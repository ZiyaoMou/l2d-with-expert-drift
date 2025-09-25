import csv
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from utils.misc import get_patient_names


class CheXpertDataSet_rad_shuffle(Dataset):
    def __init__(self, image_list_file, expert_model, patient_names_dict,
                 root_dir=None, transform=None, ignore_uncertain=True, policy="ones"):
        if root_dir is None:
            root_dir = os.path.dirname(image_list_file)

        image_paths, labels, weights = [], [], []
        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            header = next(csvReader, None)
            for line in csvReader:
                patient_name = line[0].split("/")[2]
                if patient_name not in patient_names_dict:
                    continue

                img_path = os.path.join(root_dir, line[0])
                label = line[5:]
                weight = [1] * 14
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if ignore_uncertain:
                                weight[i] = 0
                            label[i] = 1 if policy == "ones" else 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                image_paths.append(img_path)
                labels.append(label)
                weights.append(weight)

        self.image_paths = image_paths
        self.labels = labels
        self.weights = weights
        preds = expert_model.predict(self.labels)
        self.rad_1, self.rad_2, self.rad_3 = preds
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (img,
                torch.FloatTensor(self.labels[index]),
                torch.FloatTensor(self.rad_1[index]),
                torch.FloatTensor(self.rad_2[index]),
                torch.FloatTensor(self.rad_3[index]),
                torch.FloatTensor(self.weights[index]))


def split_dataset_shuffle(train_size=0.8, random_seed=66, root_dir=None, pathFileValid="",
                              pathFileTrain="", exp_fake=None, trBatchSize=16):
    dataset_all_names = get_patient_names(pathFileTrain)
    patients_train, patients_test = train_test_split(list(dataset_all_names.keys()), test_size=0.1, random_state=random_seed)
    patients_train, patients_val = train_test_split(patients_train, test_size=0.111, random_state=random_seed)
    patients_train_leftout, patients_train = train_test_split(patients_train, test_size=train_size, random_state=random_seed)

    patients_train_dict = {p: 1 for p in patients_train}
    patients_val_dict = {p: 1 for p in patients_val}
    patients_test_dict = {p: 1 for p in patients_test}

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    imgtransCrop = 224
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(imgtransCrop),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize((imgtransCrop, imgtransCrop)),
        transforms.ToTensor(),
        normalize
    ])

    dataset_train = CheXpertDataSet_rad_shuffle(pathFileTrain, exp_fake, patients_train_dict, root_dir=root_dir, transform=transform_train)
    dataset_val   = CheXpertDataSet_rad_shuffle(pathFileTrain, exp_fake, patients_val_dict,   root_dir=root_dir, transform=transform_test)
    dataset_test  = CheXpertDataSet_rad_shuffle(pathFileTrain, exp_fake, patients_test_dict,  root_dir=root_dir, transform=transform_test)

    dataLoaderTrain = DataLoader(dataset_train, batch_size=trBatchSize, shuffle=True,  num_workers=2, pin_memory=True)
    dataLoaderVal   = DataLoader(dataset_val, batch_size=trBatchSize, shuffle=False, num_workers=2, pin_memory=True)
    dataLoaderTest  = DataLoader(dataset_test, batch_size=trBatchSize, shuffle=False, num_workers=2, pin_memory=True)

    dataset_all_names_val = get_patient_names(pathFileValid)
    datasetValid = CheXpertDataSet_rad_shuffle(pathFileValid, exp_fake, dataset_all_names_val, root_dir=root_dir, transform=transform_test)
    dataLoaderOfficialValid = DataLoader(datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=2, pin_memory=True)

    print("Number of data points:")
    print(f' train: {len(dataset_train)}, test {len(dataset_test)}, val {len(dataset_val)}')

    return dataLoaderTrain, dataLoaderVal, dataLoaderTest, dataLoaderOfficialValid, patients_train_leftout