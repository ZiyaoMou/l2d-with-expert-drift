import torch
from models.densenet_ce import DenseNet121_CE
from models.densenet_defer import DenseNet121_defer
from utils.calibrate import ModelWithTemperature, ModelWithTemperature_rad

nnClassCount = 14

def model_loader(nnClassCount=14, dataLoaderValidTrain=None):
    device = torch.device("cuda")

    # 1) raw classifier (wrapped)
    model_classifier = DenseNet121_CE(nnClassCount).to(device)
    model_classifier = torch.nn.DataParallel(model_classifier)
    model_classifier.load_state_dict(torch.load(
        "checkpoints/densenet_model_ce.pth", map_location=device))

    # 2) calibrated classifier – LOAD first, then wrap
    model_class_calib = ModelWithTemperature(model_classifier)
    model_class_calib.set_temp(dataLoaderValidTrain)

    # 3) RED model (wrapped)
    model_rad = DenseNet121_CE(nnClassCount).to(device)
    model_rad = torch.nn.DataParallel(model_rad)
    model_rad.load_state_dict(torch.load(
        "checkpoints/densenet_red.pth", map_location=device), strict=False)

    # 4) calibrated RED – LOAD first, then wrap
    model_rad_calib = ModelWithTemperature_rad(model_rad)
    model_rad_calib.set_temp(dataLoaderValidTrain)

    # 5) defer model (wrapped)
    model_defer = DenseNet121_defer(nnClassCount).to(device)
    model_defer = torch.nn.DataParallel(model_defer)
    model_defer.load_state_dict(torch.load(
        "checkpoints/densenet_defer.pth", map_location=device), strict=False)

    return (model_classifier,
            model_class_calib,
            model_rad,
            model_rad_calib,
            model_defer)

        