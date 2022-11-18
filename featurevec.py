import torch
from torch import optim, nn
from torchvision import models, transforms
from torch.autograd import Variable
import os
import glob
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image

def extract_feature(path,model,trans):
  files = os.listdir(path)
  save_path='./empty_features'
  for i in files:
    feature_path=os.path.join(save_path,i.split(".")[0])
    if i.split(".")[1]!='jpg':
      print("wrong!")
    img_path=os.path.join(path,i)
    img=Image.open(img_path)
    img=img.resize((224,224))
    img=trans(img)
    img.unsqueeze_(dim=0)
    img=img.cuda()
    model.eval()
    features = model(img).data.tolist()
    np.save(feature_path,features)
if __name__ == '__main__':
    path='./empty_images'
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    model = models.vgg19(pretrained=True)
    #print(model)
    x = model.features
    y = model.avgpool
    z = model.classifier
    new_model = model
    new_classification = z[:4]
    new_classification.add_module("4", nn.LeakyReLU(inplace=True))
    new_classification.add_module("5", nn.Dropout(p=0.5, inplace=False))
    new_classification.add_module("6", nn.Linear(in_features=4096, out_features=256, bias=True))
    new_model.classification = new_classification
    new_model = new_model.eval()
    # Change the device to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    new_model.to(device)
    extract_feature(path,new_model,trans)