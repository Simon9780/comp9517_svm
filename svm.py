import os
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import argparse

def train(path, empty_path):
    x_set = []
    for file_name in os.listdir(path):
        x_set.append(np.load(path+'/'+file_name)[0])

    y_set = [1 for i in x_set]

    counter = 0
    for file_name in os.listdir(empty_path):
        if counter < 10000:
            x_set.append(np.load(empty_path+'/'+file_name)[0])
            y_set.append(-1)
        counter += 1

    print("data loaded")
    print(len(x_set))

    train_data,test_data,train_label,test_label = train_test_split(x_set,y_set,random_state=0,train_size=0.8,test_size=0.2)
    print("data splited")
    print("Train set size:", len(train_data))
    print("Test set size:", len(test_data))

    classifier=svm.SVC(C=1,kernel='rbf',gamma=1,decision_function_shape='ovr')
    classifier.fit(train_data,train_label)
    print("model fitted")

    print("Test set score:",classifier.score(test_data,test_label))

    s=pickle.dumps(classifier)
    f=open('svm.model', "wb+")
    f.write(s)
    f.close()
    print ("Done\n")

    return classifier

def test(model, vgg_model, trans, test_set_path, test_label_path):
    results = []
    for file_name in os.listdir(test_set_path):
        label_file = test_label_path + file_name.replace(".jpg", ".txt")

        image = cv2.imread(test_set_path + file_name)

        splits = []
        for i in range(int(720/8)-1):
            for j in range(int(1280/8)-1):
                split_img = []
                for m in range(8*i, 8*i+16):
                    pixel = []
                    for n in range(8*j, 8*j+16):
                        pixel.append(image[m][n])
                    split_img.append(pixel)
                array = np.array(split_img)
                splits.append(array)
        
        features = []
        for img in splits:
            features.append(extract_feature(img, vgg_model, trans)[0])
        
        pred_y = np.array(model.predict(features)).reshape((159, 89))
        
        bounding_box_list = []
        for i in range(len(pred_y)):
            for j in range(len(pred_y[0])):
                if pred_y[i][j] == 1:
                    x1 = i - 2 if (i - 2) > 0 else 0
                    y1 = j
                    x2 = (i + 2) if (i + 2) < len(pred_y) else len(pred_y) - 1
                    y2 = (j + 4) if (j + 4) < len(pred_y[0]) else len(pred_y[0]) - 1

                    ls = False
                    bs = False
                    rs = False
                    delete = False
                    
                    while not ls or not bs or not rs or delete:
                        delete = False
                        sum_t = 0
                        sum_b = 0
                        for m in range(x2 - x1):
                            sum_t += pred_y[m+x1][y1] if pred_y[m+x1][y1] > 0 else 0
                            sum_b += pred_y[m+x1][y2] if pred_y[m+x1][y2] > 0 else 0
                        if sum_t / (x2 - x1) < 0.3:
                            for m in range(x2 - x1):
                                pred_y[m+x1][y1] = 0
                            y1 += 1
                            delete = True
                        if sum_b / (x2 - x1) < 0.3:
                            for m in range(x2 - x1):
                                pred_y[m+x1][y2] = 0
                            y2 -= 1
                            bs = True
                            delete = True
                        else: y2 += 1 if not bs and y2 + 1 < len(pred_y[0]) else 0
                        
                        sum_l = 0
                        sum_r = 0
                        for m in range(y2 - y1):
                            sum_l += pred_y[x1][m+y1] if pred_y[x1][m+y1] > 0 else 0
                            sum_r += pred_y[x2][m+y1] if pred_y[x2][m+y1] > 0 else 0
                        if sum_l / (y2 - y1) < 0.3:
                            for m in range(y2 - y1):
                                pred_y[x1][m+y1] = 0
                            x1 += 1
                            ls = True
                            delete = True
                        else: x1 -= 1 if not ls and x1 - 1 >= 0 else 0
                        if sum_b / (y2 - y1) < 0.3:
                            for m in range(y2 - y1):
                                pred_y[x2][m+y1] = 0
                            y2 -= 1
                            rs = True
                            delete = True
                        else: x2 += 1 if not rs and x2 + 1 < len(pred_y) else 0

                        if y2 == len(pred_y[0]) - 1:
                            bs = True
                        if x1 == 0:
                            ls = True
                        if x2 == len(pred_y) - 1:
                            rs = True

                        
                    
                    for m in range(x2 - x1 + 1):
                        for n in range(y2 - y1 + 1):
                            pred_y[m+x1][n+y1] = 0
                    
                    x = x1/1280
                    y = y1/720
                    w = ((x2 - x1) * 8 + 16)/1280
                    h = ((y2 - y1) * 8 + 16)/720

                    bounding_box_list.append([x,y,w,h])

        target_list = []
        f = open(label_file, "r")
        lines = f.readlines()
        for line in lines:
            l = line.split(" ")
            target_list.append([float(l[1]), float(l[2]), float(l[3]), float(l[4])])

        
        for t in target_list:
            r_list = []
            for b in bounding_box_list:
                t_x1 = t[0]
                t_y1 = t[1]
                t_x2 = t[2] + t_x1
                t_y2 = t[3] + t_y1

                b_x1 = b[0]
                b_y1 = b[1]
                b_x2 = b[2] + b_x1
                b_y2 = b[3] + b_y1

                i_x1 = t_x1 if t_x1 > b_x1 else b_x1
                i_y1 = t_y1 if t_y1 > b_y1 else b_y1
                i_x2 = t_x2 if t_x2 < b_x2 else b_x2
                i_y2 = t_y2 if t_y2 < b_y2 else b_y2
                
                area_t = t[2] * t[3]
                area_b = b[2] * b[3]
                area_i = (i_x2 - i_x1) * (i_y2 - i_y1)
                area_u = area_t + area_b - area_i

                p = area_i / area_b
                r = area_i / area_t
                iou = area_i / area_u

                r_list.append([iou, p, r])
            
            picked = []
            for r in r_list:
                if picked == [] or r[0] > picked[0]:
                    picked = r
            #print(picked)
            results.append(picked)
        
        cv_image = cv2.imread(test_set_path + file_name,1)
        for b in bounding_box_list:
            cv2.rectangle(cv_image,(int(b[0]*1280),int(b[1]*720)),(int(b[0]+b[2]*1280),int(b[0]+b[3]*720)),(0,0,255),5)

        cv2.imwrite("./results/"+file_name,cv_image)
        return
    #print(results)

    iou = 0
    f2 = 0
    for r in results:
        f2 += 5*r[1]*r[2]/(4*(r[1]+r[2]))
        iou += r[0]

    iou_average = iou/len(results)
    f2_average = f2/len(results)

    print("Average IoU:", iou_average)
    print("Average F2:", f2_average)

def make_vgg_model():
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

    return new_model, trans

def extract_feature(image_array,model,trans):
    img = Image.fromarray(image_array)
    img=img.resize((224,224))
    img=trans(img)
    img.unsqueeze_(dim=0)
    img=img.cuda()
    model.eval()
    features = model(img).data.tolist()

    return features

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default="./test_images")
    parser.add_argument('--label_path', type=str, default='./test_labels')
    parser.add_argument('--train', type=bool, default=False)

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    feature_path = './features'
    no_feature_path = './empty_features'
    train_again = opt.train

    test_set_path = opt.label_path
    test_label_path = opt.images_path

    if os.path.exists("svm.model") and not train_again:
        f=open('svm.model','rb')
        s=f.read()
        model=pickle.loads(s)
    else: 
        print("Training")
        model = train(feature_path, no_feature_path)

    vgg_model, trans = make_vgg_model()
    print("Testing")
    test(model, vgg_model, trans, test_set_path, test_label_path)
