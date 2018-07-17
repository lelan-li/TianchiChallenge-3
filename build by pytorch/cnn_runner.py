import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import models
import utils
from utils import RunningMean, use_gpu
from misc import FurnitureDataset, preprocess, preprocess_with_augmentation, preprocess_hflip

torch.cuda.set_device(2)
BATCH_SIZE =10
class_nameset=['coat_length_labels','collar_design_labels','lapel_design_labels','neck_design_labels','neckline_design_labels','pant_length_labels','skirt_length_labels','sleeve_length_labels']
num_classesset=[8,5,5,5,10,6,6,9]
path='modelsnas_5.20/'
chose=['DenseNetFinetune',  'InceptionV3Finetune',  'densenet121_finetune', 'densenet161', 'densenet161_finetune', 'densenet201', 'densenet201_finetune',
       'inceptionresnetv2_finetune', 'inceptionv4_finetune', 'nasnetalarge_finetune',  'resnet101', 'resnet101_finetune', 'resnet152', 'resnet152_finetune',
       'resnet18', 'resnet18_finetune', 'resnet34', 'resnet34_finetune', 'resnet50', 'resnet50_finetune', 'senet154_finetune', 'vgg16', 'vgg16_bn', 'xception_finetune']

print('path:%s'%(path))
def get_model(index):
    print ('[+] loading model... ')
    model = models.nasnetalarge_finetune(num_classesset[index])
    #model = nn.DataParallel(model)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def predict(index):
    model = get_model(index)
    model.load_state_dict(torch.load(path+'best_val_weight_{0}.pth'.format(class_nameset[index])))
    model.eval()

    tta_preprocess = [preprocess, preprocess_hflip]

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('test',classname=class_nameset[index], transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, path+'test_prediction_{0}.pth'.format(class_nameset[index]))
def train(index):
    train_dataset = FurnitureDataset('train',class_nameset[index], transform=preprocess_with_augmentation)
    val_dataset = FurnitureDataset('val',class_nameset[index], transform=preprocess)
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)

    model = get_model(index)

    criterion = nn.CrossEntropyLoss().cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print('[+] nb learnable params {0}'.format(nb_learnable_params))

    min_loss = float("inf")
    lr = 0
    patience = 0
    for epoch in range(30):
        print('epoch: %d'%(epoch))
        if epoch == 1:
            lr = 0.0003
            print('[+] set lr={%.6f}'%(lr))
        if patience == 3:
            patience = 0
            model.load_state_dict(torch.load(path+'best_val_weight_{0}.pth'.format(class_nameset[index])))
            lr = lr / 10
            if lr <=3e-6:
              lr=3e-6
            print('[+] set lr={%.6f}'%(lr))
        if epoch == 0:
            lr = 0.001
            print('[+] set lr={%.6f}'%(lr))
            
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.data[0], 1)
            running_score.update(torch.sum(preds == labels.data), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description('{running_loss.value:.5f} {running_score.value:.3f}')
        print (' epoch %d, running_loss.value:%.5f,running_score.value:%.3f'%(epoch,running_loss.value,running_score.value))

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds == lx).float())
        print(' val:log_loss:%.5f, accuracy:%.3f'%(log_loss,accuracy))

        if log_loss < min_loss:
            torch.save(model.state_dict(), path+'best_val_weight_{0}.pth'.format(class_nameset[index]))
            print('[+] val score improved from min_loss:%.5f to log_loss:%.5f. Saved!'%(min_loss,log_loss))
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('mode', choices=['train', 'predict'])
    #args = parser.parse_args()
    #print(f'[+] start `{args.mode}`')
    for index in range(8):

          #train(index+4)
          predict(index)
