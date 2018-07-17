import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from misc import FurnitureDataset, preprocess
import datetime

testpath='/home/weikun/Desktop/compeition code/round2_question.csv'
load_te=np.loadtxt(testpath,dtype=str,delimiter=',')
load_te=load_te.astype('<U70')
n=len(load_te)
print("num of total testing set:"+str(n))

class_nameset=['coat_length_labels','collar_design_labels','lapel_design_labels','neck_design_labels','neckline_design_labels','pant_length_labels','skirt_length_labels','sleeve_length_labels']
num_classesset=[8,5,5,5,10,6,6,9]

path='modelnas/'

for index in range(len(class_nameset)):
  test_dataset = FurnitureDataset('test',class_nameset[index], transform=preprocess)

  test_pred = torch.load(path+'test_prediction_{0}.pth'.format(class_nameset[index]))
  test_prob = F.softmax(Variable(test_pred['px']), dim=1).data.numpy()
  test_prob = test_prob.mean(axis=2)
  result=[]
  for i in range(test_prob.shape[0]):
    str1 = ';'.join('{:.4f}'.format(im) for im in test_prob[i])

    result.append(str1)
  result=np.array(result)
  cur_class = class_nameset[index]
  load_te[load_te[:,1]==cur_class,2]=result
np.savetxt(path+'result_final_'+datetime.datetime.now().strftime('%m%d_%H%M') + '.csv', load_te, delimiter = ',',fmt='%s')