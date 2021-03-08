from datagenerator import data_generator
from model import Net
from torch.autograd import Variable
import torch
from torch import optim 
import os
import numpy as np
from tqdm import tqdm

path='dataset'
phase='test'
batch_size=1
image_side=28
weight_path='weight/4.pth'
n_classes = 10

def load_model():
	model = Net()
	print(model)
	return model

model = load_model()
model.load_state_dict(torch.load(weight_path))
model.eval()

dg_test = data_generator(path,'test',batch_size,image_side)

iteration_per_epoch_test=len(dg_test.dataset)//batch_size

print(iteration_per_epoch_test)

confusion_matrix = np.zeros((n_classes, n_classes),dtype='uint8')

for iteration in tqdm(range(iteration_per_epoch_test)):
	image,label = dg_test.load_batch()
	label =label.type(torch.FloatTensor)
	with torch.no_grad():
		output = model(Variable(image))
		predicted = torch.argmax(output)
		gt = torch.argmax(label)
		#print(predicted, torch.argmax(label))			
		confusion_matrix[predicted,gt]+=1
print(confusion_matrix)
print('accuracy', np.trace(confusion_matrix)/np.sum(confusion_matrix))
		





