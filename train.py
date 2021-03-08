from datagenerator import data_generator
from model import Net
from torch.autograd import Variable
import torch
from torch import optim 
import os

path='dataset'
phase='train'
batch_size=100
image_side=28
weight_path='weight'

if not os.path.exists(weight_path):
	os.mkdir(weight_path)

def load_model():
	model = Net()
	print(model)
	return model

def loss_fucntion(predicted,gt):
	predicted = torch.clamp(predicted,max=0.99, min=1e-7)
	loss = -gt*torch.log(predicted)-(1-gt)*torch.log(1-predicted)
	loss = torch.mean(loss)
	return loss

model = load_model()

dg_train = data_generator(path,'train',batch_size,image_side)
dg_val = data_generator(path,'test',batch_size,image_side)

epochs = 5;

iteration_per_epoch_train=len(dg_train.dataset)//batch_size
iteration_per_epoch_val=len(dg_val.dataset)//batch_size

print(iteration_per_epoch_train)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(epochs):
	for iteration in range(iteration_per_epoch_train):
		image,label = dg_train.load_batch()
		label =label.type(torch.FloatTensor)
		optimizer.zero_grad();
		output = model(Variable(image))
		loss = loss_fucntion(output,label)
		loss.backward()
		optimizer.step()
		#print(output.shape,loss.item())
		print('Epoch :',epoch+1,' Iteration : ',iteration+1, ' loss : ', loss.item())
	torch.save(model.state_dict(), os.path.join(weight_path,str(epoch)+'.pth'))
