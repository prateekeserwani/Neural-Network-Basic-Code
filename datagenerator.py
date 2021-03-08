import cv2
import os
import random
import numpy as np
import torch

class data_generator():

	def __init__(self,path,phase,batch_size,image_side, n_classes=10):
		self.dataset=[]
		self.phase=phase
		self.n_classes = n_classes 	
		self.load_dataset(path)
		self.pointer=0;		
		self.batch_size=batch_size
		self.image_side=image_side


	def load_dataset(self,path):
		temp=[]	
		for root, dirs, files in os.walk(path):
			for file in files:
				if file.endswith(".jpg"):
					path_split = root.split(os.sep);
					if self.phase==path_split[-2]:
						lab = np.zeros((self.n_classes), dtype='uint8')
						lab[int(path_split[-1])]=1
						self.dataset.append((os.path.join(root, file),lab))
			
		print('Number of images')		
		print(len(self.dataset))
		
		print('Data shuffling')
		for i in range(10):
			random.shuffle(self.dataset)


	def transform(self,image):
		mean=0.5
		standard_deviation =0.5
		image = torch.from_numpy(image);
		image = image.type(torch.FloatTensor)
		image = image/255.0
		image = (image - mean)/standard_deviation
		return image

	def load_instance(self):
		image_name = self.dataset[self.pointer][0]
		image_label = self.dataset[self.pointer][1]
		data = cv2.imread(image_name,0).reshape(-1)
		#print(data.shape)
		data = self.transform(data)
		return data,image_label
	
	def load_batch(self):
		input_batch = torch.zeros(self.batch_size,self.image_side*self.image_side).type(torch.FloatTensor)
		batch_label =[]
		batch_counter=0;		
		while(1):
			if batch_counter>=self.batch_size:
				break;
			image, label = self.load_instance();
			#print(image.shape)
			input_batch[batch_counter,:]=image
			if self.pointer+1>=len(self.dataset):
				self.pointer=0;
			else:
				self.pointer+=1
			batch_label.append(label)
			batch_counter+=1;
		batch_label = np.asarray(batch_label)
		batch_label = torch.from_numpy(batch_label)

		return input_batch, batch_label

'''
# unit testing 
batch_size=5
dg_train = data_generator(path='dataset',phase='train',batch_size=batch_size,image_side=28)
iteration_per_epoch_train=len(dg_train.dataset)//batch_size
		
for iteration in range(iteration_per_epoch_train):
	image,label = dg_train.load_batch()
	print(image.shape, label)

'''

