import os
import struct
#import sys
import cv2
from array import array
from os import path
import numpy as np
import matplotlib.pyplot as plt
#from progress.bar import Bar
import argparse

# source: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py

def read(dataset = "train", path = "."):
	'''
	Function to read the dataset provided from the source in byte-form : http://yann.lecun.com/exdb/mnist/ 
	Training samples :60000
	Testing samples  :10000

	Parameters :
		dataset : the part of the dataset which has been extracted 
		path    : the path of the byte-form dataset 
	Return :
		label	: the label of the images (0-9)
		image	: the image array 
		size	: the number of instances (images)
		rows	: the number of rows in an image (28 for MNIST)
		cols	: the number of column in an image (28 for MNIST)
	'''

	if dataset == "train":
		image_path = os.path.join(path, 'train-images-idx3-ubyte')
		label_path = os.path.join(path, 'train-labels-idx1-ubyte')
	elif dataset == "test":
		image_path = os.path.join(path, 't10k-images-idx3-ubyte')
		label_path = os.path.join(path, 't10k-labels-idx1-ubyte')
	else:
		raise ValueError("dataset field should \'train\' or \'test\'")

	# open the label path file 
	label_handle = open(label_path, 'rb')

	# since the byte-form for label is stored in such a way that 
	# First 32 bit integer is magic number, next 32 bit integer is number of instances, followed by labels 
	# > : high endian, I : unsigned int 
	# the magic number and size is comprises of the 8 byte. Hence read first 8 byte to fetch this information. 
	magic_number, size = struct.unpack(">II", label_handle.read(8))

	#'b' : signed character array 
	label = array("b", label_handle.read())
	label_handle.close()


	# open the image path file 
	image_handle = open(image_path, 'rb')

	# since the byte-form for image is stored in such a way that 
	# First 32 bit is magic number, next 32 bit is number of instances, 
	# next 32 bit is number of rows (=28) and next 32 bit is the number of column (=28) 
	# > : high endian, I : unsigned int 
	magic_number, size, rows, cols = struct.unpack(">IIII", image_handle.read(16))

	#'B' : unsigned character array 
	image = array("B", image_handle.read())
	image_handle.close()

	return label, image, size, rows, cols

			
def write_image(label, image, size, rows, cols, dataset='train',path="."):
	'''
	Function to write the dataset in form of image in directory structure 
	Parameters:
		label 	: the label of the image [0-9]
		image 	: the image array
		size	: number of instances 
		rows 	: the height of an image
		cols 	: the width of an image
		path	: the writing path
	'''
	image = np.asarray(image)
	image = image.reshape(size, rows, cols)
	counter=0;
	#with Bar(' Write on disk:', max=size) as bar:
	for index in range(size):
		lbl = label[index]
		img = image[index,:,:]
		filename=os.path.join(path, dataset,str(lbl), str(counter)+'.jpg')
		if not os.path.exists(os.path.join(path,dataset, str(lbl))):
			os.mkdir(os.path.join(path,dataset, str(lbl)))
		cv2.imwrite(filename,img)
		counter+=1
	#bar.next()

def main():
	parser = argparse.ArgumentParser(description='Flags required for change from command prompt')
	parser.add_argument(
        '-r', '--raw_file_dir', required=False, type=str, help='raw byte-from file', default='./raw')
	parser.add_argument(
        '-w', '--write_dir', required=False, type=str, help='write directory', default='./dataset')
	args = parser.parse_args()

	if not os.path.exists(args.raw_file_dir):
		raise ValueError("Raw file directory is not present")
	if not os.path.exists(args.write_dir):
		print('exists')
		os.mkdir(args.write_dir)
		os.mkdir(os.path.join(args.write_dir,'train'))
		os.mkdir(os.path.join(args.write_dir,'test'))

	label, image, size, rows, cols = read(dataset = "train", path = args.raw_file_dir)
	print(' Stats of the Training Images : ')
	print(' Number of samples :',size)
	print(' Number of rows and columns are', rows,'and ', cols)
	write_image(label, image, size, rows, cols, path="./dataset")

	label, image, size, rows, cols = read(dataset = "test", path = args.raw_file_dir)
	print(' Stats of the Testing Images : ')
	print(' Number of samples :',size)
	print(' Number of rows and columns are', rows,'and ', cols)
	write_image(label, image, size, rows, cols, dataset='test',path=args.write_dir)


if __name__=='__main__':
	main()

