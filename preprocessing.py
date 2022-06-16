#!/usr/bin/python
# -*- coding:utf-8 -*-
from logging import root
import os
import numpy as np
import ipdb

root = '../data'  # data path
root = root.replace('\\','/')

filename = [] # 所有数据的名称和label

def generate(dir,label):
	files = os.listdir(dir)
	files.sort()
	print ('****************')
	print ('input :',dir)
	print ('start...')
	# listText = open('train.txt','a')
	for file in files:
		dir_path = root + '/' + file
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		filename.append(dir + '/' + file + ' ' +str(int(label)))
		# ipdb.set_trace()
		# listText.write(name)

	# 打乱文件列表
	np.random.shuffle(filename)
	train = filename[:int(len(filename)*0.8)]
	test = filename[int(len(filename)*0.8):]

	with open('train.txt','w') as f1, open('test.txt','w') as f2:
		for i in train:
			f1.write(i+'\n')
		for j in test:
			f2.write(j + '\n')
	# listText.close()
	print ('down!')
	print ('****************')
 
 

 
 
if __name__ == '__main__':
	i = 0
	folderlist = os.listdir(root)          #列举文件夹
	for folder in folderlist:
		generate(os.path.join(root, folder),i)
		i += 1

