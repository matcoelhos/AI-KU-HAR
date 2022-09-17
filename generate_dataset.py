import os
import pandas as pd
from random import random, seed, shuffle
import csv

bufsize = 200
overlap = 50

header = []
for i in range(bufsize):
	header.append('V%03d'%(i))
header.append('Class')

tc = open('train.csv','w')
vc = open('validation.csv','w')
ts = open('test.csv','w')

train_writer = csv.writer(tc, delimiter=',')
validation_writer = csv.writer(vc, delimiter=',')
test_writer = csv.writer(ts, delimiter=',')

train_writer.writerow(header)
validation_writer.writerow(header)
test_writer.writerow(header)

path = 'data/'
seed()

folderlist = os.listdir(path)
folderlist.sort()

columns = [1,2,3,5,6,7]

dummy = []
for i in range(6):
	dummy.append(0)

for folder in folderlist:
	filelist = os.listdir(path+folder)
	print(folder)
	numfiles = 1
	shuffle(filelist)
	for file in filelist:
		print('File: %03d/%03d'%(numfiles,len(filelist)),end='\r')
		numfiles+=1
		dataframe = pd.read_csv(path+folder+'/'+file,header=None)

		if numfiles < 0.9*len(filelist):
			ds = 0
		elif numfiles < 0.95*len(filelist):
			ds = 1
		else:
			ds = 2

		buf = []
		for i in range(bufsize):
			buf.append(dummy)

		line = 0
		iteration = 0
		while line < len(dataframe[0])-overlap:
			for i in range(overlap):
				elem = []
				for j in range(len(columns)):
					val = dataframe[columns[j]][line]
					# if j > 2:
						# val /= 9.806
					elem.append(val)
				buf.append(elem)	
				buf.pop(0)
				line+=1
			if True:#iteration%2==0:
				if ds == 0:
					train_writer.writerow(buf+[folderlist.index(folder)])
				elif ds == 1:
					validation_writer.writerow(buf+[folderlist.index(folder)])
				else:
					test_writer.writerow(buf+[folderlist.index(folder)])
			iteration+=1
	print()

tc.close()
vc.close()
ts.close()