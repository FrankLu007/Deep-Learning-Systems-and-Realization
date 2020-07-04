import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ImageNet
from dataset import MuraDataset, Categorize
from argparser import get_args
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

def Aug(img):
	img_np = img[0].numpy()
	del img
	inputs_np = exposure.equalize_adapthist(img_np, clip_limit=0.03)
	# selem = disk(30)
	# inputs_np = rank.equalize(img_np, selem=selem)

	inputs = torch.Tensor(inputs_np).reshape(1, 256, 256)
	del inputs_np

	A = torch.Tensor()
	A = torch.cat((A, inputs))
	A = torch.cat((A, inputs))
	A = torch.cat((A, inputs))

	return A


def LoadCsv(file):
	CSV = []
	with open(file, 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_file:
			CSV.append(row)
	return CSV[1:]

def forward(name, dataloader, model, lossfunction = None, optimizer = None) :

	TotalLoss = 0.0
	correct = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	cases = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

	for i, (inputs_cpu, labels_cpu) in enumerate(dataloader):
		# initialize
		torch.cuda.empty_cache()
		if optimizer :
			optimizer.zero_grad()
		# forward
		inputs = inputs_cpu.half().cuda()
		outputs = model(inputs)
		del inputs, inputs_cpu

		# loss and step
		labels = labels_cpu.cuda()
		del labels_cpu
		if lossfunction :
			loss = lossfunction(outputs, labels)
			loss.backward()
			TotalLoss += loss.item()
			del loss
		if optimizer :
			optimizer.step()
		
		# convert to prediction
		tmp, pred = outputs.max(1)
		del tmp, outputs
		
		# calculate accuracy
		for index in range(len(pred)):
			if pred[index] == labels[index]:
				correct[labels[index]] += 1
			cases[labels[index]] += 1
		del labels, pred

	# print result
	allacc = 0.0
	for index in range(14):
		if cases[index] != 0:
			allacc += correct[index] / cases[index]
	print('\t%s :'%name, '%.2f'%TotalLoss, '%5.2f%%'%(allacc * 100 / 14))
	for index in range(14):
		if cases[index] == 0:
			print('\t\tClass', index, '%5d'%cases[index], '%5.2f%%'%(0))
		else:
			print('\t\tClass', index, '%5d'%cases[index], '%5.2f%%'%(correct[index] / cases[index] * 100))

	return allacc

if __name__ == '__main__' :

	args = get_args()
	DataPath = 'C:\\Users\\Frank\\Machine Learning\\Mura\\dataset\\'
	StoragePath = 'C:\\Users\\Frank\\Machine Learning\\Mura\\weight\\'

	transform_train = transforms.Compose([
		# transforms.RandomRotation(180),
		transforms.Resize((256, 256)),
		# transforms.ColorJitter(contrast = 2),
		# transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Lambda(Aug),
	    transforms.Normalize([0.2058, 0.2058, 0.2058], [0.1943, 0.1943, 0.1943]),
	])

	transform_test = transforms.Compose([
		transforms.Resize((256, 256)),
		# transforms.ColorJitter(contrast = 1),
	    transforms.ToTensor(),
	    transforms.Lambda(Aug),
	    transforms.Normalize([0.2058, 0.2058, 0.2058], [0.1943, 0.1943, 0.1943]),
	])

	TrainCsv = LoadCsv(DataPath + 'train_image_paths.csv')
	ValidCsv = LoadCsv(DataPath + 'valid_image_paths.csv')

	TrainList = []
	# ValidList = []
	weights = torch.Tensor([0.0 for _ in range(14)])
	for path in TrainCsv:
		# if path.find(args['class']) != -1:
		weights[Categorize(path)] += 1
			# TrainList.append(path)
	print(weights)
	total = sum(weights)
	count = torch.Tensor([total / float(weights[Categorize(path)]) for path in TrainCsv])
	sampler = torch.utils.data.sampler.WeightedRandomSampler(count, len(count))

	# for path in ValidCsv:
		# if path.find(args['class']) != -1:
			# weights[Categorize(path) % 2] += 1
			# ValidList.append(path)

	trainset = MuraDataset(TrainCsv, DataPath, transform_train)
	validationset = MuraDataset(ValidCsv, DataPath, transform_test)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size = args['batch_size'], shuffle = False, num_workers = 8)
	validationloader = torch.utils.data.DataLoader(validationset, batch_size = 128, num_workers = 8)

	model = None

	if args['load'] :
		model = torch.load(StoragePath + args['load'])
	else :
		model = ImageNet(14).half().cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum = 0.9)

	print('Training :')

	lastaccuracy = 0

	model.eval()
	with torch.no_grad() :
		lastaccuracy = forward('Validation', validationloader, model)

	if args['save'] :
		torch.save(model, StoragePath + args['save'])

	for epoch in range(args['epoch']):  # loop over the dataset multiple times

		print('\n\tEpoch : ' + str(epoch))

		model.train()
		forward('Training', trainloader, model, criterion, optimizer)

		model.eval()
		with torch.no_grad() :
			accuracy = forward('Validation', validationloader, model)

		if args['save'] and accuracy > lastaccuracy:
			lastaccuracy = accuracy
			torch.save(model, StoragePath + args['save'])