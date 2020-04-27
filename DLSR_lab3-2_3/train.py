import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ReZero
from argparser import get_args

def forward(name, dataloader, model, lossfunction = None, optimizer = None) :

	avgloss = 0.0
	avgcorrect = 0.0
	cases = 0.0
	iters = 0

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
			avgloss += loss.item()
			del loss
		if optimizer :
			optimizer.step()
		
		# convert to prediction
		tmp, pred = outputs.max(1)
		del tmp, outputs
		
		# calculate accuracy
		for i in range(len(pred)) :
			if pred[i] == labels[i] :
				avgcorrect += 1
		cases += len(pred)
		del labels, pred
		iters += 1

	# print result
	avg = 0.0
	print('\t%s :'%name, '%.2f%%'%(avgcorrect / cases * 100), '%.3f'%(avgloss / iters))
	# for i in range(11) :
	# 	print('\t\tclass', '%2d'%i, '%5d'%cases[i], '%5.2f%%'%(avgcorrect[i] / cases[i] * 100))
	# 	avg += avgcorrect[i] / cases[i]

	# print('\t\tPer-class %5.2f%%'%(avg * 100 / 11))
	# return accuracy
	return avg

if __name__ == '__main__' :

	args = get_args()
	DataPath = 'C:\\Users\\Frank\\Machine Learning\\DLSR\\dataset\\'
	StoragePath = 'C:\\Users\\Frank\\Machine Learning\\DLSR\\weight\\'

	transform_train = transforms.Compose([
		transforms.Resize((32 * args['r'], 32 * args['r'])),
		transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize([0.4965, 0.3980, 0.3058], [0.3071, 0.2927, 0.2835]),
	])

	transform_test = transforms.Compose([
		transforms.Resize((32 * args['r'], 32 * args['r'])),
	    transforms.ToTensor(),
	    transforms.Normalize([0.4965, 0.3980, 0.3058], [0.3071, 0.2927, 0.2835]),
	])

	weights = torch.Tensor([len(os.listdir(DataPath + 'skewed_training\\' + str(i) + '\\')) for i in range(11)])
	count = torch.Tensor([])
	total = sum(weights)
	for i in range(11) :
		count = torch.cat((count, torch.Tensor([total / weights[i] for s in range(int(weights[i]))])))
	sampler = torch.utils.data.sampler.WeightedRandomSampler(count, len(count))

	trainset = torchvision.datasets.ImageFolder(DataPath + 'skewed_training\\', transform_train)
	validationset = torchvision.datasets.ImageFolder(DataPath + 'validation\\', transform_test)
	testset = torchvision.datasets.ImageFolder(DataPath + 'evaluation\\', transform_test)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size = args['batch_size'], shuffle = False, num_workers = 12, pin_memory = True, drop_last = True, sampler = sampler)
	validationloader = torch.utils.data.DataLoader(validationset, batch_size = 128, num_workers = 12)
	testloader = torch.utils.data.DataLoader(testset, batch_size = 128, num_workers = 12)

	model = None

	if args['load'] :
		model = torch.load(StoragePath + args['load'])
	else :
		model = ReZero(args['d'], args['w'], args['r']).half().cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum = 0.9)

	print('Training :')

	avgloss = 0.0
	avgcorrect = [0.0] * 11
	cases = [0.0] * 11
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

	print('Testing :')

	if args['save'] :
		del model
		model = torch.load(StoragePath + args['save'])

	model.eval()
	with torch.no_grad() :
		forward('Test', testloader, model)