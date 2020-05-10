import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ImageNet
from argparser import get_args
from torch.nn.utils import prune

def forward(name, dataloader, model, lossfunction = None, optimizer = None) :

	avgloss = 0.0
	avgcorrect = 0.0
	cases = 0.0
	iters = 0

	Ccases = [0.0] * 11
	Cacc = [0.0] * 11

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
		for index in range(len(pred)):
			if pred[index] == labels[index]:
				Cacc[labels[index]] += 1
			Ccases[labels[index]] += 1
		avgcorrect += (pred == labels).sum().cpu()
		cases += len(pred)
		del labels, pred
		iters += 1

	# print result
	avg = float(avgcorrect) / cases
	print('\t%s :'%name, '%.2f%%'%(avg * 100), '%.3f'%(avgloss / iters))
	for index in range(11):
		print('\t\tClass', index, '%5.2f%%'%(Cacc[index] / Ccases[index] * 100))

	return avg

if __name__ == '__main__' :

	args = get_args()
	DataPath = 'C:\\Users\\Frank\\Machine Learning\\DLSR\\dataset\\'
	StoragePath = 'C:\\Users\\Frank\\Machine Learning\\DLSR\\weight\\'

	transform_train = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize([0.4965, 0.3980, 0.3058], [0.3071, 0.2927, 0.2835]),
	])

	transform_test = transforms.Compose([
		transforms.Resize((256, 256)),
	    transforms.ToTensor(),
	    transforms.Normalize([0.4965, 0.3980, 0.3058], [0.3071, 0.2927, 0.2835]),
	])

	weights = torch.IntTensor([len(f) for r, d, f in os.walk(DataPath + 'skewed_training\\')][1:])
	count = torch.Tensor([])
	total = sum(weights)
	for i in range(11) :
		count = torch.cat((count, torch.Tensor([total / float(weights[i]) for s in range(int(weights[i]))])))
	sampler = torch.utils.data.sampler.WeightedRandomSampler(count, len(count))

	trainset = torchvision.datasets.ImageFolder(DataPath + 'skewed_training\\', transform_train)
	validationset = torchvision.datasets.ImageFolder(DataPath + 'validation\\', transform_test)
	testset = torchvision.datasets.ImageFolder(DataPath + 'evaluation\\', transform_test)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size = args['batch_size'], shuffle = False, num_workers = 12, pin_memory = True, drop_last = True, sampler = sampler)
	validationloader = torch.utils.data.DataLoader(validationset, batch_size = 128, num_workers = 12)
	testloader = torch.utils.data.DataLoader(testset, batch_size = 64, num_workers = 12)

	model = None

	if args['load'] :
		model = torch.load(StoragePath + args['load'])
	else :
		model = ImageNet().half().cuda()
		PruneList = ['Conv2d', 'Linear']
		PrunePair = ()
		for m in model.modules():
			if len(list(m.children())) == 0 and m.__class__.__name__ in PruneList:  # skip for non-leaf module
				PrunePair += ((m, 'weight'), )
				prune.RandomStructured.apply(module = m, name = 'weight', amount = 0.5)
		# prune.global_unstructured(PrunePair, pruning_method = prune.RandomStructured, amount = 0.6)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum = 0.9)

	print('Training :')

	lastaccuracy = 0

	# model.eval()
	# with torch.no_grad() :
		# lastaccuracy = forward('Validation', validationloader, model)

	if args['save'] :
		torch.save(model, StoragePath + args['save'])

	for epoch in range(args['epoch']):  # loop over the dataset multiple times

		print('\n\tEpoch : ' + str(epoch))

		model.train()
		forward('Training', trainloader, model, criterion, optimizer)

		# model.eval()
		# with torch.no_grad() :
			# accuracy = forward('Validation', validationloader, model)

		# if args['save'] and accuracy > lastaccuracy:
			# lastaccuracy = accuracy
		torch.save(model, StoragePath + args['save'])

	print('Testing :')

	if args['save'] :
		del model
		model = torch.load(StoragePath + args['save'])

	model.eval()
	with torch.no_grad() :
		forward('Test', testloader, model)