import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ImageNet
from argparser import get_args

if __name__ == '__main__' :

	args = get_args()

	transform_train = transforms.Compose([
		transforms.Resize(512),
	    transforms.RandomCrop(256),
	    transforms.RandomHorizontalFlip(),
	    transforms.ColorJitter(0.5, 0.5, 0.5),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.Resize(256),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	weights = torch.IntTensor([len(f) for r, d, f in os.walk("C:\\Users\\Frank\\Downloads\\food11\\food11re\\skewed_training\\")][1:])
	count = torch.Tensor([])
	total = sum(weights)
	for i in range(11) :
		count = torch.cat((count, torch.Tensor([total / weights[i] for s in range(weights[i])])))
	sampler = torch.utils.data.sampler.WeightedRandomSampler(count, len(count))

	trainset = torchvision.datasets.ImageFolder(root = 'C:\\Users\\Frank\\Downloads\\food11\\food11re\\skewed_training\\', transform = transform_train)
	validationset = torchvision.datasets.ImageFolder(root = 'C:\\Users\\Frank\\Downloads\\food11\\food11re\\food11re\\validation\\', transform = transform_test)
	testset = torchvision.datasets.ImageFolder(root = 'C:\\Users\\Frank\\Downloads\\food11\\food11re\\food11re\\evaluation\\', transform = transform_test)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size = args['batch_size'], shuffle = False, num_workers = args['thread'], pin_memory = True, drop_last = True, sampler = sampler)
	validationloader = torch.utils.data.DataLoader(validationset, batch_size = args['batch_size'], shuffle = False, num_workers = args['thread'])
	testloader = torch.utils.data.DataLoader(testset, batch_size = args['batch_size'], shuffle = False, num_workers = args['thread'])

	model = None

	if args['load'] :
		model = torch.load(args['load'])
	else :
		model = ImageNet()

	model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum = 0.9)

	print('Training :')

	LastLoss = 100

	if args['save'] :
		torch.save(model, args['save'])

	for epoch in range(args['epoch']):  # loop over the dataset multiple times
	    
		avgloss = 0.0
		avgcorrect = [0.0] * 11
		cases = [0.0] * 11

		print('\n\tEpoch : ' + str(epoch))
	    
		avgloss = 0.0
		avgcorrect = [0.0] * 11
		cases = [0.0] * 11

		model.train()
		for i, (inputs, labels) in enumerate(trainloader):
			inputs = inputs.cuda() 
			labels = labels.cuda()
			optimizer.zero_grad()
			outputs = model(inputs)
			tmp, pred = outputs.max(1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			avgloss += loss.cpu().item()
			for i in range(len(pred)) :
				if pred[i] == labels[i] :
					avgcorrect[labels[i]] += 1
				cases[labels[i]] += 1

			del outputs, inputs, labels, loss, pred, tmp

		print('\tTraining : ' + '%.5lf'%(avgloss / sum(cases)) + ' / ' + '%.2f%%'%(sum(avgcorrect) / sum(cases) * 100))

		for i in range(11) :
			if cases[i] == 0 :
				print('\t\tclass', '%2d'%i, '%5d'%cases[i], 'NULL')
			else :
				print('\t\tclass', '%2d'%i, '%5d'%cases[i], '%5.2f%%'%(avgcorrect[i] / cases[i] * 100))

		model.eval()
		with torch.no_grad() :
			for i, (inputs, labels) in enumerate(validationloader):
				inputs = inputs.cuda()
				labels = labels.cuda()
				outputs = model(inputs)
				tmp, pred = outputs.max(1)
				loss = criterion(outputs, labels)
				avgloss += loss.cpu().item()
				for i in range(len(pred)) :
					if pred[i] == labels[i] :
						avgcorrect[labels[i]] += 1
					cases[labels[i]] += 1

				del outputs, inputs, labels, loss, pred, tmp

		print('\tValidation : ' + '%.5lf'%(avgloss / sum(cases)) + ' / ' + '%.2f%%'%(sum(avgcorrect) / sum(cases) * 100))
		with open(args['log'], 'a') as f :
			f.write('Epoch ' + str(epoch) + '  '  + '%.5lf'%(avgloss / sum(cases)) + ' / ' + '%.2f%%'%(sum(avgcorrect) / sum(cases) * 100) + '\n')

		for i in range(11) :
			print('\t\tclass', '%2d'%i, '%5.2f%%'%(avgcorrect[i] / cases[i] * 100))

		if args['save'] and (avgloss / sum(cases)) < LastLoss:
			LastLoss = (avgloss / sum(cases))
			torch.save(model, args['save'])
		elif args['save']:
			model = torch.load(args['save'])

	print('Testing :')

	avgloss = 0.0
	avgcorrect = [0.0] * 11
	cases = [0.0] * 11

	model.eval()
	with torch.no_grad() :
		for i, (inputs, labels) in enumerate(validationloader):
			inputs = inputs.cuda()
			labels = labels.cuda()
			outputs = model(inputs)
			tmp, pred = outputs.max(1)
			loss = criterion(outputs, labels)
			avgloss += loss.cpu().item()
			for i in range(len(pred)) :
				if pred[i] == labels[i] :
					avgcorrect[labels[i]] += 1
				cases[labels[i]] += 1

			del outputs, inputs, labels, loss, pred, tmp

	print('\tTest : ' + '%.5lf'%(avgloss / sum(cases)) + ' / ' + '%.2f%%'%(sum(avgcorrect) / sum(cases) * 100))
    
	for i in range(11) :
		print('\t\tclass', '%2d'%i, '%5.2f%%'%(avgcorrect[i] / cases[i] * 100))