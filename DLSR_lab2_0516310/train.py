import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ImageNet
from argparser import get_args
from dataloader import Food11Dataset#, ImgAugTransform

def forward(name, dataloader, model, lossfunction = None, optimizer = None) :

	avgcorrect = [0.0] * 11
	cases = [0.0] * 11

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
			del loss
		if optimizer :
			optimizer.step()
		
		# convert to prediction
		tmp, pred = outputs.max(1)
		del tmp, outputs
		
		# calculate accuracy
		for i in range(len(pred)) :
			if pred[i] == labels[i] :
				avgcorrect[labels[i]] += 1
			cases[labels[i]] += 1
		del labels, pred

	# print result
	avg = 0.0
	print('\t%s :'%name, '%.2f%%'%(sum(avgcorrect) / sum(cases) * 100))
	for i in range(11) :
		print('\t\tclass', '%2d'%i, '%5d'%cases[i], '%5.2f%%'%(avgcorrect[i] / cases[i] * 100))
		avg += avgcorrect[i] / cases[i]

	print('\t\tPer-class %5.2f%%'%(avg * 100 / 11))
	# return accuracy
	return avg

if __name__ == '__main__' :

	args = get_args()

	transform_train = transforms.Compose([
		transforms.Resize((256, 256)),
		# ImgAugTransform(),
	    transforms.ToTensor(),
	    transforms.Normalize([0.4965, 0.3980, 0.3058], [0.3071, 0.2927, 0.2835]),
	])

	transform_test = transforms.Compose([
		transforms.Resize((256, 256)),
	    transforms.ToTensor(),
	    transforms.Normalize([0.4965, 0.3980, 0.3058], [0.3071, 0.2927, 0.2835]),
	])

	weights = torch.Tensor([len(os.listdir('C:\\Users\\Frank\\Machine Learning\\DLSR\\dataset\\skewed_training\\' + str(i) + '\\')) for i in range(11)])
	count = torch.Tensor([])
	total = sum(weights)
	for i in range(11) :
		count = torch.cat((count, torch.Tensor([total / weights[i] for s in range(int(weights[i]))])))
	sampler = torch.utils.data.sampler.WeightedRandomSampler(count, len(count))

	trainset = Food11Dataset('C:\\Users\\Frank\\Machine Learning\\DLSR\\dataset\\skewed_training\\', transform_train)
	validationset = Food11Dataset('C:\\Users\\Frank\\Machine Learning\\DLSR\\dataset\\validation\\', transform_test)
	testset = Food11Dataset('C:\\Users\\Frank\\Machine Learning\\DLSR\\dataset\\evaluation\\', transform_test)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size = args['batch_size'], shuffle = False, num_workers = args['thread'], pin_memory = True, drop_last = True, sampler = sampler)
	validationloader = torch.utils.data.DataLoader(validationset, batch_size = 64, num_workers = args['thread'])
	testloader = torch.utils.data.DataLoader(testset, batch_size = 64, num_workers = args['thread'])

	model = None

	if args['load'] :
		model = torch.load('C:\\Users\\Frank\\Machine Learning\\DLSR\\weight\\' + args['load'])
	else :
		model = ImageNet().half().cuda()

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
		torch.save(model, 'C:\\Users\\Frank\\Machine Learning\\DLSR\\weight\\' + args['save'])

	for epoch in range(args['epoch']):  # loop over the dataset multiple times

		print('\n\tEpoch : ' + str(epoch))

		model.train()
		forward('Training', trainloader, model, criterion, optimizer)

		model.eval()
		with torch.no_grad() :
			accuracy = forward('Validation', validationloader, model)

		if args['save'] and accuracy > lastaccuracy:
			lastaccuracy = accuracy
			torch.save(model, 'C:\\Users\\Frank\\Machine Learning\\DLSR\\weight\\' + args['save'])

	print('Testing :')

	if args['save'] :
		del model
		model = torch.load('C:\\Users\\Frank\\Machine Learning\\DLSR\\weight\\' + args['save'])

	model.eval()
	with torch.no_grad() :
		forward('Test', testloader, model)