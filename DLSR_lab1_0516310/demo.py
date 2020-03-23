import torch
import torchvision
import torchvision.transforms as transforms
from model import ImageNet, conv_deconv
from argparser import get_args
from thop import profile

def count_conv2d(m, x, y):
	x_size = x[0].shape
	y_size = y[0].shape
	kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
	bias = 1 if m.bias is not None else 0
	# N x Cout x H x W x  (Cin x Kw x Kh + bias)
	m.total_ops += torch.Tensor([int(y.nelement() * (x_size[1] * kernel_ops))])
	m.total_flops += torch.Tensor([int(x_size[0] * (2 * x_size[2] * x_size[3] * y_size[0] * (x_size[1] * kernel_ops + bias)))])

def count_relu(m, x, y):
	x = x[0]
	nelements = x.numel()
	# m.total_ops += torch.Tensor([0])
	m.total_flops += torch.Tensor([int(nelements)])

def count_linear(m, x, y):
	x_size = x[0].shape
	y_size = y[0].shape
	m.total_ops += torch.Tensor([int( x_size[1] * y_size[0])])
	m.total_flops += torch.Tensor([int(x_size[0] * (2 * x_size[1] - (0 if m.bias is not None else 1)) * y_size[0])])

def count_bn(m, x, y):
	x = x[0]
	m.total_ops += torch.Tensor([int(x.numel())])
	m.total_flops += torch.Tensor([int(2 * x.numel())])

def count_adap_avgpool(m, x, y):
	kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
	total_add = torch.prod(kernel)
	total_div = 1
	kernel_ops = total_add + total_div
	total_ops = kernel_ops * y.numel()
	m.total_ops += torch.Tensor([int(total_ops)])
	m.total_flops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
	x = x[0]
	m.total_flops += torch.Tensor([int(x.numel())])

def my_hook_function(self, input, output):
	if 'Conv' in self.__class__.__name__ :
		count_conv2d(self, input, output)
	elif 'ReLU' == self.__class__.__name__ :
		count_relu(self, input, output)
	elif 'Linear' == self.__class__.__name__ :
		count_linear(self, input, output)
	elif 'BatchNorm2d' == self.__class__.__name__ :
		count_bn(self, input, output)
	elif 'AdaptiveAvgPool2d' == self.__class__.__name__ :
		count_adap_avgpool(self, input, output)
	elif 'ool' in self.__class__.__name__ :
		count_maxpool(self, input, output)

	print('\t\t%-20s'%str(self.__class__.__name__), '%-20s'%str(input[0].shape)[11:-1], '%-20s'%str(output[0].shape)[11:-1], '%10d'%self.total_params, '%15d'%self.total_ops, '%15d'%self.total_flops)

def add_hooks(m):
	if len(list(m.children())) > 0:
		return

	m.register_buffer('total_ops', torch.zeros(1))
	m.register_buffer('total_params', torch.zeros(1))
	m.register_buffer('total_flops', torch.zeros(1))

	for p in m.parameters():
		m.total_params += torch.Tensor([p.numel()])

	m.register_forward_hook(my_hook_function)


if __name__ == '__main__' :

	args = get_args()

	print('Lab 1-1:')
	transform_test = transforms.Compose([
		transforms.Resize((256, 256)),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	testset = torchvision.datasets.ImageFolder(root = 'C:\\Users\\Frank\\Downloads\\food11\\food11re\\food11re\\evaluation\\', transform = transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size = args['batch_size'], shuffle = False, num_workers = args['thread'])

	if args['load'] == None :
		print('No weights to load.')
		exit()
	model = torch.load(args['load'])
	model.eval()

	avgcorrect = [0.0] * 11
	cases = [0.0] * 11
	with torch.no_grad() :
		for i, (inputs, labels) in enumerate(testloader):
			inputs = inputs.cuda()
			labels = labels.cuda()
			outputs = model(inputs)
			tmp, pred = outputs.max(1)
			for i in range(len(pred)) :
				if pred[i] == labels[i] :
					avgcorrect[labels[i]] += 1
				cases[labels[i]] += 1

			del outputs, inputs, labels, pred, tmp

	print('\tTest : ', '%.2f%%'%(sum(avgcorrect) / sum(cases) * 100))
    
	for i in range(11) :
		print('\t\tclass', '%2d'%i, '%5d'%cases[i], '%7.2f%%'%(avgcorrect[i] / cases[i] * 100))
	del model, avgcorrect, cases

	print('Lab 1-2:')
	model1 = torchvision.models.wide_resnet50_2().cuda()
	model1.eval()
	model2 = conv_deconv().cuda()
	model2.eval()
	inputs = torch.randn(1, 3, 256, 256).cuda()
	with torch.no_grad():
		
		print('\tModel 1 : Wide ResNet 50')
		macs, params = profile(model1, inputs=(inputs, ))
		print('\t\tTotal params:', '%10.2fM'%(params / 1000000))
		print('\t\tTotal MACs:', '%12.2fM'%(macs / 1000000))

		print('\tModel 2 : CNN and DeCNN')
		macs, params = profile(model2, inputs=(inputs, ))
		print('\t\tTotal params:', '%10.2fM'%(params / 1000000))
		print('\t\tTotal MACs:', '%12.2fM'%(macs / 1000000))

	del macs, params

	print('Lab 1-3:')
	print('\t\t%-20s'%'Name', '%-20s'%'Input Shape', '%-20s'%'Output Shape', '%10s'%'Total params', '%15s'%'Total MACs', '%15s'%'Total FLOPs')
	print('\n\tModel 1 : Wide ResNet 50')
	model1.apply(add_hooks)
	outputs = model1(inputs)
	del outputs
	total_ops = 0
	total_params = 0
	total_flops = 0
	for m in model1.modules():
		if len(list(m.children())) > 0:  # skip for non-leaf module
			continue
		total_ops += m.total_ops
		total_params += m.total_params
		total_flops += m.total_flops
	print('\t\t-----------')
	print('\t\tTotal params:', '%10.2fM'%(total_params / 1000000))
	print('\t\tTotal MACs:', '%12.2fG'%(total_ops / 1000000000))
	print('\t\tTotal FLOPs:', '%11.2fG'%(total_flops / 1000000000))


	print('\n\tModel 2 : CNN and DeCNN')
	model2.apply(add_hooks)
	outputs = model2(inputs)
	del outputs
	total_ops = 0
	total_params = 0
	total_flops = 0
	for m in model2.modules():
		if len(list(m.children())) > 0:  # skip for non-leaf module
			continue
		total_ops += m.total_ops
		total_params += m.total_params
		total_flops += m.total_flops
	print('\t\t-----------')
	print('\t\tTotal params:', '%10.2fM'%(total_params / 1000000))
	print('\t\tTotal MACs:', '%12.2fM'%(total_ops / 1000000))
	print('\t\tTotal FLOPs:', '%11.2fG'%(total_flops / 1000000000))