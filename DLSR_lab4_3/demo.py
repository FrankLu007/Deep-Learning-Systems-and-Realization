import torch
import torchvision
import torchvision.transforms as transforms
from model import ImageNet
from argparser import get_args

def add_hooks(m):
	if len(list(m.children())) > 0:
		return

	m.register_buffer('total_params', torch.zeros(1))
	m.register_buffer('total_ZeroParams', torch.zeros(1))
	for p in m.parameters():
		m.total_params += torch.Tensor([p.numel()])
	for weight in m.named_buffers():
		if weight[0] == 'weight_mask':
			m.total_ZeroParams += (weight[1] == 0).sum()

	print('\t\t%-20s'%str(m.__class__.__name__), '%10d'%m.total_params, '%10d'%m.total_ZeroParams, '%10.2f'%(m.total_ZeroParams / float(m.total_params) if m.total_params > 0 else 0))


if __name__ == '__main__' :

	args = get_args()

	print('Lab 1-3:')
	print('\t\t%-20s'%'Name', '%10s'%'Total params', '%10s'%'Zero params', '%10s'%'Sparsity')
	model = torch.load('C:\\Users\\Frank\\Machine Learning\\DLSR\\weight\\' + args['load'])
	model.apply(add_hooks)

	total_params = 0
	total_ZeroParams = 0
	for m in model.modules():
		if len(list(m.children())) > 0:  # skip for non-leaf module
			continue
		total_params += m.total_params
		total_ZeroParams += m.total_ZeroParams
	print('\t\t-----------')
	print('\t\tTotal params:', '%10.2fM'%(total_params / 1000000))
	print('\t\tTotal Zeroparams:', '%10.2fM'%(total_ZeroParams / 1000000))
	print('\t\tTotal sparsity:', '%10.2f%%'%(total_ZeroParams / float(total_params)))