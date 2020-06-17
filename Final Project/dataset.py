import torch
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as transforms
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
import matplotlib.pyplot as plt

def Aug(img):
	img_np = img[0].numpy()
	del img
	# inputs_np = exposure.equalize_hist(img_np, clip_limit=0.03)
	selem = disk(1)
	inputs_np = rank.equalize(img_np, selem=selem)

	inputs = torch.Tensor(inputs_np).reshape(1, 256, 256)
	del inputs_np

	A = torch.Tensor()
	A = torch.cat((A, inputs))
	A = torch.cat((A, inputs))
	A = torch.cat((A, inputs))

	return A

def Categorize(path):
	if path.find('ELBOW') != -1:
		label = 0
	elif path.find('FINGER') != -1:
		label = 1
	elif path.find('FOREARM') != -1:
		label = 2
	elif path.find('HAND') != -1:
		label = 3
	elif path.find('HUMERUS') != -1:
		label = 4
	elif path.find('SHOULDER') != -1:
		label = 5
	else:
		label = 6
	label *= 2
	if path.find('positive') != -1:
		label += 1
	return label

class MuraDataset(torch.utils.data.Dataset):
	def __init__(self, CsvFile, DataPath, transform):
		super(MuraDataset, self).__init__()
		self.list = CsvFile
		self.DataPath = DataPath
		self.transform = transform

	def __getitem__(self, index):
		PILImage = pil_loader(self.DataPath + self.list[index][:-1])
		InputImage = self.transform(PILImage)
		del PILImage

		return InputImage, Categorize(self.list[index])

	def __len__(self):
		return len(self.list)

if __name__ == '__main__' :


	DataPath = 'C:\\Users\\Frank\\Machine Learning\\Mura\\dataset\\'
	image = pil_loader(DataPath + 'MURA-v1.1\\train\\XR_ELBOW\\patient00011\\study1_negative\\image1.png')

	transform_train = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ColorJitter(contrast = (2, 2)),
	    transforms.ToTensor(),
	    # transforms.Lambda(Aug),
	    transforms.ToPILImage(),
	])

	plt.figure(figsize = (10, 10))
	plt.imshow(transform_train(image), interpolation='nearest',)

	# plt.figure(figsize = (10, 10))
	# plt.imshow(image, interpolation='nearest',)

	plt.axis('off')

	plt.show()

