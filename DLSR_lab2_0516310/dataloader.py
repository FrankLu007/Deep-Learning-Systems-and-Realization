import os
import PIL
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa


def show_dataset(dataset, n = 5):
	img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n))) for i in range(3)))
	plt.imshow(img)
	plt.axis('off')

baseline = transforms.Compose([
	transforms.Resize((256, 256)),
	transforms.ToTensor(),
	transforms.Normalize([0.4965, 0.3980, 0.3058], [0.3071, 0.2927, 0.2835]),
])

class ImgAugTransform:
	def __init__(self, no = 5):
		self.no = no
		self.aug = iaa.Sequential([
			iaa.Sometimes(0.5, iaa.GaussianBlur(sigma = (0, 3.0))),
			iaa.Fliplr(0.5),
			iaa.Sometimes(0.5, iaa.Affine(rotate = (-45, 45), mode='symmetric')),
			iaa.Sometimes(0.5, iaa.OneOf([iaa.Dropout(p = (0, 0.1)), iaa.CoarseDropout(0.1, size_percent=0.5)])),
        	iaa.Sometimes(0.5, iaa.AddToHueAndSaturation(value = (-10, 10), per_channel=True)),
		])
      
	def __call__(self, img):
		img1 = np.array(img)
		if self.no < 5 :
			img2 = self.aug[self.no].augment_image(img1)
		else :
			img2 = self.aug.augment_image(img1)
		img3 = PIL.Image.fromarray(img2)
		del img, img1, img2
		return img3

class Food11Dataset(torch.utils.data.Dataset):
	def __init__(self, directory, transform):
		super(Food11Dataset, self).__init__()
		self.image_base = directory
		self.images = []
		self.transform = transform

		for category in range(11) :
			path = directory + str(category) + '\\'
			for file in os.listdir(path) :
				self.images.append(str(category) + '\\' + file)

	def __getitem__(self, index):
		with open(self.image_base + self.images[index], 'rb') as f:
			img = PIL.Image.open(f).convert('RGB')
		input = self.transform(img)
		label = int(self.images[index].split('\\')[0])
		del img
		if label > 1 :
			label += 1
		if label == 11 :
			label = 2
		return input, label

	def __len__(self):
		return len(self.images)

demo = transforms.Compose([
	transforms.Resize((256, 256)),
	ImgAugTransform()
])

if __name__ == '__main__' :
	show_dataset(Food11Dataset('C:\\Users\\Frank\\Machine Learning\\DLSR\\dataset\\skewed_training\\', demo))
	plt.show()