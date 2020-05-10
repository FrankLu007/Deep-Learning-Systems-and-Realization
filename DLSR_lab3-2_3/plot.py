import sys

d = int(sys.argv[1])
w = int(sys.argv[2])
r = int(sys.argv[3])

with open('Record/Record_%d_%d_%d.txt'%(d, w, r), 'r') as file:
	while True:
		data = file.readline()
		if data.find('Testing') != -1:
			break
		elif data.find('Validation') != -1:
			print(data.split(' ')[2])