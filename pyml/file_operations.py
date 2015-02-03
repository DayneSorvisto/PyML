import numpy as np

def load(file, sep=','):
	with open(file, 'r') as fp:
		raw_x = fp.read().splitlines()
		x = []
		for idx in raw_x:
			raw_val = idx.split(sep)
			raw_val = list(map(float, raw_val))
			x.append(raw_val)
	return np.matrix(x)

def save(file, x, mode='w', sep=','):
	if mode not in ['w', 'a']:
		raise Exception('Invalid mode')
	raw_x = x.tolist()
	temp = ''
	for i in raw_x:
		raw_val = list(map(str, i))
		temp += sep.join(raw_val) + '\n'
	with open(file, mode) as fp:
		fp.write(temp)
	return True
