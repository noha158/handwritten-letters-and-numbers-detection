import numpy as np
import sys
import os
from mlxtend.data import loadlocal_mnist
from model.network import Network
from urllib import urlretrieve
import functools
import operator
import gzip
import struct
import array
import json
import matplotlib.pyplot as plt

print('INITIALIZING...\n')

if len(sys.argv) < 2:
	print('Failure: Expected argument (configuration file [/path/to/file/config.json])\n')
	sys.exit()

config_file_path = sys.argv[1]

print('LOADING CONFIGURATION...\n')

config = {}

if os.path.isfile(config_file_path):
	with open(config_file_path) as config_file:
		try:
			config = json.load(config_file)
		except:
			print('Failure: Invalid configuration file! Make sure the file type is JSON\n')
			sys.exit()
else:
	print('Failure: Configuration file "' + config_file_path + '" not found\n')
	sys.exit()

run_type = config['run_type']

print('LOADING DATASET...\n')
num_classes = config['num_of_classes']
dataset_paths = config['dataset_paths']
removal_pctg = 1 - config['dataset_percentage']

def parse_idx(fd):
	DATA_TYPES = {0x08: 'B',  # unsigned byte
				  0x09: 'b',  # signed byte
				  0x0b: 'h',  # short (2 bytes)
				  0x0c: 'i',  # int (4 bytes)
				  0x0d: 'f',  # float (4 bytes)
				  0x0e: 'd'}  # double (8 bytes)

	header = fd.read(4)
	if len(header) != 4:
		raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

	zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

	if zeros != 0:
		raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
							 'Found 0x%02x' % zeros)

	try:
		data_type = DATA_TYPES[data_type]
	except KeyError:
		raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

	dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
									fd.read(4 * num_dimensions))

	data = array.array(data_type, fd.read())
	data.byteswap()  # looks like array.array reads data as little endian

	expected_items = functools.reduce(operator.mul, dimension_sizes)
	if len(data) != expected_items:
		raise IdxDecodeError('IDX file has wrong number of items. '
							 'Expected: %d. Found: %d' % (expected_items, len(data)))

	return np.array(data).reshape(dimension_sizes)

def load_and_parse(fname):
	fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
	with fopen(fname, 'rb') as fd:
		return parse_idx(fd)

if run_type == 'train' or run_type == 'trainTest':
	train_images = load_and_parse('dataset/train_images')
	train_labels = load_and_parse('dataset/train_labels')
	train_images = train_images[int(len(train_images) * removal_pctg):, :]
	train_labels = train_labels[int(len(train_labels) * removal_pctg):]

if run_type == 'trainTest' or run_type == 'testWithWeights':
	test_images = load_and_parse('dataset/test_images')
	test_labels = load_and_parse('dataset/test_labels')
	test_images = test_images[int(len(test_images) * removal_pctg):, :]
	test_labels = test_labels[int(len(test_labels) * removal_pctg):]


print('PREPROCESSING DATASET...\n')
img_width = config['image_width']
img_height = config['image_height']
 
if run_type == 'train' or run_type == 'trainTest':
	train_images -= int(np.mean(train_images))
	train_images /= int(np.std(train_images))

if run_type != 'train':
	test_images -= int(np.mean(test_images))
	test_images /= int(np.std(test_images))


if run_type == 'train' or run_type == 'trainTest':
	training_data = train_images.reshape(train_images.shape[0], 1, img_width, img_height)
	training_labels = np.eye(num_classes)[train_labels]
	before_img = np.copy(train_images[1000])
	for i in range(0, len(training_data)):
		training_data[i][0] = np.rot90(training_data[i][0])
		training_data[i][0] = np.flip(training_data[i][0], axis=0)

if run_type != 'train':
	testing_data = test_images.reshape(test_images.shape[0], 1, img_width, img_height)
	testing_labels = np.eye(num_classes)[test_labels]
	for i in range(0, len(testing_data)):
		testing_data[i][0] = np.rot90(testing_data[i][0])
		testing_data[i][0] = np.flip(testing_data[i][0], axis=0)

# GRAPH DEBUG
# index = 0
# # print(training_data[index].shape)
# for img in testing_data:
# 	# if train_labels[index] == 12345 or train_labels[index] == 2:
# 	print(test_labels[index])
# 	plt.imshow(img[0])
# 	plt.show()
# 	index += 1
# 	break

plt.subplot(1, 2, 1)
plt.title('Without Rotation/Flip')
plt.imshow(before_img)
plt.subplot(1, 2, 2)
plt.title('With Rotation/Flip')
plt.imshow(train_images[1000])
plt.tight_layout()
plt.show()
exit()

print('TRAINING NETWORK...\n')
network_config_keys = ['batch_size', 'epochs', 'learning_rate', 'layers', 'weights_filename', 'test_size', 'scores_filename']
network_config = dict((k, config[k]) for k in network_config_keys if k in config)
net = Network(network_config)

if run_type == 'train' or run_type == 'trainTest':
	net.train(training_data, training_labels)
if run_type == 'trainTest':
	print('TESTING NETWORK...\n')
	net.test(testing_data, testing_labels)
if run_type == 'testWithWeights':
	print('TESTING NETWORK WITH PRETRAINED WEIGHTS...\n')
	net.test_with_pretrained_weights(testing_data, testing_labels)
