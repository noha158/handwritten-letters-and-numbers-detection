import numpy as np
import pickle 
import sys
from time import *
from model.loss import *
from model.layers import *
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Network:
	def __init__(self, config):
		self.lr = config['learning_rate']
		self.layers = []
		self.epochs = config['epochs']
		self.set_layers(config['layers'])
		self.lay_num = len(self.layers)
		self.batch_size = config['batch_size']
		self.test_size = config['test_size']
		self.scores_filename = config['scores_filename']
		self.weights_file = config['weights_filename']

	def set_layers(self, layers):
		layer_name_idx = 1
		for layer in layers:
			if layer['type'] == 'Conv':
				self.layers.append(Convolution2D(inputs_channel=layer['input_channels'], num_filters=layer['filter_count'], kernel_size=layer['filter_size'], padding=layer['padding'], stride=layer['stride'], learning_rate=self.lr, name="conv{0}".format(layer_name_idx)))
				layer_name_idx += 1
			elif layer['type'] == 'Dense':
				self.layers.append(Dense(num_inputs=layer['input'], num_outputs=layer['output'], learning_rate=self.lr, name="fc{0}".format(layer_name_idx)))
				layer_name_idx += 1
			elif layer['type'] == 'MaxPool':
				self.layers.append(Maxpooling2D(pool_size=layer['size'], stride=layer['stride'], name="maxpool{0}".format(layer_name_idx)))
				layer_name_idx += 1
			elif layer['type'] == 'ReLu':
				self.layers.append(ReLu())
			elif layer['type'] == 'Flatten':
				self.layers.append(Flatten())
			elif layer['type'] == 'Softmax':
				self.layers.append(Softmax())
			else:
				print("Failure: Invalid layer type \"{0}\"\n".format(layer['type']))
				sys.exit()


	def train(self, training_data, training_label):
		try:
			total_acc = 0
			for e in range(self.epochs):
				for batch_index in range(0, training_data.shape[0], self.batch_size):
					# batch input
					if batch_index + self.batch_size < training_data.shape[0]:
						data = training_data[batch_index:batch_index+self.batch_size]
						label = training_label[batch_index:batch_index + self.batch_size]
					else:
						data = training_data[batch_index:training_data.shape[0]]
						label = training_label[batch_index:training_label.shape[0]]
					loss = 0
					acc = 0
					start_time = time()
					for b in range(self.batch_size):
						if b >= len(data):
							break
						x = data[b]
						y = label[b]
						# forward pass
						for l in range(self.lay_num):
							output = self.layers[l].forward(x)
							x = output
						loss += cross_entropy(output, y)
						if np.argmax(output) == np.argmax(y):
							acc += 1
							total_acc += 1
						# backward pass
						dy = y
						for l in range(self.lay_num-1, -1, -1):
							dout = self.layers[l].backward(dy)
							dy = dout
					# time
					end_time = time()
					batch_time = end_time-start_time
					remain_time = (training_data.shape[0]*self.epochs-batch_index-training_data.shape[0]*e)/self.batch_size*batch_time
					hrs = int(remain_time)/3600
					mins = int((remain_time/60-hrs*60))
					secs = int(remain_time-mins*60-hrs*3600)
					# result
					loss /= self.batch_size
					batch_acc = float(acc)/float(self.batch_size)
					training_acc = float(total_acc)/float((batch_index+self.batch_size)*(e+1))
					print('=== Epoch: {0:d}/{1:d} === Iter:{2:d} === Loss: {3:.2f} === BAcc: {4:.2f} === TAcc: {5:.2f} === Remain: {6:d} Hrs {7:d} Mins {8:d} Secs ===\n'.format(e,self.epochs,batch_index+self.batch_size,loss,batch_acc,training_acc,int(hrs),int(mins),int(secs)))
			# dump weights and bias
			obj = []
			for i in range(self.lay_num):
				cache = self.layers[i].extract()
				obj.append(cache)
			with open(self.weights_file, 'wb') as handle:
				pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
		except:
			# dump weights and bias
			obj = []
			for i in range(self.lay_num):
				cache = self.layers[i].extract()
				obj.append(cache)
			with open(self.weights_file, 'wb') as handle:
				pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


	def test(self, data, label):
		toolbar_width = 40
		sys.stdout.write("[%s]" % (" " * (toolbar_width-1)))
		sys.stdout.flush()
		sys.stdout.write("\b" * (toolbar_width))
		step = float(self.test_size)/float(toolbar_width)
		st = 1
		total_acc = 0

		y_true = []
		y_pred = []
		for i in range(self.test_size):
			if i == round(step):
				step += float(self.test_size)/float(toolbar_width)
				st += 1
				sys.stdout.write(".")
				#sys.stdout.write("%s]a"%(" "*(toolbar_width-st)))
				#sys.stdout.write("\b" * (toolbar_width-st+2))
				sys.stdout.flush()
			if i >= len(data):
				break
			x = data[i]
			y = label[i]
			for l in range(self.lay_num):
				output = self.layers[l].forward(x)
				x = output
			y_pred.append(np.argmax(output))
			y_true.append(np.argmax(y))
			if np.argmax(output) == np.argmax(y):
				total_acc += 1
		sys.stdout.write("\n")
		 # scores
		
		scores_dict = {'Accuracy': "{0:.2f}".format(float(total_acc)/float(self.test_size)), 'F1': f1_score(y_true, y_pred, average='micro'), 'Precision': precision_score(y_true, y_pred, average='micro'), 'Recall': recall_score(y_true, y_pred, average='micro')}
		# print('IT WORKEEED WEEEEE')
		with open(self.scores_filename, 'w') as outfile:
			json.dump(scores_dict, outfile)

		print('=== Test Size:{0:d} === Test Acc:{1:.2f} ===\n'.format(self.test_size, float(total_acc)/float(self.test_size)))
	
	def feed_layers(self, pkl):
		idx = 0
		for layer in self.layers:
			if layer.type == 'Conv' or layer.type == 'Dense':
				self.layers[idx].feed(pkl[idx]["{0}.weights".format(layer.name)], pkl[idx]["{0}.bias".format(layer.name)])
			
			idx += 1

	def test_with_pretrained_weights(self, data, label):
		with open(self.weights_file, 'rb') as handle:
			b = pickle.load(handle)
		self.feed_layers(b)
		toolbar_width = 40
		sys.stdout.write("[%s]" % (" " * (toolbar_width-1)))
		sys.stdout.flush()
		sys.stdout.write("\b" * (toolbar_width))
		step = float(self.test_size)/float(toolbar_width)
		st = 1
		total_acc = 0
		y_true = []
		y_pred = []
		for i in range(self.test_size):
			if i == round(step):
				step += float(self.test_size)/float(toolbar_width)
				st += 1
				sys.stdout.write(".")
				#sys.stdout.write("%s]a"%(" "*(toolbar_width-st)))
				#sys.stdout.write("\b" * (toolbar_width-st+2))
				sys.stdout.flush()
			x = data[i]
			y = label[i]
			for l in range(self.lay_num):
				output = self.layers[l].forward(x)
				x = output
			y_pred.append(np.argmax(output))
			y_true.append(np.argmax(y))
			if np.argmax(output) == np.argmax(y):
				total_acc += 1

		sys.stdout.write("\n")

		# scores
		scores_dict = {'Accuracy': "{0:.2f}".format(float(total_acc)/float(self.test_size)), 'F1': f1_score(y_true, y_pred, average='micro'), 'Precision': precision_score(y_true, y_pred, average='micro'), 'Recall': recall_score(y_true, y_pred, average='micro')}
		
		with open(self.scores_filename, 'w') as outfile:
			json.dump(scores_dict, outfile)
		print('=== Test Size:{0:d} === Test Acc:{1:.2f} ===\n'.format(self.test_size, float(total_acc)/float(self.test_size)))
	
	
	def predict_with_pretrained_weights(self, inputs):
		with open(self.weights_file, 'rb') as handle:
			b = pickle.load(handle)
		self.feed_layers(b)

		for l in range(self.lay_num):
			output = self.layers[l].forward(inputs)
			inputs = output
		digit = np.argmax(output)
		probability = output[0, digit]
		return digit, probability

