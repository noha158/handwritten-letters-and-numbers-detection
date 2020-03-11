from model.network import Network
from flask import Flask, jsonify, render_template, request
from app.preprocessing import *
import sys
import os
import json
# import matplotlib.pyplot as plt

config = {}
labels = []

app = Flask(__name__)
@app.route('/')
def index():
	return render_template('index.html')
@app.route('/predict', methods=['POST'])

def predict():
	def map_char(char):
		return labels[char]

	if (request.method == "POST"):
		img = request.get_json()
		img = preprocess(img)
		# plt.imshow(img[0])
		# plt.savefig('image_debug.jpg')
		# plt.close()
		net = Network(network_config)
		character, confidence = net.predict_with_pretrained_weights(img)
		print('received', character, confidence)
		data = { "character": map_char(character), "confidence": float(int(confidence*100))/100. }
		return jsonify(data)

if __name__ == "__main__":
	print('INITIALIZING...\n')

	if len(sys.argv) < 2:
		print('Failure: Expected argument (configuration file [/path/to/file/config.json])\n')
		sys.exit()

	config_file_path = sys.argv[1]

	# Init all possible labels for response mapping
	for i in range(0, 36):
		if i < 10:
			labels.append(chr(i + 48))
		else:
			labels.append(chr(i + 87))
	labels.append('a')
	labels.append('b')
	labels.append('d')
	labels.append('e')
	labels.append('f')
	labels.append('g')
	labels.append('h')
	labels.append('n')
	labels.append('q')
	labels.append('r')
	labels.append('t')

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

	network_config_keys = ['batch_size', 'epochs', 'learning_rate', 'layers', 'weights_filename', 'test_size', 'scores_filename']
	network_config = dict((k, config[k]) for k in network_config_keys if k in config)

	print('APPLICATION READY...\n')

	app.run(debug=True)
