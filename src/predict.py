# imports
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from generator import generator_two_inputs
from metrics import sensitivity, specificity


def load_model(model_path):
	'''
	Loads in a previously stored Keras model object in HDF5 format.

	Parameters
	----------
	model_path : str

	Returns
	-------
	model : Keras model object
	'''

	model = tf.keras.models.load_model(model_path, 
		custom_objects={'sensitivity':sensitivity, 'specificity':specificity})
	return model

def predict(model, data, 
	aerial_dir='../data/training/aerial_images/', gsv_dir='../data/training/street_view_images/',
	gsv_image_dim = (128,128, 3), aer_image_dim = (128,128, 4)):
	'''
	Generates model predictions for data.

	Parameters
	----------
	model : Keras model object
	data : pd.DataFrame

	Returns
	-------
	preds : np.array
	'''

	preds = model.predict_generator(generator_two_inputs(data, aerial_dir = aerial_dir, gsv_dir = gsv_dir, 
		batch_size = data.shape[0], gsv_image_dim = gsv_image_dim, aer_image_dim = aer_image_dim),
	steps = 1)

	return preds


if __name__ == '__main__':

	# load model
	model_path = '../models/basicmodel.h5'
	model = load_model(model_path)

	# load data
	data = pd.read_csv('../labels/training_labels_updated.csv')
	data['temp_label'] = data['final_label'].apply(lambda x: np.round(x)).astype('int').astype('str')

	# sample data for illustrative purposes
	addresses_gsv_filename = ['1_ESSEX_ST.jpg', '8_GILMAN_ST.jpg', '9_MELVILLE_RD.jpg','10_CENTRAL_ST.jpg',
                         '14_MANSFIELD_ST.jpg']
	pred_sample = data[data.gsv_filename.isin(addresses_gsv_filename)]
	
	# get model predictions
	preds = predict(model, pred_sample)

	# print results
	print('Predictions\n', preds)
	print(f"\nThere are {preds.sum()} drvieways.")


