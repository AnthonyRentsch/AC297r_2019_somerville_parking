# imports
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import optimizers
import pandas as pd
import numpy as np
import glob

import sys
sys.path.append('../')
from generator import generator_three_inputs
from models import three_input_model
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

	# redefine optimizer used when model was compiled
	adam = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, amsgrad=False)

	# load model without compiling
	model = tf.keras.models.load_model(model_path,
		custom_objects={'sensitivity':sensitivity, 'specificity':specificity},
		compile = False)
	
	# compile model with correct loss function and optimizer
	model.compile(loss = 'categorical_crossentropy', optimizer=adam)

	return model

def predict(model, 
	data, 
	y_column='temp_label',
	aerial_dir='../data/training/aerial_images/', 
	gsv_dir='../data/training/street_view_images/',
	tabular_path='../data/residence_addresses_googlestreetview_clean.csv',
	tabular_predictor_cols=None,
	gsv_image_dim = (128,128, 3), 
	aer_image_dim = (128,128, 4)):
	'''
	Generates model predictions for data.

	Parameters
	----------
	model : Keras model object
	data : pd.DataFrame
	y_column : str
	aerial_dir : str
	gsv_dir : str
	gsv_image_dim : tuple 
	aer_image_dim : tuple

	Returns
	-------
	preds : np.array
	'''

	# load tabular data
	tabular_df = pd.read_csv(tabular_path, index_col=0)
	tabular_predictor_cols = list(tabular_df.columns ^ ['MBL']) # TEMP

	# create generator
	gen = generator_three_inputs(data, 
		tabular_data = tabular_df,
		tabular_predictor_cols = tabular_predictor_cols, #tabular_predictor_cols,
		aerial_dir = aerial_dir, gsv_dir = gsv_dir, 
		batch_size = data.shape[0], gsv_image_dim = gsv_image_dim, aer_image_dim = aer_image_dim,
		y_column = y_column)

	# predict using this generator
	preds = model.predict_generator(gen, steps = 1)

	return preds

def create_parcel_df(aerial_dir='../data/training/aerial_images/', 
	gsv_dir='../data/training/sv_images/',
	parcel_mbl_path='../data/Parcels_FY19/VisionExtract_FY19.txt',
	temp_label_col = 'temp_label'):
	'''
	Creates dataframe to pass into predict function for all parcels in Somerville.

	Parameters
	----------
	aerial_dir : str
	gsv_dir : str
	parcel_mbl_path : str
		Path to parcel data file, i.e. Vision Extract.
	temp_label_col : str
		Name of column to use for label. Label is not real 
		but specificying one is needed for data generator.

	Returns
	-------
	df : pd.DataFrame
	'''

	# get all file names
	aerial_files = glob.glob(aerial_dir + '*.png') 
	gsv_files = glob.glob(gsv_dir + '*.jpg') 

	# strip them of extension so we can match them
	aerial_files_raw = [x.split('/')[-1].replace('_aerial.png', '') for x in aerial_files]
	gsv_files_raw = [x.split('/')[-1].replace('.jpg', '') for x in gsv_files]

	# get MBLs to merge tabular features 
	### FIX ##
	parcel_df = pd.read_csv(parcel_mbl_path, error_bad_lines=False)
	parcel_df = parcel_df[['SITE_ADDR', 'MBL']]
	parcel_df['SITE_ADDR'] = parcel_df['SITE_ADDR'].str.replace(' ', '_')
	
	# create dataframes to join
	aerial_df = pd.DataFrame(list(zip(aerial_files, aerial_files_raw)), 
		columns=['aerial_filename', 'aerial_ADDR'])
	gsv_df = pd.DataFrame(list(zip(gsv_files, gsv_files_raw)), 
		columns=['gsv_filename', 'gsv_ADDR'])
	
	# merge on street name - keep all records for which we have aerial/GSV imagery
	### this seems to return wrong number of rows ###
	df = aerial_df.merge(gsv_df, how='outer', left_on='aerial_ADDR', right_on='gsv_ADDR')
	df = df.merge(parcel_df, how='inner', left_on='aerial_ADDR', right_on='SITE_ADDR')

	# append temp label column - only needed to instantiate generator
	df[temp_label_col] = np.random.choice(['0', '1', '2'], size=df.shape[0])

	return df

if __name__ == '__main__':

	# load model
	print('Loading model...')
	model_path = '../models/imageandtabular_model.h5'
	model = load_model(model_path)

	# # load data
	# data = pd.read_csv('../labels/training_labels_updated.csv')
	# print(data.dtypes)
	# data['temp_label'] = data['final_label'].apply(lambda x: np.round(x)).astype('int').astype('str')

	# # sample data for illustrative purposes
	# addresses_gsv_filename = ['1_ESSEX_ST.jpg', '8_GILMAN_ST.jpg', '9_MELVILLE_RD.jpg','10_CENTRAL_ST.jpg',
 #                         '14_MANSFIELD_ST.jpg']
	# pred_sample = data[data.gsv_filename.isin(addresses_gsv_filename)]
	
	# # get model predictions
	# preds = predict(model, pred_sample)

	# # print results
	# print('Predictions\n', preds)
	# print(f"\nThere are {preds.sum()} driveways.")

	print('Loading data...')
	df = create_parcel_df()
	
	print('Making predictions...')
	preds = predict(model, df)

	print(preds)

	### currently no images being found by generator ###


