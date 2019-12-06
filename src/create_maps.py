# imports
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib as mpl

def create_label_map(labels, save_path, parcels_path='../data/Parcels_FY19'):
	'''
	Create map of where we labeled data.

	Parameters
	----------
	labels : pd.DataFrame
	save_path : str
	parcels_path : str

	Returns
	-------
	None
	'''

	# load data
	parcels = geopandas.read_file(parcels_path)

	# merge
	parcel_labels = parcels.merge(labels, how='left', left_on='TaxParMBL', right_on='MBL')

	# create map
	cmap = mpl.colors.ListedColormap(["red", "white", "green", "blue"])
	fig, ax = plt.subplots(figsize=(20,10))
	parcel_labels.plot(ax=ax, color='white', edgecolor='grey', alpha=0.1)
	parcel_labels.plot(ax=ax, column='3_labels', alpha=0.9, legend=True, cmap=cmap)
	plt.title("", fontsize=22)
	plt.axis('off')

	# save map
	plt.savefig(save_path, dpi=800)
	

def create_predictions_map(parcels_path, predictions_path):
	return

if __name__ == '__main__':

	# read labels
	manual_labels = pd.read_csv('../labels/training_labels_updated_111219.csv')
	additional_manual_labels = pd.read_csv('../labels/additional_training_labels_120319.csv')
	labels = pd.concat([manual_labels, additional_manual_labels])

	# map to 3 categories
	label_mapping = {0: 'no', 0.1:'no', 0.5:'unsure', 0.9:'yes', 1:'yes', np.nan:'no label'}
	abels['3_labels'] = labels['final_label'].map(label_mapping)

	# create maps
	create_label_map()
	create_predictions_map()

