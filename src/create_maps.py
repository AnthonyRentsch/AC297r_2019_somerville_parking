# imports
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib as mpl

def create_label_map(labels, save_path='../images/label_map.png', parcels_path='../data/Parcels_FY19'):
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

	# map to 3 categories
	label_mapping = {0:'no', 0.1:'no', 0.5:'unsure', 0.9:'yes', 1:'yes', np.nan:'no label'}
	parcel_labels['3_labels'] = parcel_labels['final_label'].map(label_mapping)

	# create map
	cmap = mpl.colors.ListedColormap(["purple", "white", "darkorange", "green"])
	fig, ax = plt.subplots(figsize=(20,20))
	parcel_labels.plot(ax=ax, color='white', edgecolor='grey', alpha=0.4)
	parcel_labels.plot(ax=ax, column='3_labels', alpha=0.9, legend=True, cmap=cmap)
	plt.axis('off')

	# save map
	plt.savefig(save_path, dpi=800)
	

def create_predictions_map(save_path='../images/predictions_map.png',
	parcels_path='../data/Parcels_FY19', 
	predictions_path='../data/driveway_predictions.csv'):
	'''
	Create map with parcel-level predictions.

	Parameters
	----------
	save_path : str
	parcels_path : str
	predictions_path : str

	Returns
	-------
	None
	'''

	# load data
	parcels = geopandas.read_file(parcels_path)
	predictions = pd.read_csv(predictions_path)
	parcel_df_driveways = parcels.merge(predictions, how='left', left_on='TaxParMBL', right_on='MBL')

	# instantiate map
	fig, ax = plt.subplots(figsize=(20,20))

	# plot
	parcel_df_driveways.plot(ax=ax, color='white', edgecolor='white', alpha=0.1)
	parcel_df_driveways.plot(column='driveway_yes', ax=ax, cmap='Reds', alpha=0.7)	  

	# Create colorbar as a legend
	sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=1))
	# empty array for the data range
	sm._A = []
	# add the colorbar to the figure
	cbar = fig.colorbar(sm, shrink=0.5, orientation='horizontal', pad=0)

	plt.axis('off')

	# save
	plt.savefig(save_path, dpi=800)



if __name__ == '__main__':

	# read labels
	manual_labels = pd.read_csv('../labels/training_labels_updated_111219.csv')
	additional_manual_labels = pd.read_csv('../labels/additional_training_labels_120319.csv')
	labels = pd.concat([manual_labels, additional_manual_labels], axis=0, sort=False)

	# create maps
	create_label_map(labels)
	create_predictions_map()

