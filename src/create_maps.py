# imports
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

def create_label_map(labels, parcels, save_path='../images/label_map.png'):
	'''
	Create map of where we labeled data.

	Parameters
	----------
	labels : pd.DataFrame
	parcels : geopandas.DataFrame
	save_path : str

	Returns
	-------
	None
	'''

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
	plt.savefig(save_path, dpi=450)
	

def create_predictions_map(parcels, predictions, col, bins=None, save_path='../images/predictions_map.png'):
	'''
	Create map with parcel-level predictions.

	Parameters
	----------
	parcels : geopandas.DataFrame
	predictions : pd.DataFrame
	col : str
	bins : list
	save_path : str

	Returns
	-------
	None
	'''

	# load data
	parcel_df_driveways = parcels.merge(predictions, how='left', left_on='TaxParMBL', right_on='MBL')

	# instantiate map
	fig, ax = plt.subplots(figsize=(20,20))
	parcel_df_driveways.plot(ax=ax, color='white', edgecolor='white', alpha=0.1)

	# plot

	if bins: # discrete

		parcel_df_driveways[parcel_df_driveways[col].notnull()].plot(ax=ax, column=col, 
			cmap='Reds', scheme='user_defined', classification_kwds={'bins':bins}, legend=False)

		cmap = plt.cm.Reds 
		cmaplist = [cmap(i) for i in range(cmap.N)]
		cmap = mpl.colors.LinearSegmentedColormap.from_list(
		    'Custom cmap', cmaplist, cmap.N)
		norm = mpl.colors.BoundaryNorm(bins, cmap.N)

		ax_cm = fig.add_axes([0.25, 0.05, 0.5, 0.02]) #left bottom width height
		cbar = mpl.colorbar.ColorbarBase(ax_cm, cmap=cmap, norm=norm, orientation='horizontal',
		    spacing='uniform', ticks=bins, boundaries=bins)
		cbar.ax.tick_params(labelsize=16)

	else: # continuous
		
		parcel_df_driveways.plot(column=col, ax=ax, cmap='Reds', alpha=0.7)	  

		sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=1))
		sm._A = []
		cbar = fig.colorbar(sm, shrink=0.5, orientation='horizontal', pad=0)

	ax.axis('off')

	# save
	plt.savefig(save_path, dpi=450)



if __name__ == '__main__':

	# read in data
	manual_labels = pd.read_csv('../labels/training_labels_updated_111219.csv')
	additional_manual_labels = pd.read_csv('../labels/additional_training_labels_120319.csv')
	labels = pd.concat([manual_labels, additional_manual_labels], axis=0, sort=False)
	parcels = geopandas.read_file('../data/Parcels_FY19')
	calibrated_predictions = pd.read_csv('../data/calibrated_driveway_predictions.csv', index_col=0)

	# create maps
	create_label_map(labels, parcels)
	create_predictions_map(parcels, calibrated_predictions, 'calibrated_yes_driveway')

	# aggregate at block-level and produce new map with discrete scale
	# CAUTION - this breaks when running from command line - run from Jupyter notebook to be safe
	calibrated_predictions['block'] = calibrated_predictions['MBL'].apply(lambda x: x.split('-')[0] + x.split('-')[1])
	block_pred_counts = calibrated_predictions.groupby('block').mean()['calibrated_yes_driveway'].to_frame().reset_index()
	block_pred_counts.columns = ['block', 'block_probs']
	block_calibrated_predictions = calibrated_predictions.merge(block_pred_counts, how='left', left_on='block', right_on='block')

	bins = [0.1,0.5,0.6,0.7,0.8,0.9,1]
	create_predictions_map(parcels, block_calibrated_predictions, bins, 
		'block_probs', '../images/block_predictions_map.png')

