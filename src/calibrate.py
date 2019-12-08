from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_calibration_curve(labels, preds, col, save_path='../images/calibration_curve.png'):
	'''
	Plot calibration curve.

	Parameters
	----------
	labels : pd.DataFrame
	preds : pd.DataFrame
	col : str
	save_path : str

	Returns
	-------
	None
	'''

	# create data
	calibrate_data = labels.merge(preds, how='left', left_on='MBL', right_on='MBL')
	calibrate_data = calibrate_data[calibrate_data[col].notnull()]

	# calculate values 
	prob_true, prob_pred = calibration_curve(y_true=calibrate_data['has_parking'], 
		y_prob=calibrate_data[col],
		n_bins=20)

	# plot
	fig, ax = plt.subplots(1, 1, figsize=(20,10))
	ax.plot(prob_pred, prob_true, color='steelblue')
	ax.plot(np.linspace(0,1,50), np.linspace(0,1,50), linestyle='dashed', color='black')
	ax.set_xlabel('Predicted probability', fontsize=22)
	ax.set_ylabel('True occurence', fontsize=22)
	plt.savefig(save_path, dpi=450)

def perform_calibration(raw_probs, labels):
	'''
	Perform calibration.

	Parameters
	----------
	raw_probs : pd.DataFrame
	labels : pd.DataFrame

	Returns
	-------
	calibrated_probs : pd.DataFrame
	'''

	# load and rebalance data
	# add copy of negative class to make it roughly 80-20 balanced
	calibrate_data = labels.merge(raw_probs, how='left', left_on='MBL', right_on='MBL')
	calibrate_data = calibrate_data[calibrate_data['yes_driveway'].notnull()]
	rebalanced_calibrate_data = pd.concat([calibrate_data, 
		calibrate_data[calibrate_data['has_parking']==0]], 
		axis=0)

	# fit isotonic regression
	calibrator = IsotonicRegression()
	calibrator.fit(rebalanced_calibrate_data['yes_driveway'],rebalanced_calibrate_data['has_parking']) 
	
	# predict
	raw_probs['calibrated_yes_driveway'] = calibrator.predict(raw_probs['yes_driveway'])
	calibrated_probs = raw_probs[['MBL', 'calibrated_yes_driveway']]

	return calibrated_probs


if __name__ == '__main__':
	
	# read in data
	labels = pd.read_csv('../labels/labels_final.csv', index_col=0)
	labels = labels[labels['has_parking'] != 2] # remove unknowns
	preds = pd.read_csv('../data/driveway_predictions.csv', index_col=0)

	# calibration curve
	plot_calibration_curve(labels, preds, 'yes_driveway', '../images/pre_calibration_curve.png')

	# perform calibration
	calibrated_preds = perform_calibration(preds, labels)
	calibrated_preds.to_csv('../data/calibrated_driveway_predictions.csv', index=False)

	# calibration curve again
	plot_calibration_curve(labels, calibrated_preds, 'calibrated_yes_driveway', '../images/post_calibration_curve.png')
