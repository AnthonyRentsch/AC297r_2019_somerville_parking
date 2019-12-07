from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from predict import load_sklearn_model

def plot_calibration_curve(labels, preds, save_path='../images/calibration_curve.png'):
	'''
	Plot calibration curve.

	Parameters
	----------
	labels : pd.DataFrame
	preds : pd.DataFrame
	save_path : str

	Returns
	-------
	None
	'''

	calibrate_data = labels.merge(preds, how='left', left_on='MBL', right_on='MBL')

	prob_true, prob_pred = calibration_curve(y_true=calibrate_data['has_parking'], 
		y_prob=calibrate_data['driveway_yes'],
		n_bins=10)

	fig, ax = plt.subplots(1, 1, figsize=(20,10))
	ax.plot(prob_pred, prob_true, color='steelblue')
	ax.plot(np.linspace(0,1,50), np.linspace(0,1,50), linestyle='dashed', color='black')
	plt.savefig(save_path, dpi=450)

def perform_calibration(model, scaled_data, labels):
	'''
	Perform calibration.

	Parameters
	----------
	model : fitted scikit-learn model object
	scaled_data : pd.DataFrame
	labels : pd.DataFrame

	Returns
	-------
	calibrated_preds : pd.DataFrame
	'''

	calibrate_data = labels.merge(scaled_data, how='left', left_on='MBL', right_on='MBL')
	calibrate_X = calibrate_data[scaled_data.columns ^ ['MBL']]

	calibrator = CalibratedClassifierCV(base_estimator=model, cv='prefit')
	calibrator.fit(calibrate_X)

	calibrated_preds = calibrator.predict(scaled_data)

	return calibrated_preds


if __name__ == '__main__':
	
	# read in data
	labels = pd.read_csv('../labels/labels_final.csv', index_col=0)
	labels = labels[labels['has_parking'] != 2] # remove unknowns
	preds = pd.read_csv('../data/driveway_predictions.csv', index_col=0)

	tabular_df = pd.read_csv(tabular_df_path, index_col=0)
	cols = list(tabular_df.columns ^ ['MBL'])
	scaled_data = preprocesser.transform(tabular_df[cols])

	# calibration curve
	plot_calibration_curve(labels, preds, '../images/pre_calibration_curve.png')

	# perform calibration
	calibrated_preds = perform_calibration(model, scaled_data, labels)

	# calibration curve again
	plot_calibration_curve(labels, calibrated_preds, '../images/post_calibration_curve.png')
