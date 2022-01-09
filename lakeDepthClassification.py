# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy import stats
import pylab as pl
from scipy import stats
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier


# ------------------------------------LOAD DATA---------------------------------------------
def load_data(path):

	data =  pd.read_csv('/content/drive/MyDrive/glacierMLproject/filtered_dataset.csv')

	return data



# ------------------------------------DATA SUMMARY------------------------------------------
def describe_data(data):

	# PRINT THE FIRST 5 ROWS OF THE DATA
	print(data.head(5)) 

	# PRINT THE DATA TYPE AND NUMBER OF NON-NULL VALUES IN EACH COLUMN
	data.info()

	# PRINT THE DATA STATISTICS
	print(data.describe())


# -----------------------------------DATA PREPROCESSING-------------------------------------------------------------------
def preprocess_data(data):

	# APPLY BAND AND ELEVATION CONSTRAINTS
	data=data[data['elevation_m']>0]
	data=data[data['B10']>0]
	data=data[data['B11']>0]


	# REMOVE OUTLIERS USING 3-SIGMA RULE
	z_scores = stats.zscore(data[['maxdepth','elevation_m','lake_area_m2','B2','B3','B4','B5','B6','B7','B10','B11']])

	abs_z_scores = np.abs(z_scores)
	filtered_entries = (abs_z_scores <3).all(axis=1)
	data = data[filtered_entries]


	# CATEGORIZE THE maxdepth INTO y LABELS
	data.loc[(data['maxdepth'] <= 15) & (data['maxdepth'] >0), 'target'] = 0
	data.loc[(data['maxdepth'] >15), 'target'] = 1


	return data



# ------------------------------------DATA VISUALIZATION-------------------------------------------
def visualize_data(data):

	# CORRELATION HEATMAP
	plt.figure(figsize=(20, 8))
	sns.heatmap(data.corr(method='spearman'), annot = True)
	plt.show()


	# HISTOGRAM 
	data.hist(figsize=(15,15))
	plt.show()




# ------------------------------------PREPARE TRAIN AND TEST DATA-----------------------------------------------
def prepare_train_test_data(data):

	# FEATURE SELECTION
	data1 = data.copy()
	data1 = data.drop(['B3','B4','B7','B11'], axis = 1)

	# STORE THE DATAFRAME VALUES INTO ARRAY
	X = data1.iloc[:, 1:-1].values    #   X -> Feature Variables
	y = data1.iloc[:, -1].values #   y ->  Target


	# SPLIT DATA INTO TRAIN AND TEST
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 16, stratify=y)


	## DATA STANDARDIZATION
	scalerX = StandardScaler().fit(X_train)
	X_train = scalerX.transform(X_train)
	X_test = scalerX.transform(X_test)


	return X_train, X_test, y_train, y_test




# ----------------------------------INITIALIZE THE MODEL------------------------------------
def model(X_train, y_train):

	Model = LGBMClassifier(
						    # objective='multiclass',
						    boosting='gbdt',
						    learning_rate = 0.01,
						    max_depth = 8,
						    num_leaves = 10,
						    max_bin = 512,
						    feature_fraction = 0.8,
						    n_estimators = 3000,
						    bagging_fraction = 0.9,
						    reg_alpha = 2.5,
						    reg_lambda = 2.5)



	Model.fit(X_train, y_train)

	return Model



# ----------------------------------MODEL PREDICTION AND RESULTS------------------------------------------
def model_predict(Model, X_train, X_test, y_train, y_test):

	# MODEL PREDICTION ON TEST DATA
	y_pred = Model.predict(X_test)
	train_pred = Model.predict(X_train)

	# RESULTS

	## CONFUSION MATRIX
	cf_matrix = confusion_matrix(y_pred, y_test)

	print(classification_report(y_test, y_pred))
	print(confusion_matrix(y_pred, y_test))

	## ACCURACY SCORE

	### TEST ACCURACY
	print('Test accuracy is ',accuracy_score(y_pred, y_test))

	### TRAIN ACCURACY
	print('Train accuracy is ',accuracy_score(train_pred, y_train))


	# VISUALIZE THE CONFUSION MATRIX
	group_names = ['True Neg','False Pos','False Neg','True Pos']
	
	group_percentages = ['{0:.2%}'.format(value) for value in
	                     cf_matrix.flatten()/np.sum(cf_matrix)]

	labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]

	labels = np.asarray(labels).reshape(2,2)


	sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
	plt.show()


	return y_pred



# ================================================ MAIN ===============================================
def main():

	path = 'dataset.csv'
	# LOAD THE DATASET
	data = load_data(path)

	# DESCRIBE THE DATASET
	describe_data(data)

	# PREPROCESS THE DATASET TO REMOVE OUTLIERS AND SET CONSTRAINTS
	data = preprocess_data(data)

	# CHECK CORRLEATIONS AMONG THE FEATURES AND TRUE LABELS
	visualize_data(data)

	# SPLIT THE DATASET INTO TRAIN AND TEST
	X_train, X_test, y_train, y_test = prepare_train_test_data(data)

	# INITIALIZE A MODEL INSTANCE
	Model = model(X_train, y_train)

	# DRAW PREDICTIONS AND CONCLUDE THE RESULTS
	y_pred = model_predict(Model, X_train, X_test, y_train, y_test)



if __name__ == '__main__':
    main()

    
