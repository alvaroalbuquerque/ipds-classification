import numpy as np
import math
import time
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,cross_validate,StratifiedKFold,RepeatedStratifiedKFold,KFold,LeaveOneOut

from sklearn.preprocessing import MultiLabelBinarizer,MinMaxScaler
import ast

from functools import partial
from sklearn.metrics import confusion_matrix,precision_score, make_scorer,recall_score,accuracy_score,classification_report

# read the csv file
filename = "result_radius.csv"
df_original = pd.read_csv('./results/{}'.format(filename))
print(filename,"->  Shape:",df_original.shape)


df_original['phi'] = df_original['phi'].apply(ast.literal_eval)
df_original['haralick'] = df_original['haralick'].apply(ast.literal_eval)
df_original['lbp'] = df_original['lbp'].apply(ast.literal_eval)



returned_analysis = []
for radius_curr in range(2, 9):
	print(radius_curr)
	df = df_original.loc[df_original['radius'] == radius_curr]

	# convert the arrays to columns in different dataframes
	df2 = pd.DataFrame(df['phi'].values.tolist(), index= df.index)
	df3 = pd.DataFrame(df['haralick'].values.tolist(), index= df.index)
	df4 = pd.DataFrame(df['lbp'].values.tolist(), index= df.index)

	# merge the dataframes
	dfFinal = pd.concat([df2, df3, df4], axis=1, ignore_index=True)

	# X is the vectors to be used in classification
	X = dfFinal
	# normalize the values to interval [0, 1]
	X = pd.DataFrame(MinMaxScaler().fit_transform(X))

	# y is the labels of vectors X
	y = df.iloc[:,[-1]]

	accuracy_list_full = []
	recall_list_full = []
	precision_list_full = []

	y_true_full = np.array([])
	y_pred_full = np.array([])
	best = [0,0]
	k = 5

	# fill nan values with 0 and then check if X still has nan values
	X = X.fillna(0) 
	print("NaN counter = ",X.isnull().sum().sum())


	start = time.perf_counter()


	# iterate 100 times
	for count in range(100):
		print(">>>> count: ", count)

		# init KNN classifier with K=5
		classifier_3NN = KNeighborsClassifier(n_neighbors=k, metric='minkowski')

		# init validation method with 10 splits
		skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=None)

		accuracy_list = []
		recall_list = []
		precision_list = []

		# iterate over the splits
		for train_index, test_index in skf.split(X,y):
			X_train, X_test = X.iloc[train_index], X.iloc[test_index]
			y_train, y_test = y.iloc[train_index], y.iloc[test_index]

			# set the training dataset on KNN
			classifier_3NN.fit(X_train,y_train.values.ravel())

			# predict the labels on test set
			y_pred = classifier_3NN.predict(X_test)
			# actual values of the test set
			y_true = y_test.values.ravel()

			a = accuracy_score(y_true, y_pred)
			r = recall_score(y_true, y_pred, average='macro')
			p = precision_score(y_true, y_pred, average='macro')

			accuracy_list.append(a)
			recall_list.append(r)
			precision_list.append(p)

		# get the metrics mean and concatenate on a list
		accuracy_list_full.append(np.mean(accuracy_list))
		recall_list_full.append(np.mean(recall_list))
		precision_list_full.append(np.mean(precision_list))
		print(np.mean(accuracy_list))

	# print the metrics
	print("accuracy: ", np.mean(accuracy_list_full))
	print("recall: ", np.mean(recall_list_full))
	print("precision: ", np.mean(precision_list_full))

	# store the metric for the current radius on a list
	returned_analysis.append((radius_curr,  np.mean(accuracy_list_full), np.mean(recall_list_full), np.mean(precision_list_full)))

df_final = pd.DataFrame(returned_analysis, columns=['radius', 'accuracy', 'recall', 'precision'])
df_final['radius'] = df_final['radius'].astype(int)
df_final['accuracy'] = df_final['accuracy'].astype(float)
df_final['recall'] = df_final['recall'].astype(float)
df_final['precision'] = df_final['precision'].astype(float)

df_final.to_csv('./results/results_radius_metrics.csv', index=False)

# end timer
end = time.perf_counter()
print("Ellapsed time: "+ str(end - start))