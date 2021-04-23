import os
import gradio as gr
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler

# We recreate the final model using the pipeline for easier deployment, those parameters is the same in the part 2

# DATASET_URL = "https://gist.githubusercontent.com/YHYeoh/ad1a7f7170c72d621d05a70637540152/raw/5a6059c199e2c46d2f3d258f03d93cfea98e2749/marketing_campaign.csv"
# data = pd.read_csv(DATASET_URL, sep = ';')

# education_order = [['Basic', 'Graduation', 'Master', '2n Cycle', 'PhD']]
# ordinal_encoder = OrdinalEncoder(categories=education_order)

# data["Education"] = (ordinal_encoder.fit_transform(data["Education"].values.reshape(-1, 1))).astype(int)
# # print(ordinal_encoder.categories_)

# #encode categorical column
# categorical = ['Marital_Status']
# marital_status_ohe = pd.get_dummies(data.Marital_Status,prefix="Marital")
# data = data.join(marital_status_ohe)

# #drop original column after encoding
# data.drop(['Marital_Status'], axis = 1,inplace = True)

# print(data.columns)

# tunable_cols = ["Year_Birth", "Income","Dt_Customer"]

# y = data.Response
# X = data.drop("Response",axis=1)

# numerical_bool_col = [x for x in data.columns if data[x].isin([0,1]).all()] # print(numerical_bool_col)
# numerical_scalable_col = [x for x in data.columns if x not in numerical_bool_col]

# def extractFromDate(data):
# 	data['enroll_year'] = pd.DatetimeIndex(data.Dt_Customer).year
# 	data['enroll_month'] = pd.DatetimeIndex(data.Dt_Customer).month
# 	data['enroll_day'] = pd.DatetimeIndex(data.Dt_Customer).day
# 	data.drop(['Dt_Customer'], axis = 1, inplace= True)
# 	return data

# def getNormalizedAndBinnedIncome(data):
# 	data = data[(np.abs(stats.zscore(data[['Income']])) < 3)]
# 	data['Income'] = pd.cut(data['Income'], bins=[0, 15000, 60000, 110000, 700000], labels=False, precision=0).convert_dtypes()
# 	return data

# classifier_columns = {
# 		"SVC":["AcceptedCmp2","AcceptedCmp5","MntSweetProducts","Complain","MntWines","Year_Birth","MntGoldProds","NumDealsPurchases"],
# 		"LGBMClassifier":["Kidhome","MntWines","Education","Teenhome","AcceptedCmp4","MntFishProducts","AcceptedCmp2","AcceptedCmp5"],
# 		"RandomForestClassifier":["Kidhome","Teenhome","Education","MntWines","MntFishProducts","AcceptedCmp4","AcceptedCmp5","AcceptedCmp2"],
# 		"XGBClassifier":["AcceptedCmp2","AcceptedCmp5","Complain","Marital_Together","Marital_Married","NumDealsPurchases","Kidhome","Year_Birth"],
# 		"LogisticRegression":["AcceptedCmp2","AcceptedCmp5","Complain","AcceptedCmp1","enroll_year","Marital_Married","Marital_Together","Year_Birth"]
# }

# x_copy = X.copy()
# x_copy = x_copy.drop([x for x in x_copy.columns if x not in classifier_columns["SVC"]],axis = 1)

# X_train, X_test, y_train, y_test = train_test_split(x_copy,y, test_size=0.25, random_state=42)

# X_train= X_train.fillna(method = "ffill")
# X_test = X_test.fillna(method = "ffill")

# if('Dt_Customer' in x_copy.columns):
#     X_train = extractFromDate(X_train)
#     X_test = extractFromDate(X_test)

# if('Income' in x_copy.columns):
#     X_train = getNormalizedAndBinnedIncome(X_train)
#     X_test = getNormalizedAndBinnedIncome(X_test)

# y_train = y_train[y_train.index.isin(X_train.index)]
# y_test = y_test[y_test.index.isin(X_test.index)]
# oversampler = RandomOverSampler(sampling_strategy=0.5,random_state=42)
# X_train,y_train = oversampler.fit_resample(X_train, y_train)

# svc = SVC(C=1000,gamma=0.0001, kernel='linear', random_state=42)
# logrec = LogisticRegression(C = 100, solver='newton-cg')
# lgbm = LGBMClassifier(max_depth = 15, metric = 'binary_logloss', min_data = 50, min_split_gain = 0.1, n_estimators = 800, num_leaves = 40, random_state=42, sub_feature=0.1)

# print(X_train.columns)

# model = Pipeline([
#         ('quantileTrans', QuantileTransformer()),
#         ('nystroem', Nystroem()),    
#         ('clf', VotingClassifier(estimators=[("pip1", svc), 
#                                              ("pip2", logrec), 
#                                              ("pip3", lgbm)]))
#     ])


# # pipe = make_pipeline(QuantileTransformer(),Nystroem(),VotingClassifier(estimators = [svc,logrec,lgbm], voting = 'hard'))
# # print(X_train)
# # print(y_train)
# model.fit(X_train, y_train)

# pkl_filename = 'model.pkl'
# with open(pkl_filename, 'wb') as file:
# 	pickle.dump(model, file)

pkl_filename = "model.pkl"
if os.path.exists(pkl_filename):
     with open(pkl_filename, 'rb') as file:  
         voting = pickle.load(file)
         print("Model file loaded successfully")
else:
	print("Model file not found!")

MntSweetProducts = gr.inputs.Slider(maximum = 1000, step = 10,label = "Amount spent on sweet products in the last 2 years")
Complain = gr.inputs.Checkbox(label = "Complained in last 2 years?")
AcceptedCmp2 = gr.inputs.Checkbox(label = "Accepted Campaign 2?")
AcceptedCmp5 = gr.inputs.Checkbox(label = "Accepted Campaign 5?")
MntWines = gr.inputs.Slider(maximum = 1000, step = 10,label = "Amount spent on wine products in the last 2 years")
Year_Birth = gr.inputs.Slider(minimum = 1940, maximum = 2021, step = 1,label = "Year birth")
MntGoldProds = gr.inputs.Slider(maximum = 1000, step = 10,label = "Amount spent on gold products in the last 2 years")
NumDealsPurchases = gr.inputs.Slider(maximum = 100, step = 1,label = "Number of purchases made with discount")
label = gr.outputs.Label(type="auto", label = "Response YES or NO")

def predict(MntSweetProducts,
	Complain,
	AcceptedCmp2,
	AcceptedCmp5,
	MntWines,
	Year_Birth,
	MntGoldProds,
	NumDealsPurchases):
	df = pd.DataFrame.from_dict(
		{
		'yb': [Year_Birth],
		'mntWines':[MntWines],
		'mntsp': [MntSweetProducts],
		'mntgp':[MntGoldProds],
		'numdp':[NumDealsPurchases], 
		'acp2': [AcceptedCmp2],
		'acp5': [AcceptedCmp5],
		'cmp': [Complain], 	
		}
		)
	pred = voting.predict(df)
	print(pred[0])
	return 'Yes' if pred[0] == 1 else 'No'

io = gr.Interface(fn=predict, inputs = [
	MntSweetProducts, 
	Complain, 
	AcceptedCmp2, 
	AcceptedCmp5,
	MntWines,
	Year_Birth,
	MntGoldProds,
	NumDealsPurchases], outputs = label)

io.launch(share = True)
