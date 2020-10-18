import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import copy
import scipy
from sklearn.preprocessing import FunctionTransformer

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import xgboost as xgb

from sklearn import metrics 
from sklearn.metrics import roc_auc_score
import pickle

data = pd.read_csv("data.csv", header = 0) 
data

data['Dependents'] = np.where(data['Dependents']=='0',0,
                             np.where(data['Dependents']=='1',1, 
                             np.where(data['Dependents']=='2',2, 
                             np.where(data['Dependents']=='3+',3, data['Dependents']))))

# impute marital with mode
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
# impute gender based on income (Bias!)
data.loc[data['ApplicantIncome']<5446, 'Gender'] = data.loc[data['ApplicantIncome']<5446, 'Gender'].fillna('Female')
data.loc[data['ApplicantIncome']>=5446, 'Gender'] = data.loc[data['ApplicantIncome']>=5446, 'Gender'].fillna('Male')
# impute Loan_amount_term with mean
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean())
# impute dependents based on marital
data.loc[data['Married']=='Yes', 'Dependents'] = data.loc[data['Married']=='Yes', 'Dependents'].fillna(1)
data.loc[data['Married']=='No', 'Dependents'] = data.loc[data['Married']=='No', 'Dependents'].fillna(0)
# impute loan amount based on income
data.loc[data['ApplicantIncome']<5403, 'LoanAmount'] = data.loc[data['ApplicantIncome']<5403, 'LoanAmount'].fillna(118)
data.loc[data['ApplicantIncome']>=5403, 'LoanAmount'] = data.loc[data['ApplicantIncome']>=5403, 'LoanAmount'].fillna(216)

# impute Self_employed
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
# imputer Credit_history
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

numerical = list(data.dtypes[data.dtypes != 'object'].index)
categorical = list(data.dtypes[data.dtypes == 'object'].index)

data['co-applicant'] = np.where(data['CoapplicantIncome']==0,0,1)

data['log_ApplicantIncome'] = np.log(data['ApplicantIncome'])
data['log_LoanAmount'] = np.log(data['LoanAmount'])

data['Total_income'] = data['ApplicantIncome'] + data['CoapplicantIncome'] 
data['log_Total_income'] = np.log(data['Total_income'])

data_normal = copy.deepcopy(data.drop(['Loan_ID','ApplicantIncome', 'CoapplicantIncome', 'log_ApplicantIncome', 'LoanAmount','Total_income'], axis = 1))
data_normal.head()

to_std = ['log_Total_income', 'log_LoanAmount','Loan_Amount_Term']

num_std = data_normal[to_std].values
num_scaled = StandardScaler().fit_transform(num_std)

for i in range(len(to_std)):
    data_normal[to_std[i]] = num_scaled[:,i]
    
data_normal['Gender'] = pd.factorize(data_normal['Gender'])[0]
data_normal['Married'] = pd.factorize(data_normal['Married'])[0]
data_normal['Education'] = pd.factorize(data_normal['Education'])[0]
data_normal['Self_Employed'] = pd.factorize(data_normal['Self_Employed'])[0]
data_normal['Property_Area'] = pd.factorize(data_normal['Property_Area'])[0]

data_normal['Loan_Status'] = np.where(data_normal['Loan_Status']=='Y',1,0)

data_normal['Dependents'] = data_normal['Dependents'].astype(int)

X = copy.deepcopy(data_normal.drop(['Loan_Status'], axis = 1))
y = copy.deepcopy(data_normal['Loan_Status'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

rf = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_leaf=3).fit(X_train,y_train)

pickle.dump(rf, open( "project_IV_2.p", "wb" ))