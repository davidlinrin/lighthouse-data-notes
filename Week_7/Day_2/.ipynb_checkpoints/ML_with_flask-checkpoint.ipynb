{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 233,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.datasets import load_wine\n",
    "import json\n",
    "\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.pipeline import Pipeline, FeatureUnion\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.feature_selection import SelectKBest\n",
    "\n",
    "import sklearn\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "import pickle\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>alcohol</th>\n",
       "      <th>malic_acid</th>\n",
       "      <th>ash</th>\n",
       "      <th>alcalinity_of_ash</th>\n",
       "      <th>magnesium</th>\n",
       "      <th>total_phenols</th>\n",
       "      <th>flavanoids</th>\n",
       "      <th>nonflavanoid_phenols</th>\n",
       "      <th>proanthocyanins</th>\n",
       "      <th>color_intensity</th>\n",
       "      <th>hue</th>\n",
       "      <th>od280/od315_of_diluted_wines</th>\n",
       "      <th>proline</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14.23</td>\n",
       "      <td>1.71</td>\n",
       "      <td>2.43</td>\n",
       "      <td>15.6</td>\n",
       "      <td>127.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>3.06</td>\n",
       "      <td>0.28</td>\n",
       "      <td>2.29</td>\n",
       "      <td>5.64</td>\n",
       "      <td>1.04</td>\n",
       "      <td>3.92</td>\n",
       "      <td>1065.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>13.20</td>\n",
       "      <td>1.78</td>\n",
       "      <td>2.14</td>\n",
       "      <td>11.2</td>\n",
       "      <td>100.0</td>\n",
       "      <td>2.65</td>\n",
       "      <td>2.76</td>\n",
       "      <td>0.26</td>\n",
       "      <td>1.28</td>\n",
       "      <td>4.38</td>\n",
       "      <td>1.05</td>\n",
       "      <td>3.40</td>\n",
       "      <td>1050.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13.16</td>\n",
       "      <td>2.36</td>\n",
       "      <td>2.67</td>\n",
       "      <td>18.6</td>\n",
       "      <td>101.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>3.24</td>\n",
       "      <td>0.30</td>\n",
       "      <td>2.81</td>\n",
       "      <td>5.68</td>\n",
       "      <td>1.03</td>\n",
       "      <td>3.17</td>\n",
       "      <td>1185.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14.37</td>\n",
       "      <td>1.95</td>\n",
       "      <td>2.50</td>\n",
       "      <td>16.8</td>\n",
       "      <td>113.0</td>\n",
       "      <td>3.85</td>\n",
       "      <td>3.49</td>\n",
       "      <td>0.24</td>\n",
       "      <td>2.18</td>\n",
       "      <td>7.80</td>\n",
       "      <td>0.86</td>\n",
       "      <td>3.45</td>\n",
       "      <td>1480.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>13.24</td>\n",
       "      <td>2.59</td>\n",
       "      <td>2.87</td>\n",
       "      <td>21.0</td>\n",
       "      <td>118.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>2.69</td>\n",
       "      <td>0.39</td>\n",
       "      <td>1.82</td>\n",
       "      <td>4.32</td>\n",
       "      <td>1.04</td>\n",
       "      <td>2.93</td>\n",
       "      <td>735.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \\\n",
       "0    14.23        1.71  2.43               15.6      127.0           2.80   \n",
       "1    13.20        1.78  2.14               11.2      100.0           2.65   \n",
       "2    13.16        2.36  2.67               18.6      101.0           2.80   \n",
       "3    14.37        1.95  2.50               16.8      113.0           3.85   \n",
       "4    13.24        2.59  2.87               21.0      118.0           2.80   \n",
       "\n",
       "   flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \\\n",
       "0        3.06                  0.28             2.29             5.64  1.04   \n",
       "1        2.76                  0.26             1.28             4.38  1.05   \n",
       "2        3.24                  0.30             2.81             5.68  1.03   \n",
       "3        3.49                  0.24             2.18             7.80  0.86   \n",
       "4        2.69                  0.39             1.82             4.32  1.04   \n",
       "\n",
       "   od280/od315_of_diluted_wines  proline  \n",
       "0                          3.92   1065.0  \n",
       "1                          3.40   1050.0  \n",
       "2                          3.17   1185.0  \n",
       "3                          3.45   1480.0  \n",
       "4                          2.93    735.0  "
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = load_wine()\n",
    "df = pd.DataFrame(data['data'])\n",
    "df.columns = data['feature_names']\n",
    "y = data['target']\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### CLASS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "metadata": {},
   "outputs": [],
   "source": [
    "class RawFeats:\n",
    "    def __init__(self, feats):\n",
    "        self.feats = feats\n",
    "\n",
    "    def fit(self, X, y=None):\n",
    "        pass\n",
    "\n",
    "\n",
    "    def transform(self, X, y=None):\n",
    "        return X[self.feats]\n",
    "\n",
    "    def fit_transform(self, X, y=None):\n",
    "        self.fit(X)\n",
    "        return self.transform(X)\n",
    "\n",
    "\n",
    "# features we want to keep for PCA\n",
    "feats = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium',\n",
    "         'total_phenols','flavanoids','nonflavanoid_phenols']\n",
    "# creating class object with indexes we want to keep.\n",
    "raw_feats = RawFeats(feats)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "metadata": {},
   "outputs": [],
   "source": [
    "sc = StandardScaler()\n",
    "pca = PCA(n_components=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### SELECTKBEST"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "metadata": {},
   "outputs": [],
   "source": [
    "selection = SelectKBest(k=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf = RandomForestClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "metadata": {},
   "outputs": [],
   "source": [
    "PCA_pipeline = Pipeline([\n",
    "    (\"rawFeats\", raw_feats),\n",
    "    (\"scaler\", sc),\n",
    "    (\"pca\", pca)\n",
    "])\n",
    "\n",
    "kbest_pipeline = Pipeline([(\"kBest\", selection)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_features = FeatureUnion([\n",
    "    (\"pcaPipeline\", PCA_pipeline), \n",
    "    (\"kBestPipeline\", kbest_pipeline)\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "metadata": {},
   "outputs": [],
   "source": [
    "main_pipeline = Pipeline([\n",
    "    (\"features\", all_features),\n",
    "    (\"rf\", rf)\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 27 candidates, totalling 135 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 6 concurrent workers.\n",
      "[Parallel(n_jobs=-1)]: Done   1 tasks      | elapsed:    1.4s\n",
      "[Parallel(n_jobs=-1)]: Done   6 tasks      | elapsed:    1.5s\n",
      "[Parallel(n_jobs=-1)]: Done  13 tasks      | elapsed:    1.5s\n",
      "[Parallel(n_jobs=-1)]: Done  20 tasks      | elapsed:    1.5s\n",
      "[Parallel(n_jobs=-1)]: Batch computation too fast (0.1752s.) Setting batch_size=2.\n",
      "[Parallel(n_jobs=-1)]: Done  29 tasks      | elapsed:    1.6s\n",
      "[Parallel(n_jobs=-1)]: Batch computation too fast (0.0613s.) Setting batch_size=4.\n",
      "[Parallel(n_jobs=-1)]: Done  40 tasks      | elapsed:    1.7s\n",
      "[Parallel(n_jobs=-1)]: Done  64 tasks      | elapsed:    1.8s\n",
      "[Parallel(n_jobs=-1)]: Batch computation too fast (0.0959s.) Setting batch_size=8.\n",
      "[Parallel(n_jobs=-1)]: Done 102 tasks      | elapsed:    1.9s\n",
      "[Parallel(n_jobs=-1)]: Done 121 tasks      | elapsed:    2.0s\n",
      "[Parallel(n_jobs=-1)]: Done 135 out of 135 | elapsed:    2.0s finished\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "GridSearchCV(estimator=Pipeline(steps=[('features',\n",
       "                                        FeatureUnion(transformer_list=[('pcaPipeline',\n",
       "                                                                        Pipeline(steps=[('rawFeats',\n",
       "                                                                                         <__main__.RawFeats object at 0x7ff51654f790>),\n",
       "                                                                                        ('scaler',\n",
       "                                                                                         StandardScaler()),\n",
       "                                                                                        ('pca',\n",
       "                                                                                         PCA(n_components=2))])),\n",
       "                                                                       ('kBestPipeline',\n",
       "                                                                        Pipeline(steps=[('kBest',\n",
       "                                                                                         SelectKBest(k=4))]))])),\n",
       "                                       ('rf', RandomForestClassifier())]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={'features__kBestPipeline__kBest__k': [1, 2, 3],\n",
       "                         'features__pcaPipeline__pca__n_components': [1, 2, 3],\n",
       "                         'rf__n_estimators': [2, 5, 10]},\n",
       "             verbose=10)"
      ]
     },
     "execution_count": 202,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# set up our parameters grid\n",
    "param_grid = {\"features__pcaPipeline__pca__n_components\": [1, 2, 3],\n",
    "                  \"features__kBestPipeline__kBest__k\": [1, 2, 3],\n",
    "                  \"rf__n_estimators\":[2, 5, 10]\n",
    "             }\n",
    "\n",
    "# create a Grid Search object\n",
    "grid_search = GridSearchCV(main_pipeline, param_grid, n_jobs = -1, verbose=10, refit=True)    \n",
    "\n",
    "# fit the model and tune parameters\n",
    "grid_search.fit(df, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 234,
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump( grid_search, open( \"model.p\", \"wb\" ) )\n",
    "joblib.dump(grid_search, open( \"model_joblib.p\", \"wb\" ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 235,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(estimator=Pipeline(steps=[('features',\n",
       "                                        FeatureUnion(transformer_list=[('pcaPipeline',\n",
       "                                                                        Pipeline(steps=[('rawFeats',\n",
       "                                                                                         <__main__.RawFeats object at 0x7ff5164e6a00>),\n",
       "                                                                                        ('scaler',\n",
       "                                                                                         StandardScaler()),\n",
       "                                                                                        ('pca',\n",
       "                                                                                         PCA(n_components=2))])),\n",
       "                                                                       ('kBestPipeline',\n",
       "                                                                        Pipeline(steps=[('kBest',\n",
       "                                                                                         SelectKBest(k=4))]))])),\n",
       "                                       ('rf', RandomForestClassifier())]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={'features__kBestPipeline__kBest__k': [1, 2, 3],\n",
       "                         'features__pcaPipeline__pca__n_components': [1, 2, 3],\n",
       "                         'rf__n_estimators': [2, 5, 10]},\n",
       "             verbose=10)"
      ]
     },
     "execution_count": 235,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.load(open(\"model_joblib.p\", 'rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(estimator=Pipeline(steps=[('features',\n",
       "                                        FeatureUnion(transformer_list=[('pcaPipeline',\n",
       "                                                                        Pipeline(steps=[('rawFeats',\n",
       "                                                                                         <__main__.RawFeats object at 0x7ff51654fc10>),\n",
       "                                                                                        ('scaler',\n",
       "                                                                                         StandardScaler()),\n",
       "                                                                                        ('pca',\n",
       "                                                                                         PCA(n_components=2))])),\n",
       "                                                                       ('kBestPipeline',\n",
       "                                                                        Pipeline(steps=[('kBest',\n",
       "                                                                                         SelectKBest(k=4))]))])),\n",
       "                                       ('rf', RandomForestClassifier())]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={'features__kBestPipeline__kBest__k': [1, 2, 3],\n",
       "                         'features__pcaPipeline__pca__n_components': [1, 2, 3],\n",
       "                         'rf__n_estimators': [2, 5, 10]},\n",
       "             verbose=10)"
      ]
     },
     "execution_count": 204,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pickle.load(open(\"model.p\", 'rb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# OTHERS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'[[140.34, 1.68, 2.7, 0, 98.0, 2.8, 1.31, 5.53, 2.7, 130.0, 4.57, 1.96, 60.0]]'"
      ]
     },
     "execution_count": 208,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import requests\n",
    "import json\n",
    "\n",
    "data = [[140.34, 1.68, 2.7, 0, 98.0, 2.8, 1.31, 5.53, 2.7, 130.0, 4.57, 1.96, 60.0]]\n",
    "j_data = json.dumps(data)\n",
    "j_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "metadata": {},
   "outputs": [],
   "source": [
    "json_data = {'alcohol': 14.23, 'malic_acid':1.71, 'ash':2.43, 'alcalinity_of_ash':15.6, 'magnesium':127.0, 'total_phenols':2.8, 'flavanoids':3.06, 'nonflavanoid_phenols': 0.28, 'proanthocyanins': 2.29, 'color_intensity': 5.64, 'hue': 1.04, 'od280/od315_of_diluted_wines':3.92, 'proline': 1065.0}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 236,
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "URL = \"http://localhost:5000/api/\"\n",
    "# sending get request and saving the response as response object \n",
    "r = requests.get(url = URL, json = json_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 237,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Response [405]>"
      ]
     },
     "execution_count": 237,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 232,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'http://127.0.0.1:5000/api/'"
      ]
     },
     "execution_count": 232,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r.url"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "url = 'http://localhost:5000/api/'\n",
    "\n",
    "data = [[140.34, 1.68, 2.7, 0, 98.0, 2.8, 1.31, 5.53, 2.7, 130.0, 4.57, 1.96, 60.0]]\n",
    "j_data = json.dumps(data)\n",
    "headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}\n",
    "r = requests.post(url, data=j_data, headers=headers)\n",
    "print(r)\n",
    "print(\"Your wine belongs to class: \" + r.text)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
