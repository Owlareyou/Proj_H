Å#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:59:34 2023

@author: jing
"""
#%%
#print("Welcome to Machine Learning!")

import sys
assert sys.version_info >= (3, 7)

from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")


import os
abspath = os.path.abspath(__file__) #current file path
dname = os.path.dirname(abspath)#the directory of this file, dataset is also here
os.chdir(dname)



#%%
#Data importation


from pathlib import Path #a better way of os
import pandas as pd
import tarfile #read and write tar archive files
import urllib.request #HOWTO Fetch Internet Resources Using The urllib Package¶
import numpy as np

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
#%%
#Data Structure (20640, 10)
import matplotlib.pyplot as plt

housing.head() #10 attributes

housing.info() #9 float64, 1 text object
#total bedroom has Null

housing['ocean_proximity'].value_counts()

housing.describe()

#%%
# extra code – code to save the figures as high-res PNGs for the book

IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")  # extra code
plt.show()
#%%





#%%
#create test set

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)

#not good enough, need stratified sampling to avoid bias
#ensure test set has representative for all incomes

#cut median income
housing['income_cat'] = pd.cut(housing['median_income'], bins = [0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
housing['income_cat'].value_counts().sort_index().plot.bar(rot = 0, grid = True)
plt.xlabel('Income Category')
plt.ylabel('Labels')
plt.show()


#startidied plits of the same dataset
from sklearn.model_selection import StratifiedShuffleSplit

"""
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []

for train_index, test_index in splitter.split(housing, housing['income_cat']):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
    

#start with the first split
strat_train_set_n, strat_test_set_n = strat_splits[0]
"""

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing['income_cat'], random_state=42)

#%%
#check to see if stratification work
strat_train_set.head()

strat_train_set['income_cat'].value_counts()/len(strat_train_set)
#these two are similar in value
train_set['income_cat'].value_counts() / len(train_set)


#revert income_cat
for set_ in (strat_test_set, strat_train_set):
    set_.drop('income_cat', axis = 1, inplace = True)

#%%
#explore + visualize data

#make copy of exploratory data
housing = strat_train_set.copy()

housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', grid = True, 
             s = housing['population']/100, label = 'Population', 
             c = 'median_house_value', cmap = 'jet', colorbar = True,
             legend = True, sharex = False, figsize = (10,7))
plt.show()
#%%
# extra code – this cell generates the first figure in the chapter

# Download the California image
filename = "california.png"
if not (IMAGES_PATH / filename).is_file():
    homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
    url = homl3_root + "images/end_to_end_project/" + filename
    print("Downloading", filename)
    urllib.request.urlretrieve(url, IMAGES_PATH / filename)

housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed.plot(
             kind="scatter", x="Longitude", y="Latitude",
             s=housing_renamed["Population"] / 100, label="Population",
             c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))

california_img = plt.imread(IMAGES_PATH / filename)
axis = -124.55, -113.95, 32.45, 42.05
plt.axis(axis)
plt.imshow(california_img, extent=axis)

save_fig("california_housing_prices_plot")
plt.show()
#%%





#%%
#look for linear correlation
#housing = housing.iloc[:,:-1]

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)

from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize=(12,8))
plt.show()

#best correlation attributes
housing.plot(kind='scatter', x = 'median_income', y = 'median_house_value',
             alpha = 0.1, grid = True)
plt.show()



#attribute combinations
#more attributes
housing['rooms_per_house'] = housing['total_rooms']/ housing['households'] #correlation of 0.14
housing['bedrooms_ratio'] = housing['total_bedrooms']/ housing['total_rooms']
housing['people_per_house'] = housing['population']/ housing['households']

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)

#%%




#%%
#data preperation for machine learning: Nan instances
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()

#deal with Nan in total_bedrooms
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'median')


housing_num = housing.select_dtypes(include =[np.number])
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)

#recover column and index after transformation
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
#%%




#%%
#data preperation for mahcine learning: text and categorical attributes

#convert categorical atttribute to numbers
from sklearn.preprocessing import OrdinalEncoder

housing_cat = housing[['ocean_proximity']]

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

ordinal_encoder.categories_


#turn categorical values into one-hot encoder
from  sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot #this is a sparse matrix because its mostly 0
housing_cat_1hot.toarray()

#skit store column names in .feature_names_in_
cat_encoder.feature_names_in_
cat_encoder.get_feature_names_out()
#%%




#%%
#feature scaling and transformation
#min-max scaling, aka normalization
from sklearn.preprocessing import MinMaxScaler

min_max_scalar = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scalar.fit_transform(housing_num)

#standardization
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

#models do not like heavy tail distribution, a symmetrical one is much better
#use power , log may help

#or, use buecketizing to reshape the feature

#for multimodal distribution, ie two peaks, it is also good to bucketize with few buckets

#for multimodal distribution, add a feature for each of the modes, ie RBF(radial basis function)
from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[['housing_median_age']], [[35]], gamma = 0.1)


#labels, outputs may also need transformation. Remeber to inverse_transform() back
"""
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(), trasformer = strandardscaler())
model.fit(housing[['median_income']], housing_labels)
predictions = model.predict(newdatafromsomewhereelsethatcanbepredicted)
"""
#%%
#custom transformers

from sklearn.preprocessing import FunctionTransformer

#a log transformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

#a rbf transformer
rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

#another sbf
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])


#trainable in the fit() method, can be used later in the transform() method
#custom class
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
    
    
#another custom trasnformer with estrimators in its implimentation
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
#%%
#there are so many steps that need to be in the right order
# skitlearn provides the pipeline class to help

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

num_pipeline = Pipeline([
    ('impute',SimpleImputer(strategy = 'median')), #naming does not matter, its for hyperparameter tuning
    ('standardize', StandardScaler()),
    ])

housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)

#recover loss dataframe featurenames
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns = num_pipeline.get_feature_names_out(),
    index = housing_num.index
    )

from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])


from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)


#%%
#actual pipeline building
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

#run the pipeline
housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape

preprocessing.get_feature_names_out()

#%%




#%%
#time to select training model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_prediction = lin_reg.predict(housing)
print(housing_prediction[:5].round(-2))
print(housing_labels.iloc[:5].values)

lin_rmse = mean_squared_error(housing_labels, housing_prediction, squared = False)
print('lin_rmse:',lin_rmse)

#result: underfitting
#%%
#time for a more powerful model:
#decision tree regessor

from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_prediction, squared = False)

print('tree rmse:', tree_rmse)
#result: overfit
#%%




#%%
#cross-validation, k-fold corss-validation
from sklearn.model_selection import cross_val_score

#its a utility function, not a cost function, so -1

tree_rmses = (-1)*cross_val_score(tree_reg, housing, housing_labels, 
                                  scoring= 'neg_root_mean_squared_error', 
                                  cv = 10)
pd.Series(tree_rmses).describe()

#turns out due to overfitting, the result is just as bad as linear model
#%%
#how about random forest regressor

from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)

pd.Series(forest_rmses).describe()
#short list to 2~5 models, than do hyper parameter tweaking
#%%




#%%
#Fine-tune time
#grid search
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)


grid_search.best_params_


############
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# extra code – these few lines of code just make the DataFrame look nicer
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

cv_res.head()

#%%
#randomized search is prefered if search space is large

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)


#also consider half-search
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
#%%





#%%
#analyze best model + error
final_model = rnd_search.best_estimator_  # includes preprocessing
feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)



#%%
#test Set evaluation
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)


from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))











