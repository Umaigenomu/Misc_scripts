import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p

color = sns.color_palette()
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

try:
    train = pd.read_csv('./Miscellaneous/housepricestrain.csv')
    test = pd.read_csv('./Miscellaneous/housepricestest.csv')
except FileNotFoundError:
    pass

train = pd.read_csv('housepricestrain.csv')
test = pd.read_csv('housepricestest.csv')

# ----------------------------------------exploration and preprocessing--------------------------------
train_id = train.Id
test_id = test.Id
train.drop(['Id'], axis='columns', inplace=True)
test.drop(['Id'], axis='columns', inplace=True)

def plot1(x: str):
    fig, ax = plt.subplots()
    ax.scatter(x=train[x], y=train['SalePrice'])
    ax.set_ylabel('SalePrice', fontsize=13)
    ax.set_xlabel(x, fontsize=13)
    plt.savefig("Scatter of {} and Sale Price.png".format(x), dpi=160)
    plt.close(fig)
# plot1('GrLivArea')  # 2 outliers
train.drop(train[(train.GrLivArea > 4000) & (train.SalePrice < 200000)].index, inplace=True)

# for col in train.columns:
#     plot1(col)

def plot_dist_target():
    sns.distplot(train.SalePrice, fit=stats.norm)  # includes scipy's norm.fit results for comparison
    (mu, sigma) = stats.norm.fit(train.SalePrice)
    plt.legend(['Normal dist. ($\mu=$){:.2f} ($\sigma=$){:.2f}'.format(mu, sigma)])
    plt.ylabel('Frequency')
    fig = plt.figure()  # new fig for prob plot
    res = stats.probplot(train.SalePrice, plot=plt)
    plt.savefig("Dist. and Prob. Plots of Sale Price.png", dpi=160)
def boxcox_transf_target():
    train.SalePrice = stats.boxcox(train.SalePrice)[0]
boxcox_transf_target()
y_train = train.SalePrice.values
# plot_dist_target()

ntrain, ntest = train.shape[0], test.shape[0]
all = pd.concat((train, test), sort=False).reset_index(drop=True)  # Drops old indexes
all.drop(['SalePrice'], axis=1, inplace=True)

def missing_ratio():
    all_na = (all.isnull().sum()/len(all))*100
    all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending=False)
    all_na_df = pd.DataFrame({'Missing Ratio': all_na})
    return all_na, all_na_df
def print_missing_ratio():
    all_na, all_na_df = missing_ratio()
    print(all_na.index)
    print(all_na_df.head())
def plot_missing_ratio():
    all_na = missing_ratio()[0]
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    sns.barplot(x=all_na.index, y=all_na)
    ax.tick_params(rotation=90)
    ax.set_xlabel('Features', fontsize=15)
    ax.set_ylabel('Percentage of missing values', fontsize=15)
    fig.suptitle('Missing data percentage by feature')
    plt.savefig("Missing Data Percentages.png", dpi=160)
    plt.close(fig)
# plot_missing_ratio()

def plot_correlation_matrix():
    cormat = train.corr()
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(cormat, square=True, xticklabels=True, yticklabels=True)
    plt.savefig("Correlation Matrix.png", dpi=160)
    plt.close(fig)
# plot_correlation_matrix()

# train.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis='columns', inplace=True)
# test.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis='columns', inplace=True)
def fill_na_all():
    all.PoolQC.fillna("None", inplace=True)
    all.MiscFeature.fillna("None", inplace=True)
    all.Alley.fillna("None", inplace=True)
    all.Fence.fillna("None", inplace=True)
    all.FireplaceQu.fillna("None", inplace=True)
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all[col] = all[col].fillna("None") # no garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all[col] = all[col].fillna(0)  # no garage
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all[col] = all[col].fillna(0)  # no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all[col] = all[col].fillna('None')  # no basement
    all["MasVnrType"] = all["MasVnrType"].fillna("None")  # no masonry veneer
    all["MasVnrArea"] = all["MasVnrArea"].fillna(0)
    all.MSZoning.fillna(all.MSZoning.mode()[0], inplace=True)
    all.drop(['Utilities'], axis=1, inplace=True)
    all.Functional.fillna("Typ", inplace=True)
    all.Electrical.fillna(all.Electrical.mode()[0], inplace=True)
    all.KitchenQual.fillna(all.KitchenQual.mode()[0], inplace=True)
    all.Exterior1st.fillna(all.Exterior1st.mode()[0], inplace=True)
    all.Exterior2nd.fillna(all.Exterior2nd.mode()[0], inplace=True)
    all.SaleType.fillna(all.SaleType.mode()[0], inplace=True)
    all.MSSubClass.fillna("None", inplace=True)


    # fill in missing values by the median LotFrontage of the neighborhood. Transforms a pd.Series.
    all.LotFrontage = all.groupby("Neighborhood").LotFrontage.transform(lambda x: x.fillna(x.median()))
# label encoding for ordinality
def feat_transf():
    all.MSSubClass = all.MSSubClass.apply(str)  # categorical, not numerical
    all.OverallCond = all.OverallCond.astype(str)  # same thing
    all['YrSold'] = all['YrSold'].astype(str)
    all['MoSold'] = all['MoSold'].astype(str)

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
    for col in cols:
        lbe = LabelEncoder()
        all[col] = lbe.fit_transform([*all[col].values])

    # Adding total sqfootage feature -> total basmnt, frstfl sndflr areas of each house
    all['TotalSF'] = all.TotalBsmtSF + all['1stFlrSF'] + all['2ndFlrSF']
fill_na_all()
# print(all.YrSold.unique())
feat_transf()

def skewness():
    num_feats = all.dtypes[all.dtypes != 'object'].index
    # apply(axis=0) by default (along index, apply the func to each column)
    skewed_feats = all[num_feats].apply(lambda x: stats.skew(x, None)).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    # print(skewness.iloc[[*range(15)] + [*range(-15, 0)]])
    return skewness
def plot_skewness(x):
    ordrd_skewness = skewness()
    fig = plt.figure()
    ax = sns.distplot(all['Fence'], fit=stats.norm)
    plt.show()
# plot_skewness(-15)
def box_skew_cox():
    skewn = skewness()
    to_be_fixed = skewn[np.abs(skewn) > 0.75].index
    for ind in to_be_fixed:
        all[ind] = boxcox1p(all[ind], 0.15)
box_skew_cox()

all = pd.get_dummies(all)
train = all[:ntrain]
test = all[ntrain:]


#  -------------------------------------------------------MODELLING-----------------------------------------------------
'''
https://stats.stackexchange.com/questions/92672/difference-between-primal-dual-and-kernel-ridge-regression
https://medium.com/data-design/how-to-not-be-dumb-at-applying-principal-component-analysis-pca-6c14de5b3c9d
https://nycdatascience.com/blog/student-works/house-price-prediction-with-creative-feature-engineering-and-advanced-regression-techniques/
https://www.kaggle.com/solegalli/feature-engineering-for-house-price-modelling
'''
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, ShuffleSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

n_folds = 5

def rmsleCV(model):
    kf = ShuffleSplit(n_folds, test_size=(1/n_folds), random_state=42)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse



