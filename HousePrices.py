import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
from scipy.stats import norm, skew

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
    sns.distplot(train.SalePrice, fit=norm)  # includes scipy's norm.fit results for comparison
    (mu, sigma) = norm.fit(train.SalePrice)
    plt.legend(['Normal dist. ($\mu=$){:.2f} ($\sigma=$){:.2f}'.format(mu, sigma)])
    plt.ylabel('Frequency')
    fig = plt.figure()  # new fig for prob plot
    res = stats.probplot(train.SalePrice, plot=plt)
    plt.savefig("Dist. and Prob. Plots of Sale Price.png", dpi=160)
def boxcox_transf_target():
    train.SalePrice = stats.boxcox(train.SalePrice)[0]
y_train = train.SalePrice.values
boxcox_transf_target()
# plot_dist_target()

ntrain, ntest = train.shape[0], test.shape[0]
all = pd.concat((train, test), sort=False).reset_index(drop=True) # Drops old indexes
all.drop(['SalePrice'], axis=1, inplace=True)

def missing_ratio():
    all_na = (all.isnull().sum()/len(all))*100
    all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending=False)
    all_na_df = pd.DataFrame({'Missing Ratio': all_na})
    return all_na, all_na_df
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
def fill_na():
    all.PoolQC.fillna("None", inplace=True)
    all.MiscFeature.fillna("None", inplace=True)
    all.Alley.fillna("None", inplace=True)
    all.Fence.fillna("None", inplace=True)
    all.FireplaceQu.fillna("None", inplace=True)
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all[col] = all[col].fillna("None")  # no garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all[col] = all[col].fillna(0)  # no garage
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all[col] = all[col].fillna(0)  # no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all[col] = all[col].fillna('None')  # no basement
    all["MasVnrType"] = all["MasVnrType"].fillna("None")  # no masonry veneer
    all["MasVnrArea"] = all["MasVnrArea"].fillna(0)
    all.MSZoning.fillna(all.MSZoning.mode()[0], inplace=True)

    # fill in missing values by the median LotFrontage of the neighborhood.
    all.LotFrontage = all.groupby("Neighborhood").LotFrontage.transform(lambda x: x.fillna(np.median(x)))

# fill_na()\
# print(train.groupby("Neighborhood").LotFrontage.median())







