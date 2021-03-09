######################################
# IMPORT MODULES & LOAD DATA
######################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import warnings

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from helpers.data_prep import *
from helpers.eda import *
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv("datasets/house_prices/train.csv")
test = pd.read_csv("datasets/house_prices/test.csv")
df = train.append(test).reset_index(drop=True)

######################################
# EDA
######################################
check_df(df)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)

######################################
# KATEGORIK DEGISKEN ANALIZI
######################################
for col in cat_cols:
    cat_summary(df, col)

for col in cat_cols:
    stalk(df, col)

for col in cat_but_car:
    stalk(df, col)

for col in num_but_cat:
    stalk(df, col)

######################################
# SAYISAL DEGISKEN ANALIZI
######################################
df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

for col in num_cols:
    num_summary(df, col, plot=True)

######################################
# TARGET ANALIZI
######################################

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])

# target ile bagımsız degiskenlerin korelasyonları

low_corrs, high_corrs = find_correlation(df, num_cols)

# tüm değişkenler korelasyon
corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot = True, figsize=(20,15), fmt=".2f" )
plt.title("Correlation Between Features")
plt.show()


threshold = 0.60
filtre = np.abs(corr_matrix["SalePrice"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(df[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w/ Corr Threshold 0.75)")
plt.show()
######################################
# DATA PREPROCESSING & FEATURE ENGINEERING
######################################

######################################
# MISSING_VALUES
######################################
none_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',
             'GarageArea', 'GarageCars', 'MasVnrArea']

freq_cols = ['Exterior1st', 'Exterior2nd', 'KitchenQual']

for col in zero_cols:
    df[col].replace(np.nan, 0, inplace=True)

for col in none_cols:
    df[col].replace(np.nan, "None", inplace=True)

for col in freq_cols:
    df[col].replace(np.nan, df[col].mode()[0], inplace=True)

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))
df['LotFrontage'] = df.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

df.groupby("Exterior1st").agg({"SalePrice": ["count", "mean","median"]})
stalk(df, "Exterior1st", "SalePrice")

check_df(df)
df.columns

######################################
# OUTLIERS
######################################

replace_with_thresholds(df, "SalePrice")

######################################
# FEATURE ENGINEERING
######################################
df["MSSubClass"] = df["MSSubClass"].astype(str)
df["YrSold"] = df["YrSold"].astype(str)
df["MoSold"] = df["MoSold"].astype(str)

df.loc[(df["MSZoning"] == "RH"), "MSZoning"] = "RM"

df["LotShape"].value_counts()
df.loc[(df["LotShape"] == "IR2"), "LotShape"] = "IR1"
df.loc[(df["LotShape"] == "IR3"), "LotShape"] = "IR1"

df["LotShape"].value_counts()
df.loc[(df["LotConfig"] == "Corner"), "LotConfig"] = "FR2"
df.loc[(df["LotConfig"] == "Inside"), "LotConfig"] = "FR2"
df.loc[(df["LotConfig"] == "CulDSac"), "LotConfig"] = "FR3"

df.loc[(df["LandSlope"] == "Mod"),"LandSlope"] = "Sev"

df.loc[(df["Condition1"] == "Feedr"), "Condition1"] = "Artery"
df.loc[(df["Condition1"] == "RRAe"), "Condition1"] = "Artery"
df.loc[(df["Condition1"] == "RRAn"), "Condition1"] = "Norm"
df.loc[(df["Condition1"] == "PosN"), "Condition1"] = "PosA"
df.loc[(df["Condition1"] == "RRNe"), "Condition1"] = "PosA"
df.loc[(df["Condition1"] == "RRNn"), "Condition1"] = "PosA"

df.loc[(df["HouseStyle"] == "1.5Fin"), "HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "2.5Unf"), "HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "SFoyer"), "HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "SLvl"), "HouseStyle"] = "1Story"
df.loc[(df["HouseStyle"] == "2.5Fin"), "HouseStyle"] = "2Story"

### kontrol edelim
df.loc[(df["MasVnrType"] == "BrkCmn"), "MasVnrType"] = "None"

df.loc[(df["GarageType"] == "2Types"), "GarageType"] = "Attchd"
df.loc[(df["GarageType"] == "Basment"), "GarageType"] = "Attchd"

df.loc[(df["GarageType"] == "2Types"), "GarageType"] = "Attchd"
df.loc[(df["GarageType"] == "CarPort"), "GarageType"] = "Detchd"


# Derecelendirme içeren değişkenleri ordinal yapıya getirdim.
ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['ExterQual'] = df['ExterQual'].map(ext_map).astype('int')

ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['ExterCond'] = df['ExterCond'].map(ext_map).astype('int')

bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['BsmtQual'] = df['BsmtQual'].map(bsm_map).astype('int')
df['BsmtCond'] = df['BsmtCond'].map(bsm_map).astype('int')

bsmf_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmf_map).astype('int')
df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmf_map).astype('int')

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['HeatingQC'] = df['HeatingQC'].map(heat_map).astype('int')
df['KitchenQual'] = df['KitchenQual'].map(heat_map).astype('int')
df['GarageCond'] = df['GarageCond'].map(bsm_map).astype('int')
df['GarageQual'] = df['GarageQual'].map(bsm_map).astype('int')

# Toplam banyo sayısını gösteriyor
# df["NEW_TotalBath"] = df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5 + df['FullBath'] + df['HalfBath'] * 0.5
df["NEW_TotalBath"] = df['BsmtFullBath'] + df['FullBath'] + df['HalfBath'] * 0.5

# Toplam Kat Sayısı
df['TotalSF'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])

# 1. kat ve bodrum metrekare
df["NEW_SF"] = df["1stFlrSF"] + df["TotalBsmtSF"]

# toplam m^2
df["NEW_TOTAL_M^2"] = df["NEW_SF"] + df["2ndFlrSF"]

# Garaj alanı ve metrekarelerin toplamı
df["NEW_SF_G"] = df["NEW_SF"] + df["GarageArea"]

# Toplam Veranda Alanı
df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

# evin yaşı ve durumu ile ilgili bir değişken
df["NEW_BOMBA"] = df["YearRemodAdd"] + df["YearBuilt"]

# Evin yaşını kategoriye ayırdık
df["NEW_BOMBA_CAT"] = pd.qcut(df['NEW_BOMBA'], 5, labels=[1, 2, 3, 4, 5])


# kaliteleriyle ilgili değişkenler
df["NEW_QUAL_COND"] = df['OverallQual'] + df['OverallCond']
df["NEW_BSMT_QUAL_COND"] = df['BsmtQual'] + df['BsmtCond']
df["NEW_EX_QUAL_COND"] = df['ExterQual'] + df['ExterCond']
df["NEW_BSMT_QUAL_COND"] = df['GarageQual'] + df['BsmtCond']

df['NEW_QUAL_COND'].value_counts()
# iyi dereceli evlere FLAG
df['NEW_BEST'] = (df['NEW_QUAL_COND'] >= 14).astype('int')

# HAVUZLU OLAN EVLER
df["NEW_HAS_POOL"] = (df['PoolArea'] > 0).astype('int')

# LUX EVLER
df.loc[(df['Fireplaces'] > 0) & (df['GarageCars'] >= 3), "NEW_LUX"] = 1
df["NEW_LUX"].fillna(0, inplace=True)
df["NEW_LUX"] = df["NEW_LUX"].astype(int)

# toplam alan
df["NEW_AREA"] = df["GrLivArea"] + df["GarageArea"]
# df.groupby("MiscVal").agg({"SalePrice": ["count", "mean","median"]})

# m^2/oda
# df["NEW_TOTAL_ROOM"] = df["BedroomAbvGr"] + df["KitchenAbvGr"] + df["TotRmsAbvGrd"]
df["NEW_ROOM_AREA"] = df["NEW_TOTAL_M^2"] / df["TotRmsAbvGrd"]

df.loc[(df['TotRmsAbvGrd'] >= 7) & (df['GrLivArea'] >= 1800), "NEW_TOTAL_GR"] = 1


# Create Cluster / ekrem hoca kaggle
ngb = df.groupby("Neighborhood").SalePrice.mean().reset_index()
ngb["NEW_CLUSTER_NEIGHBORHOOD"] = pd.cut(df.groupby("Neighborhood").SalePrice.mean().values, 4, labels=range(1, 5))
df = pd.merge(df, ngb.drop(["SalePrice"], axis=1), how="left", on="Neighborhood")

df["NEW_GARAGEBLTAGE"] = df.GarageYrBlt - df.YearBuilt

######################################
# RARE ENCODING
######################################
rare_analyser(df, "SalePrice", 0.01)
df = rare_encoder(df, 0.01)

drop_list = ["Street", "SaleCondition", "Functional", "Condition2", "Utilities", "SaleType", "MiscVal",
             "Alley", "LandSlope", "PoolQC", "MiscFeature", "Electrical", "Fence", "RoofStyle", "RoofMatl",
             "FireplaceQu"]

# df.groupby("MiscVal").agg({"SalePrice": ["count", "mean","median"]})

cat_cols = [col for col in cat_cols if col not in drop_list]

for col in drop_list:
    df.drop(col, axis=1, inplace=True)


######################################
# ONE-HOT ENCODING
######################################
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)
cat_cols = cat_cols + cat_but_car
df = one_hot_encoder(df, cat_cols, drop_first=True)


#######################################
# Final Model
#######################################
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

X = train_df.drop(['SalePrice', "Id"], axis=1)
y = np.log1p(train_df['SalePrice'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=46)
lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.001, 0.01, 0.05, 0.1],
               "n_estimators": [200, 500, 750],
               "max_depth": [-1, 2, 5, 8],
               "colsample_bytree": [1, 0.50, 0.75],
               "num_leaves": [25, 31, 44]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y_pred = lgbm_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))

#######################################
# Feature Importance
#######################################
def plot_importance(model, df, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': df.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('df')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_tuned, X_train, 50, True)

#######################################
# SONUCLARIN YUKLENMESI
#######################################

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"]

y_pred_sub = lgbm_tuned.predict(test_df.drop("Id", axis=1))
y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub

submission_df.head()

submission_df.to_csv('submission_rf.csv', index=False)
