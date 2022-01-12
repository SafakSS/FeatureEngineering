import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

data = pd.read_csv("datasets/Telco-Customer-Churn.csv")

df = data.copy()
df.columns = [col.upper() for col in df.columns]

# Aykırı Değerler için fonksiyonumuz:
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit # a

# kategorik, numerik, kardinal değişkenleri belirleme.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.


    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]  # kategorik değişkenler seçilir.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]  # numerik ama kategorik olanlar seçilir.
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]  # kategorik ama kardinal olanlar seçilir.
    cat_cols = cat_cols + num_but_cat  # kategorik ve numerik ama kategorik değişkenler toplanır.
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # ve kategorik değişkenlerdeği değerler kategorik ama kardinal değişkeninde yoksa kategorik değişkene tekrar ata.

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]  # numerik değişkenler.
    num_cols = [col for col in num_cols if col not in num_but_cat]
    # ve numerik değişkenlerdeki değerler numerik ama kategorik değişkenlenin içindeki değerlerde yoksa numerik değişkenine tekrar ata.

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

# aykırı değer var mı?
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        # üst limit değerinden daha büyük değer varsa veya alt limit değerinden daha küçük değer varsa True döndür.
        return True
    else:
        return False

# aykırı değeri baskılama
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# label encoder
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
######################################################################################
# Adım 1 : Keşifsel Veri Analizi

def check_df(dataframe):
    print("################################## Shape #####################################")
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")
    print("################################## Types##################################")
    print(dataframe.dtypes)
    print("################################## NA ##################################")
    print(dataframe.isnull().sum())
    print("################################## Quantiles ##################################")
    print(dataframe.quantile([0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1]).T)
    print("################################## Head ##################################")
    print(dataframe.head())
    print("################################## Tail ##################################")
    print(dataframe.tail())
check_df(df)


#########################################################################################
# Null değişkenimiz yok o yüzden 0 içeren değişkenleri kontrol ediyoruz.
for col in df.columns:
    print(col + " : " + str((df[f"{col}"] == 0).sum()))

# Tenure değişkeninde 0 değerler tespit ettik bunlar daha ilk ayını doldurmamışlar o yüzden df'den elendiler.
df = df[df["TENURE"] != 0]

# TOTALCHARGES değişkeni object gözüküyordu. Float yaptık.
df['TOTALCHARGES'] = df['TOTALCHARGES'].astype(float)

# CUSTOMER ID gereksiz.
del df["CUSTOMERID"]

# Değişkenlerin tiplerini belirleyelim.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    if check_outlier(df, col) == True:
        replace_with_thresholds(df, col)


# Çok değişkenli aykırılık analizi.
dff = df.select_dtypes(include=['float64', 'int64'])
clf = LocalOutlierFactor(n_neighbors=25)
clf.fit_predict(dff)

dff_scores = clf.negative_outlier_factor_
scores = pd.DataFrame(np.sort(dff_scores))
scores.plot(stacked=True, xlim=[0, 60], style='.-')
# plt.show()
th = np.sort(dff_scores)[35]
clf_index = df[dff_scores < th].index
df.drop(index=clf_index, inplace=True)
########################################################################################################

# Feature Engineering
# Daha doğru değişken üretmemiz için biraz daha bilgi edinelim.
def sns_heatmap(dataset, color):
    heatmap = sns.heatmap(dataset.corr(), vmin=-1, vmax=1, annot=True, cmap=color)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.show()
sns_heatmap(df,color='Greens')


def pairplot(dataset, target_column):
    sns.set(style="ticks")
    sns.pairplot(dataset, hue=target_column)
    plt.show()
pairplot(df, 'CHURN')

bins = [0, 3, 6, 12, 24, 36, 60, 72]
labels = ['1-3', '4-6', '7-12', '13-24', '25-36', '37-60', '61-72']
df["NEW_TENURE"] = pd.cut(df['TENURE'], bins=bins)

df["__TENURE"] = df["TENURE"] * df["TOTALCHARGES"]
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Label Encoder()
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col, )

# Rare Encoder()
rare_analyser(df, "CHURN", cat_cols)
# df = rare_encoder(df, 0.01)

# One-Hot Encoding()
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.head()

y = df["CHURN"]
X = df.drop(["CHURN"], axis=1)

# Test ve Train seti olarak ikiye ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

# Modeli kuralım ve accuracy_score'a ulaşalım.
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=16).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print((accuracy_score(y_pred, y_test)))


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)

