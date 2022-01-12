import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

data = pd.read_csv("datasets/diabetes.csv")
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

# Nadir görülen değerleri analiz etme.
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

# Nadir görülen değerleri encode etme.
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

# One hot encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

###########################################################################################
# Adım 1 : Eksik Değerler
# Kolonlara göre 0 içeren satır sayısı
for col in df.columns:
    print(col + " : " + str((df[f"{col}"] == 0).sum()))

df[["GLUCOSE", "BLOODPRESSURE", "SKINTHICKNESS", "INSULIN", "BMI"]] = df[
    ["GLUCOSE", "BLOODPRESSURE", "SKINTHICKNESS", "INSULIN", "BMI"]].replace(0, np.nan)

# NULL değerleri kontrol ettik.
df.isnull().sum()

# Kategorik, numerik ve kardinal değişkenleri belirledik.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Scale etmemiz için dummies değişkenine dönüştürdük.
dff = pd.get_dummies(df[num_cols], drop_first=True)
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

# KNN ile belirsiz değişkenleri tahmin ettik.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=33)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

# dataframe'e tekrar çevirdik.
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff["OUTCOME"] = df["OUTCOME"]
dff.head()
df = dff.copy()
df.head()
##########################################################################################
##########################################################################################
# Adım 2 : Aykırı Değerler :
# Aykırı değer kontrolü yapılır.
for col in num_cols:
    print(col, check_outlier(df, col))

# Aykırı değerler  baskılanır
for col in num_cols:
    replace_with_thresholds(df, col)

# Aykırı değer var mı tekrar kontrol ettik.
for col in num_cols:
    print(col, check_outlier(df, col))

##########################################################################################
##########################################################################################
# Step 3 : Feature Engineering :
# Yeni Feature oluşturmak için bilgi edinmemiz gerekiyor.

def sns_heatmap(dataset, color):
    heatmap = sns.heatmap(dataset.corr(), vmin=-1, vmax=1, annot=True, cmap=color)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.show()

sns_heatmap(df,color='Greens')
# OUTCOME'ı etkileyen en önemli değişkenlerin sırasıyla GLUCOSE, INSULIN, BMI, SKINTHICKNESS, AGE, PREGNANCIES, BLOODPRESSURE, FUNCTION
# Şimdi bu değişkenleri ayrıntılı bir şekilde inceleyelim.
def pairplot(dataset, target_column):
    sns.set(style="ticks")
    sns.pairplot(dataset, hue=target_column)
    plt.show()

pairplot(df, 'OUTCOME')

# GLUCOSE-BMI ARASINDA bir ilişki var gibi grafikten yorumlarsak.
df["NEW_GLUCOSE_BMI"] = df["GLUCOSE"] * df["BMI"]
# GLUCOSE-INSULIN ARASINDA bir ilişki var gibi grafikten yorumlarsak.
df["NEW_GLUCOSE_INSULIN"] = df["INSULIN"] / df["GLUCOSE"]
# GLUCOSE değerlerini ayırdık
df.loc[(df["GLUCOSE"]) < 140, 'NEW_GLUCOSE_GROUP'] = 'normal'
df.loc[(df["GLUCOSE"] >= 140) & (df["GLUCOSE"] < 200), 'NEW_GLUCOSE_GROUP'] = 'little_high'

# AGE aralıklarını belirleidk.
df.loc[(df["AGE"]) <= 35.0, 'NEW_AGE_GROUP'] = 'young'
df.loc[(df["AGE"] > 35.0) & (df["AGE"] <= 45.0), 'NEW_AGE_GROUP'] = 'middle'
df.loc[(df["AGE"] > 45.0) & (df["AGE"] <= 55.0), 'NEW_AGE_GROUP'] = 'mature'
df.loc[(df["AGE"] > 55.0), 'NEW_AGE_GROUP'] = 'old'

# BMI'ları belirledik.
df.loc[(df["BMI"] <= 18.4), 'NEW_BODY_MASS'] = 'underweight'
df.loc[(df["BMI"] > 18.4) & (df["BMI"] <= 24.9), 'NEW_BODY_MASS'] = 'normal_weight'
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), 'NEW_BODY_MASS'] = 'over_weight'
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), 'NEW_BODY_MASS'] = 'obesity_class+'
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 44.9), 'NEW_BODY_MASS'] = 'obesity_class++'
df.loc[(df["BMI"] > 44.9), 'NEW_BODY_MASS'] = 'obesity_class+++'

# Normal insulin değerleri.
def set_insulin(value):
    if value["INSULIN"] >= 100 and value["INSULIN"] <= 126:
        return "Normal"
    else:
        return "Abnormal"
df = df.assign(NEW_INSULIN_SCORE=df.apply(set_insulin, axis=1))

# Yeni oluşturduğumuz değişkenleri tekrar ayırdık. Kategorik - Numerik - Kardinal olarak.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Değişkenlerde sadece 2 farklı string değer olan değişkenleri tespit edelim.
df.head()
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

# Bu değişkenleri 0-1 e çeviriyoruz.
for col in binary_cols:
    label_encoder(df, col)


# Nadir görülen değerlerin oranı 0.01'den az ise bunları atıyoruz.
rare_analyser(df, "OUTCOME", cat_cols)
df = rare_encoder(df, 0.01)

# String içeren kategorik değişkenleri seçelim
df.head()
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
# one-hot encode ettik bu değerleri.
df = one_hot_encoder(df, ohe_cols)

# sonra tekrar değişkenleri türlerine göre ayırıyoruz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Yine nadir görülen kategorik değişkenlerdeki değerlere bakıyoruz.
rare_analyser(df, "OUTCOME", cat_cols)
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
df.drop(useless_cols, axis=1, inplace=True)

# RobustScaler yapıyoruz ki modeli kurabilelim diye.
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.head()

# Hedef değişkeni ve bağımsız değişkenleri ayırıyoruz.4
y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

# Test ve Train seti olarak ikiye ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

# Modeli kuralım ve accuracy_score'a ulaşalım.
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=11).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(accuracy_score(y_pred, y_test))


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

