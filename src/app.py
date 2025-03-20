from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

datos = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")

datos.to_csv("data/raw/datos.csv", index = False)

print(f"La cantidad de nombres duplicados es {datos['name'].duplicated().sum()}")
print(f"La cantidad de Host IDs duplicados es {datos['host_id'].duplicated().sum()}")
print(f"La cantidad de IDs duplicados es {datos['id'].duplicated().sum()}")

datos.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)

fig, axis = plt.subplots(2, 3, figsize=(10, 7))

sns.histplot(ax = axis[0,0], data = datos, x = "host_id")
sns.histplot(ax = axis[0,1], data = datos, x = "neighbourhood_group").set_xticks([])
sns.histplot(ax = axis[0,2], data = datos, x = "neighbourhood").set_xticks([])
sns.histplot(ax = axis[1,0], data = datos, x = "room_type")
sns.histplot(ax = axis[1,1], data = datos, x = "availability_365")
fig.delaxes(axis[1, 2])

plt.tight_layout()

plt.show()

fig, axis = plt.subplots(4, 2, figsize = (10, 14), gridspec_kw = {"height_ratios": [6, 1, 6, 1]})

sns.histplot(ax = axis[0, 0], data = datos, x = "price")
sns.boxplot(ax = axis[1, 0], data = datos, x = "price")

sns.histplot(ax = axis[0, 1], data = datos, x = "minimum_nights").set_xlim(0, 200)
sns.boxplot(ax = axis[1, 1], data = datos, x = "minimum_nights")

sns.histplot(ax = axis[2, 0], data = datos, x = "number_of_reviews")
sns.boxplot(ax = axis[3, 0], data = datos, x = "number_of_reviews")

sns.histplot(ax = axis[2,1], data = datos, x = "calculated_host_listings_count")
sns.boxplot(ax = axis[3, 1], data = datos, x = "calculated_host_listings_count")

plt.tight_layout()

plt.show()

fig, axis = plt.subplots(4, 2, figsize = (10, 16))

sns.regplot(ax = axis[0, 0], data = datos, x = "minimum_nights", y = "price")
sns.heatmap(datos[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = datos, x = "number_of_reviews", y = "price").set(ylabel = None)
sns.heatmap(datos[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = datos, x = "calculated_host_listings_count", y = "price").set(ylabel = None)
sns.heatmap(datos[["price", "calculated_host_listings_count"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)
fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])

plt.tight_layout()

plt.show()

fig, axis = plt.subplots(figsize = (5, 4))

sns.countplot(data = datos, x = "room_type", hue = "neighbourhood_group")

plt.show()

datos["room_type"] = pd.factorize(datos["room_type"])[0]
datos["neighbourhood_group"] = pd.factorize(datos["neighbourhood_group"])[0]
datos["neighbourhood"] = pd.factorize(datos["neighbourhood"])[0]

fig, axes = plt.subplots(figsize=(15, 15))

sns.heatmap(datos[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",	
                        "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

sns.pairplot(data = datos)

datos.describe()

fig, axes = plt.subplots(3, 3, figsize = (15, 15))

sns.boxplot(ax = axes[0, 0], data = datos, y = "neighbourhood_group")
sns.boxplot(ax = axes[0, 1], data = datos, y = "price")
sns.boxplot(ax = axes[0, 2], data = datos, y = "minimum_nights")
sns.boxplot(ax = axes[1, 0], data = datos, y = "number_of_reviews")
sns.boxplot(ax = axes[1, 1], data = datos, y = "calculated_host_listings_count")
sns.boxplot(ax = axes[1, 2], data = datos, y = "availability_365")
sns.boxplot(ax = axes[2, 0], data = datos, y = "room_type")

plt.tight_layout()

plt.show()

estadisticas_precios = datos["price"].describe()

iqr_precio = estadisticas_precios["75%"] - estadisticas_precios["25%"]
mas_alto_iqr_precio = estadisticas_precios["75%"] + 1.5 * iqr_precio
mas_bajo_iqr_precio = estadisticas_precios["25%"] - 1.5 * iqr_precio

datos = datos[datos["price"] > 0]

cuenta_0_precio = datos[datos["price"] == 0].shape[0]
cuenta_1_precio = datos[datos["price"] == 1].shape[0]

estadisticas_noches = datos["minimum_nights"].describe()

iqr_noches = estadisticas_noches["75%"] - estadisticas_noches["25%"]

mas_alto_iqr_noches = estadisticas_noches["75%"] + 1.5 * iqr_noches
mas_bajo_iqr_noches = estadisticas_noches["25%"] - 1.5 * iqr_noches

datos = datos[datos["minimum_nights"] <= 15]

cuenta_0_noches_minimas = datos[datos["minimum_nights"] == 0].shape[0]
cuenta_1_noches_minimas = datos[datos["minimum_nights"] == 1].shape[0]
cuenta_2_noches_minimas = datos[datos["minimum_nights"] == 2].shape[0]
cuenta_3_noches_minimas = datos[datos["minimum_nights"] == 3].shape[0]
cuenta_4_noches_minimas = datos[datos["minimum_nights"] == 4].shape[0]

estadisticas_resumen = datos["number_of_reviews"].describe()

iqr_resumen = estadisticas_resumen["75%"] - estadisticas_resumen["25%"]

mas_alto_iqr_resumen = estadisticas_resumen["75%"] + 1.5 * iqr_resumen
mas_bajo_iqr_resumen = estadisticas_resumen["25%"] - 1.5 * iqr_resumen

estadisticas_hostlist = datos["calculated_host_listings_count"].describe()

iqr_hostlist = estadisticas_hostlist["75%"] - estadisticas_hostlist["25%"]

mas_alto_iqr_hostlist = estadisticas_hostlist["75%"] + 1.5 * iqr_hostlist
mas_bajo_iqr_hostlist = estadisticas_hostlist["25%"] - 1.5 * iqr_hostlist

cuenta_0_chlc = sum(1 for x in datos["calculated_host_listings_count"] if x in range(0, 5))
cuenta_1_chlc = datos[datos["calculated_host_listings_count"] == 1].shape[0]
cuenta_2_chlc = datos[datos["calculated_host_listings_count"] == 2].shape[0]

datos = datos[datos["calculated_host_listings_count"] > 4]

datos.isnull().sum().sort_values(ascending = False)

num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(datos[num_variables])
df_scal = pd.DataFrame(scal_features, index = datos.index, columns = num_variables)
df_scal["price"] = datos["price"]
df_scal.head()

X = df_scal.drop("price", axis = 1)
y = df_scal["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


selection_model = SelectKBest(chi2, k = 4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("data/processed/datos_limpios_train.csv", index = False)
X_test_sel.to_csv("data/processed/datos_limpios_test.csv", index = False)