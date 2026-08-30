import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.conftest import dropna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

df = pd.read_csv(url)

df_class=df.copy()
df_class = df_class.dropna()

df_class["HighPrice"] = (
    df_class["median_house_value"] >
    df_class["median_house_value"].median()
).astype(int)

X = df_class.drop(["median_house_value", "HighPrice"], axis=1)
X= pd.get_dummies(X)
y = df_class["HighPrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,10))

sns.heatmap(
    cm,
    annot=True,
    fmt = "d",
)

plt.xlabel("Classe Predita")
plt.ylabel("Classe")

plt.title("Matriz de Confusão")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)

from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)

from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


X_cluster = df[['median_income', 'housing_median_age', 'total_rooms']]


scaler = StandardScaler()
X_cluster = scaler.fit_transform(X_cluster)
X_cluster = pd.DataFrame(X_cluster, columns=['median_income', 'housing_median_age', 'total_rooms'])

wss = []

K = range(1,10)

for k in K:

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_cluster)

    wss.append(model.inertia_)

plt.plot(K, wss, marker='o')

plt.xlabel("Número de clusters")
plt.ylabel("WSS")

plt.title("Elbow Method")

plt.show()