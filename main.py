import yfinance as yf
import numpy as np
from numpy import array
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

msft = yf.Ticker("MSFT")

hist = msft.history(period="max")
dates = hist.index.strftime("%Y-%m-%d").tolist()

opens = hist["Open"].values.tolist()
closes = hist["Close"].values.tolist()
highs = hist["High"].values.tolist()
lows = hist["Low"].values.tolist()
volumes = hist["Volume"].values.tolist()

features = [opens, closes, highs, lows, volumes]

for i in features:  # Deleting the last day to prevent mid-day calling
    i.pop()

classifiers1 = []
classifiers2 = []
for i in range(1, len(features[1])):
    diff = closes[i] - opens[i]
    classifiers2.append(diff)
    if diff < 0:
        classifiers1.append(0)
    if diff >= 0:
        classifiers1.append(1)

y = []
for i in closes:
    y.append(i)
for i in range(365):
    y.pop(0)

# for j in range(365):
#     for i in features:  # Removing another day because next day's difference is unknown
#         i.pop()

X = np.array(features).T
y = array(y)
X = X * 100
y = y * 100
X = X.astype(int)
y = y.astype(int)

X_train = X[:len(X) - 365]
X_test = X[len(X) - 365:]
y_train = y

# split_index = (len(X)//5) * 4
# X_train = X[:split_index]
# X_test = X[split_index:]
# y_train = y[:split_index]
# y_test = y[split_index:]
print("Split Complete")

# rfc = RandomForestClassifier(n_estimators=200, class_weight="balanced")
# rfc.fit(X_train, y_train)
# pred_rfc = rfc.predict(X_test)
#
# etc = ExtraTreesClassifier(n_estimators=200, class_weight="balanced")
# etc.fit(X_train, y_train)
# pred_etc = etc.predict(X_test)

# dtc = DecisionTreeClassifier(class_weight="balanced")
# dtc.fit(X_train, y_train)
# pred_dtc = dtc.predict(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print("Logistic Regression Complete")
#
# svc = SVC(class_weight="balanced")
# svc.fit(X_train, y_train)
# pred_svc = svc.predict(X_test)
# print("SVC Complete")
#
# lsvc = LinearSVC(class_weight="balanced", dual=False)
# lsvc.fit(X_train, y_train)
# pred_lsvc = lsvc.predict(X_test)
# print("Linear SVC Complete")

# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# pred_knn = knn.predict(X_test)
# print("K-Nearest Neighbors Complete")

# print(pred_lr)
x_values = []
# for i in range(split_index, len(X)):
#     x_values.append(i)
for i in range(len(X) - 365, len(X)):
    x_values.append(i)

# plt.plot(dates[:len(dates) - 365 - 1], y_train.tolist() + y_test.tolist())
# plt.plot(dates[split_index + 365 + 1:], pred_lr, color="red")

# plt.plot(y_train.tolist() + y_test.tolist())
# plt.plot(x_values, pred_lr, color="red")
# plt.show()

plt.plot(y_train)
plt.plot(x_values, pred_lr, color="red")
plt.show()

# print("RFC")
# print(classification_report(y_test, pred_rfc, zero_division=1))
# print("ETC")
# print(classification_report(y_test, pred_etc, zero_division=1))
# print("DTC")
# print(classification_report(y_test, pred_dtc, zero_division=1))
# print("LR")
# print(classification_report(y_test, pred_lr, zero_division=1))
# print("SVC")
# print(classification_report(y_test, pred_svc, zero_division=1))
# print("LSVC")
# print(classification_report(y_test, pred_lsvc, zero_division=1))
# print("KNN")
# print(classification_report(y_test, pred_knn, zero_division=1))
