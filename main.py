import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

df = pd.read_csv('creditcard.csv')
x = df.drop(df.columns[-1], axis=1)
y = df.iloc[:, -1]
y.value_counts().plot(kind='barh')
plt.xlabel('Counts')
plt.ylabel('Class')
plt.show()

train_ratio = 0.70
val_ratio = 0.10
test_ratio = 0.20

'''----------------------SUPERVISED LEARNING PART-----------------------'''

# Resamples unbalanced data without replacement to get the number of cases to be close to equal
x_oversampled, y_oversampled = resample(x[y == 1], y[y == 1], replace=True, n_samples=x[y == 0].shape[0],
                                        random_state=123)

x2 = x.to_numpy()
y2 = y.to_numpy()

x_balanced = np.vstack((x2, x_oversampled))
y_balanced = np.hstack((y2, y_oversampled))

uni, freq = np.unique(y_balanced, return_counts=True)
plt.bar(uni, freq)
plt.xlabel('Class')
plt.ylabel('Counts')
# plt.yscale('log')
plt.show()
print(uni, freq)

# Makes train 70% of the balanced dataset
x_train, x_test, y_train, y_test = train_test_split(x_balanced, y_balanced, test_size=(1 - train_ratio))

# Makes val and test 10% and 20% respectively of the balanced dataset
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=(test_ratio / (test_ratio + val_ratio)))

# uni, freq = np.unique(y_train, return_counts=True)
# print(uni, freq)

LR = LogisticRegression(solver='lbfgs', max_iter=1000).fit(x_train, y_train)
pred = LR.predict(x_test)
print(classification_report(y_test, pred))

'''--------------------END OF SUPERVISED LEARNING--------------------'''

'''--------------------UNSUPERVISED LEARNING PART--------------------'''

# Makes train 70% of the original dataset
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=(1 - train_ratio))

# Makes val and test 10% and 20% respectively of the original dataset
x_val2, x_test2, y_val2, y_test2 = train_test_split(x_test, y_test, test_size=(test_ratio / (test_ratio + val_ratio)))

IF = IsolationForest(n_estimators=100).fit(x_train2)
pred2 = IF.predict(x_test2)

# Makes the predicted labels 0 or 1 to match with the acutal labels for comparison purposes
for i in range(len(pred2)):
    if pred2[i] == 1:
        pred2[i] = 0
    else:
        pred2[i] = 1
print(pred2)
print(classification_report(y_test2, pred2))

'''--------------------END OF UNSUPERVISED LEARNING--------------------------'''
