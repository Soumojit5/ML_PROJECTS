# importing libraries
import numpy as np
import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
b_cancer = sklearn.datasets.load_breast_cancer()
# print(b_cancer)
x = b_cancer.data
y = b_cancer.target
# print(x.shape, y.shape)
model_data = pd.DataFrame(b_cancer.data, columns=b_cancer.feature_names)
model_data['Type'] = b_cancer.target
# print(model_data.head())
# print(model_data.corr())
# print(model_data['Type'].value_counts())
# print(b_cancer.target_names)
# print(model_data.groupby('Type').mean())
x_train,x_test,y_train,y_test =train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)
# stratify distributes data properly and random_state helps the user to pick the same samples for train and test and it doesn't change
# print(y.shape,y_test.shape,y_train.shape)
# print(y.mean(),y_test.mean(),y_train.mean())
# print(x.mean(),x_test.mean(),x_train.mean())
# model = LogisticRegression
model =KNeighborsClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
c_matrix = confusion_matrix(y_test,y_pred)
print(accuracy)
# print(c_matrix)


# prdiction
user_input = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
input_data = np.asarray(user_input)
# we convert the user input to numpy array and then reshape it as we are predicting for only one instance
reshaped_data = input_data.reshape(1,-1)
prediction = model.predict(reshaped_data)
if prediction[0] == 0:
    print("Your Breast Cancer is in MALIGNANT stage")
else:
    print("Your Breast Cancer is in BENIGN stage")