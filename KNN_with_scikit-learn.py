#libraries import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Read the data on which we will do the study
dataset = pd.read_csv('C:/Users/ferna/PycharmProjects/Project_Python_OD_FL/gamingset_purchase_records.csv')
dataset.head()

dataset = dataset.drop('Gender', axis=1) # since the column of the gende is not needed, we remove it in other also to have a simplest case
dataset.head()

X = dataset.drop('Purchase Gaming set', axis=1) # Our x are now the columns age and salary
y = dataset['Purchase Gaming set'] # and our y are the column purchase, wether or not the person bought it or not

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .3, random_state= 42)#test size 30%, which means 70% of the data are for the algorithm to learn
                                                                                            # and the 30% it is to test if what was learnt is actually working
                                                                                            #random sate in other to split the data randomly

k_range = range (1,15)
scores= {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k) #defining an object from KneighborsClassifier
    knn.fit (X_train, y_train)  #Fit the k-nearest neighbors classifier from the training dataset.
    y_pred = knn.predict(X_test) #Predict the class labels for the provided data, X_test.
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))   ##accuracy test in order to determine the optimal k_value, for which accuracy is high


plt.plot(k_range, scores_list, color='blue', marker='x', linestyle='dashed')
plt.xlabel('Value of K for KNN')
plt.ylabel('testing accuracy')
plt.show()

knn1 = KNeighborsClassifier(n_neighbors=5)  #after the compilation, we can see that for k = 5, accuracy is the highest
knn1.fit(X,y)

X_new =[[80, 45], [0, 150000]]
y_predictions = knn1.predict(X_new)

#classify the points defined above
if (y_predictions[0] == 0):
    print ("did not buy")
else:
    print ("bought")

if (y_predictions[1] == 0):
    print ("did not buy")
else:
    print ("bought")



