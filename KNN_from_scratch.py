import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scipy.stats import mode

#Read the data on which we will do the study
dataset = pd.read_csv('C:/Users/ferna/PycharmProjects/Project_Python_OD_FL/gamingset_purchase_records.csv')
dataset.head()

dataset = dataset.drop('Gender', axis=1) # since the column of the gende is not needed, we remove it in other also to have a simplest case
dataset.head()

X = dataset.drop('Purchase Gaming set', axis=1)# Our x are now the columns age and salary
Y = dataset['Purchase Gaming set'] # and our y are the column purchase, wether or not the person bought it or not

print(X)
print(Y)


def euclidean_distance(pt1, pt2):   #function to calculate the eucludian distance
    distance = np.sqrt(np.sum(pt1 - pt2) ** 2)
    return distance

#a small test to verify the eucludian distance
a = np.array([4, 4])
b = np.array([2, 7])

print(euclidean_distance(a, b))

# the main function knn
def KNN(X_train, X_test, Y_train, Y_test, k_val):
    y_values = []
    for test_pt in X_test.to_numpy():  #in order to iterate through every tester data to know their class
        distances = []  # in that list we will store all the distances from the test point
        for i in range(len(X_train)):
            distances.append(euclidean_distance(np.array(X_train.iloc[i]), test_pt))  # calculate the distance from the test point to the training point and stock it in the list
                                                                                      # .iloc - sum indices

        distance_data = pd.DataFrame(data=distances, columns=['distance'], index=Y_train.index) #dataframe in other to changes our 'distances' into a usable spreadsheet of usable data

        k_neighbors_list = distance_data.sort_values(by=['distance'], axis=0)[:k_val] #sort our data from 0 to k_val closest neighbors

        labels = Y_train.loc[k_neighbors_list.index] #look at the y values to these nearest neighbors

        mostCommon = mode(labels).mode[0]  #mode in order to find the y, the label with the most occurencies

        y_values.append(mostCommon) # our prediction, in our case it's 0 or 1
    return y_values

#use of the function
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=42)#test size 30%, which means 70% of the data are for the algorithm to learn
                                                                                            # and the 30% it is to test if what was learnt is actually working
                                                                                            #random sate in other to split the data randomly
y_values_test = KNN(X_train, X_test, Y_train, Y_test, k_val=5) #outputs, to set always the k_value to an odd number in orther to avoid tie situation during the classification

#accuracy test in order to determine the optimal k_value, for which accuracy is high
accuracy_vals = []
for i in range(1, 15):
    y_values_test = KNN(X_train, X_test, Y_train, Y_test, k_val=i) # outputs for every case,each k_val until 14
    accuracy_vals.append(accuracy_score(Y_test, y_values_test)) #accuracy between 0 and 1
plt.plot(range(1, 15), accuracy_vals, color='blue', marker='x', linestyle='dashed')
plt.figure()


y_values_test = KNN(X_train, X_test, Y_train, Y_test, k_val=5)  #after compilation, we know that k_val = 5(has the highest accuracy)
print(accuracy_score(Y_test, y_values_test))  #to visualise the value of the highest accuracy

for i in range(len(y_values_test)):
    if (y_values_test[i] == 0):  #class of people who did not buy the gaming set
        plt.scatter(X_test.iloc[i]['Age'], X_test.iloc[i]['Salary'], color='blue')
    if (y_values_test[i] == 1):  #class of people who bought the gaming set
        plt.scatter(X_test.iloc[i]['Age'], X_test.iloc[i]['Salary'], color='red')


plt.xlabel('Age')
plt.ylabel('Salary')


#classification of two points (80,0) and (45,150000)
dataset = pd.read_csv('C:/Users/ferna/Downloads/Feuille de calcul sans titre - Feuille 1.csv')
dataset.head()
dataset = dataset.drop('Gender', axis=1) # since the column of the gende is not needed, we remove it in other also to have a simplest case
dataset.head()
X_new = dataset
Y_new = []

y_values_test1 = KNN(X_train, X_new, Y_train, Y_test, k_val=5)
for i in range(len(y_values_test1)):
    if (y_values_test1[i] == 0):  #class of people who did not buy the gaming set
        print("did not buy")
    if (y_values_test1[i] == 1):  #class of people who bought the gaming set
        print("bought")

plt.show()
