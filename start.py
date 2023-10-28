#Implementing _____ model

#Author: Andrew Root

import numpy as np
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == '__main__':

    #Get the dataset
    x, y = load_wine(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

    models = [KNeighborsClassifier(3), DecisionTreeClassifier(criterion='entropy'), SVC()]

    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    
        print("Report for " + str(model) + ":\n")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred) + "\n\n")

    

    