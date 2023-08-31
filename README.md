# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Praveen D
RegisterNumber: 212222240076
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]) 
*/
```

## Output:

![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/ac26ca89-3b37-4384-8696-ce5e81f555c6)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/ba97748b-9f05-4134-802d-9f0f37f2a78a)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/8243014d-5440-4472-86da-466d6731e71f)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/5a22e62f-6c2e-4388-8d49-a91cdb96f98d)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/7f30e925-0474-46e8-9234-127aec1c87ae)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/9e8a7c47-b671-497c-8e36-779a54f8a131)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/c4719581-e775-47c6-b8d1-ac01552e23cb)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/65010508-166f-42b5-9cad-a901f3795e37)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/1990880f-fd8d-4f32-a936-03e9b1186116)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/21883bd0-3eea-4951-9e50-4099e36f685e)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/47512832-8f38-4161-a7ba-ceb62f26dcce)
![image](https://github.com/praveenmax55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497509/b81386d7-2e2d-4903-b2ab-129bd1746883)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
