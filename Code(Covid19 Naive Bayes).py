import numpy as np
import pandas as pd

df=pd.read_csv("covid_19_clean_complete.csv")
df.drop(["Deaths","Recovered"],axis=1,inplace=True)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
df["DateLE"]=le.fit_transform(df["Date"].values)

df["Results"]=df["Confirmed"]!=0
df.replace(to_replace=True,value="Present",inplace=True)
df.replace(to_replace=False,value="Absent",inplace=True)


from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")

            
ax.scatter3D(df[df["Results"]=="Present"]["Lat"].values,df[df["Results"]=="Present"]["Long"].values,
             df[df["Results"]=="Present"]["DateLE"].values,cmap='hsv',c='red',label='Corona is Present')

ax.scatter3D(df[df["Results"]=="Absent"]["Lat"].values,df[df["Results"]=="Absent"]["Long"].values,
             df[df["Results"]=="Absent"]["DateLE"].values,cmap='hsv',c='blue',label='Corona is Absent')

ax.legend()
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Date')

plt.show()


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df.loc[:,["Lat","Long","DateLE"]].values,df["Results"].values,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test) 

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_test,y_pred)

print("Confusion Matrix \n",cm)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=10) 
print("Final Accuracy is :",accuracies.mean()*100)

