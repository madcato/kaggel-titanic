# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# https://www.kaggle.com/nandhuelan/titanic-diving-deep-into-atlantic-ocean

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib osx
# from IPython import get_ipython
# ipy = get_ipython()
# if ipy is not None:
#     ipy.run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./data"))



df=pd.read_csv('./data/train.csv')
testData=pd.read_csv('./data/test.csv')
print(df.head(5))


df.info()


print(df.isnull().sum())




f,ax=plt.subplots(1,2,figsize=(18,8))
df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
plt.tight_layout()
sns.countplot('Survived',data=df,ax=ax[1],palette="Set2")
ax[1].set_title('Survived')

df.groupby(['Sex','Survived'])['Survived'].count()



f,ax=plt.subplots(1,2,figsize=(18,8))
df[['Sex','Survived']].groupby(['Sex']).count().plot.bar(ax=ax[0])
ax[0].set_title('Sex vs Survived')
sns.countplot('Sex',hue='Survived',data=df,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')



pd.crosstab(df.Pclass,df.Survived).style.background_gradient(cmap='PuBu')



sns.factorplot('Pclass','Survived',hue='Sex',data=df)


print('Oldest passenger age in the ship',df.Age.max())
print('Youngest passenger age  in the ship',df.Age.min())
print('Average passenger  age in the ship',df.Age.mean())


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=df,split=True,ax=ax[0],palette=[ "#34495e", "#2ecc71"])
ax[0].set_title('Pclass vs Age')
ax[0].set_yticks(range(0,100,10))
sns.violinplot("Sex","Age", hue="Survived", data=df,split=True,ax=ax[1],palette=["#95a5a6", "#e74c3c"])
ax[1].set_title('Sex vs Age')
ax[1].set_yticks(range(0,100,10))


df['Initial']=0
for i in df:
    df['Initial']=df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
    
    testData['Initial']=0
for i in testData:
    testData['Initial']=testData.Name.str.extract('([A-Za-z]+)\.')
    
pd.crosstab(df.Sex,df.Initial).style.background_gradient(cmap='coolwarm')



pd.crosstab(testData.Sex,testData.Initial).style.background_gradient(cmap='coolwarm')






df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
testData['Initial'].replace(['Col','Dona','Dr','Rev','Ms'],['Mr','Mrs','Mr','Other','Miss'],inplace=True)



df.groupby('Initial')['Age'].mean() #lets check the average age by Initials


testData.groupby('Initial')['Age'].mean()



## Assigning the NaN Values with the Ceil values of the mean ages
df.loc[(df.Age.isnull())&(df.Initial=='Mr'),'Age']=33
df.loc[(df.Age.isnull())&(df.Initial=='Mrs'),'Age']=36
df.loc[(df.Age.isnull())&(df.Initial=='Master'),'Age']=5
df.loc[(df.Age.isnull())&(df.Initial=='Miss'),'Age']=22
df.loc[(df.Age.isnull())&(df.Initial=='Other'),'Age']=46
testData.loc[(testData.Age.isnull())&(testData.Initial=='Mr'),'Age']=33
testData.loc[(testData.Age.isnull())&(testData.Initial=='Mrs'),'Age']=39
testData.loc[(testData.Age.isnull())&(testData.Initial=='Master'),'Age']=7
testData.loc[(testData.Age.isnull())&(testData.Initial=='Miss'),'Age']=22
testData.loc[(testData.Age.isnull())&(testData.Initial=='Other'),'Age']=36


print(df.Age.isnull().sum())

print(testData.Age.isnull().sum())

f,ax=plt.subplots(1,2,figsize=(20,10))
df[df['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black')
ax[0].set_title('Not Survived')
df[df['Survived']==1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black')



sns.factorplot('Pclass','Survived',col='Initial',data=df)


pd.crosstab([df.Embarked,df.Pclass],[df.Sex,df.Survived],margins=True).style.background_gradient(cmap='Paired')



f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=df,ax=ax[0,0],palette=["#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=df,ax=ax[0,1],palette=["#9b59b6", "#3498db", "#95a5a6"])
ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=df,ax=ax[1,0],palette=["#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=df,ax=ax[1,1],palette=["#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
ax[1,1].set_title('Embarked vs Pclass')



df['Embarked'].fillna('S',inplace=True)
testData['Embarked'].fillna('S',inplace=True)


df.Embarked.isnull().sum()


pd.crosstab(index=df['SibSp'],columns=df['Survived']).style.background_gradient(cmap='Paired')




f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived',data=df,ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp','Survived',data=df,ax=ax[1],kind='violin')
ax[1].set_title('SibSp vs Survived')
plt.close(2)


pd.crosstab(df.Parch,df.Survived).style.background_gradient(cmap='Accent')


f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('Parch','Survived',data=df,ax=ax[0])
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch','Survived',data=df,ax=ax[1])
ax[1].set_title('Parch vs Survived')
plt.close(2)



print('Highest Fare',df.Fare.max())
print('Lowest Fare',df.Fare.min())
print('Average Fare',df.Fare.mean())





f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(df[df['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(df[df['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(df[df['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')



sns.heatmap(df.corr(),annot=True)
fig=plt.gcf()
fig.set_size_inches(10,8)



df['Age_band']=0
df.loc[df['Age']<=16,'Age_band']=0
df.loc[(df['Age']>16)&(df['Age']<=32),'Age_band']=1
df.loc[(df['Age']>32)&(df['Age']<=48),'Age_band']=2
df.loc[(df['Age']>48)&(df['Age']<=64),'Age_band']=3
df.loc[df['Age']>64,'Age_band']=4
df.head(2)





testData['Age_band']=0
testData.loc[testData['Age']<=16,'Age_band']=0
testData.loc[(testData['Age']>16)&(testData['Age']<=32),'Age_band']=1
testData.loc[(testData['Age']>32)&(testData['Age']<=48),'Age_band']=2
testData.loc[(testData['Age']>48)&(testData['Age']<=64),'Age_band']=3
testData.loc[testData['Age']>64,'Age_band']=4



sns.factorplot('Age_band','Survived',data=df,col='Pclass')


df['Family_Size']=0
df['Alone']=0
df['Family_Size']=df['Parch']+df['SibSp']#family size
df.loc[df.Family_Size==0,'Alone']=1#Alone

testData['Family_Size']=0
testData['Alone']=0
testData['Family_Size']=testData['Parch']+testData['SibSp']#family size
testData.loc[testData.Family_Size==0,'Alone']=1#Alone

f,ax=plt.subplots(1,2,figsize=(18,6))
sns.factorplot('Family_Size','Survived',data=df,ax=ax[0])
ax[0].set_title('Family_Size vs Survived')
sns.factorplot('Alone','Survived',data=df,ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)


df['Fare_Range']=pd.qcut(df['Fare'],4)
df.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


df['Fare_category']=0
df.loc[df['Fare']<=7.91,'Fare_category']=0
df.loc[(df['Fare']>7.91)&(df['Fare']<=14.454),'Fare_category']=1
df.loc[(df['Fare']>14.454)&(df['Fare']<=31),'Fare_category']=2
df.loc[(df['Fare']>31)&(df['Fare']<=513),'Fare_category']=3


sns.factorplot(x='Fare_category',y='Survived',data=df,hue='Sex',palette=["#34495e", "#2ecc71"])



df['Sex']=df['Sex'].apply(lambda x : 0 if x=='male' else 1)
df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
df['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


testData['Sex']=testData['Sex'].apply(lambda x : 0 if x=='male' else 1)
testData['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
testData['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)



df.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)



df.drop('SibSp',axis=1,inplace=True)
testData.drop('SibSp',axis=1,inplace=True)


df.drop('Fare_category',axis=1,inplace=True)



#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


train,test=train_test_split(df,test_size=0.3,random_state=0,stratify=df['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
testData.set_index('PassengerId',inplace=True)
testData.drop(['Name','Ticket','Cabin','Age','Fare'],inplace=True,axis=1)
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]



model=svm.SVC(C=1,gamma=0.1)
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
testprediction=model.predict(testData)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))


model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))


model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))


model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_Y))


model=GaussianNB()
model.fit(train_X,train_Y)
prediction6=model.predict(test_X)
print('The accuracy of the NaiveBayes is',metrics.accuracy_score(prediction6,test_Y))


model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
prediction7=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction7,test_Y))


model=KNeighborsClassifier() 
model.fit(train_X,train_Y)
prediction5=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_Y))


a_index=list(range(1,11))
a=pd.Series()
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
plt.plot(a_index, a)
plt.xticks(a_index)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())




from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction

X=df[df.columns[1:]]
Y=df['Survived']

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
accuracy=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]

for i in models:
    cv_result = cross_val_score(i,X,Y, cv = kfold,scoring = "accuracy")
    accuracy.append(cv_result.mean())
new_models_dataframe2=pd.DataFrame({'CV Mean':accuracy},index=classifiers) 
new_models_dataframe2



new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)



f,ax=plt.subplots(3,3,figsize=(12,10))

y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')

y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')

y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')

y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')

y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')

y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)



from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)



n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(train_X,train_Y)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())



from sklearn.ensemble import BaggingClassifier

model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())



model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())



from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())



import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())



from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())




n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)





f,ax=plt.subplots(2,2,figsize=(15,12))

model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')

model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1])
ax[0,1].set_title('Feature Importance in AdaBoost')

model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0])
ax[1,0].set_title('Feature Importance in Gradient Boosting')

model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1])
ax[1,1].set_title('Feature Importance in XgBoost')




finalResult=pd.DataFrame(testprediction)
finalResult.columns=['Survived']
finalResult.index=testData.index



finalResult.reset_index()
testData.reset_index()
finalOutput=pd.merge(finalResult,testData,on='PassengerId',how='inner')
finalOutput.reset_index()



finalOutput.to_csv('outputs.csv')