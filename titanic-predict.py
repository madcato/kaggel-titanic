import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor

main_file_path = './data/train.csv'
data = pd.read_csv(main_file_path)
data.set_index('PassengerId')
print(data.describe())

def transformCabin(x):
    return 0

def transformNan(x):
    if math.isnan(x):
        return 0
    return x

def transformAge(x):
    if math.isnan(x):
        return -1
    return x

# predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Embarked']
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

## Preprocess data to convert each value to float
def transform(dt):
    dt = dt.replace({'Embarked': {'S': 3, 'C': 1, 'Q': 2}})
    dt = dt.replace({'Pclass': {1: 0.9, 2: 0.5, 3: 0.2}})
    dt['Cabin'] = dt['Cabin'].apply(lambda x: transformCabin(x))
    dt['Age'] = dt['Age'].apply(lambda x: transformAge(x))
    dt = dt.replace({'Sex': {'male': 0, 'female': 1}})
    dt['Pclass'] = dt['Pclass'].apply(lambda x: transformNan(x))
    dt['Sex'] = dt['Sex'].apply(lambda x: transformNan(x))
    dt['Pclass'] = dt['Pclass'].apply(lambda x: transformNan(x))
    dt['Age'] = dt['Age'].apply(lambda x: transformNan(x))
    dt['SibSp'] = dt['SibSp'].apply(lambda x: transformNan(x))
    dt['Parch'] = dt['Parch'].apply(lambda x: transformNan(x))
    dt['Cabin'] = dt['Cabin'].apply(lambda x: transformNan(x))
    dt['Embarked'] = dt['Embarked'].apply(lambda x: transformNan(x))
    # dt = dt.dropna()
    return dt

def transformPrediction(x):
    if x < 0.9:
        return 0
    return 1

data = transform(data)

X = data[predictors]
Y = data.Survived

print X

# Define model
model = DecisionTreeRegressor()

# Fit model
model.fit(X, Y)

print("Making predictions for the following 5 passengers:")
print(X.head())
print("The predictions are")
print(model.predict(X.head()))


# Check test
main_file_path = './data/test.csv'
test_data = pd.read_csv(main_file_path)
test_data.set_index('PassengerId')
test_data = transform(test_data)
Z = test_data[predictors]
result = model.predict(Z)

prediction = pd.DataFrame(result,test_data['PassengerId'])

prediction[0] = prediction[0].apply(lambda x: transformPrediction(x))

prediction.to_csv("prediction.csv")
