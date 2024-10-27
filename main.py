import pandas as pd
from DecisionTreeClassifier import *
import matplotlib.pyplot as plt


# Read the data
dataset = pd.read_csv('PlayTennis.csv', header=0)
# print first five samples in the dataset
print(dataset.head())
# print data statistics
print(dataset.describe())
print(dataset.info())


# Split the dataset into train and test
X = dataset.drop(columns=['Play Tennis'])
Y = dataset['Play Tennis']
# Create model and train it
DTModel = DecisionTreeClassifier()
DTModel.fit(X,Y)


# Print the nodes, its children and thresholds
DTModel.print()

# Make predictions
test_X = np.array([['Overcast','Cool','High','Strong'],
                    ['Sunny','Mild','Normal','Strong'],
                    ['Rain','Mild','High','Strong']])
predictions = DTModel.predict(test_X)

# Visualize the tree 
DTModel.plot()
myimage = plt.imread('/content/decision_tree.png')
plt.imshow(myimage)




