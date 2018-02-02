import sklearn as sk
from sklearn import tree

##TRAINING DATA
# [weight, bumpy=1,smooth=0]
features = [[140,1],[130,1],[150,0],[170,0]]
#[apple=0 orange=1 ]
labels = [0,0,1,1]
### Use data to train the classifer on which is a orange which is an apple
clf = tree.DecisionTreeClassifier()
#training algorith here
#fit() = find patterns in data
clf = clf.fit(features, labels)
#Test out algo
print(clf.predict([[150,0]]))
