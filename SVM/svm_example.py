import matplotlib.pyplot as plt 
from sklearn import datasets  # datasets are used as a sample dataset, contains set that has number recognition data
from sklearn import svm 

digits = datasets.load_digits() 

print(digits.data) 

print(digits.target)  

clf = svm.SVC()  # specify the classifier using defaults

clf = svm.SVC(gamma=0.001, C=100)  # specify the classifier

X, y = digits.data[:-10], digits.target[:-10]

clf.fit(X, y)  # train

print("Prediction: ", clf.predict(digits.data[-5]))  

# visualization
plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# adjusting gamma
# larger values increase speed, lower accuracy
# speed changes by factors of 10
clf = svm.SVC(gamma=0.01, C=100)
clf.fit(X, y)  # train
print("Prediction: ", clf.predict(digits.data[-5])) 

# less accurate
clf = svm.SVC(gamma=0.0001, C=100)
clf.fit(X, y)  # train
print("Prediction: ", clf.predict(digits.data[-5]))  
