import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


from sklearn.datasets import load_iris
iris = load_iris() 
from sklearn.svm import LinearSVC
svmClassifier = LinearSVC(random_state=111)

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.1, random_state=111)

svmClassifier.fit(X_train, y_train)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X_train)
pca_2d = pca.transform(X_train)


import pylab as pl
for i in range(0, pca_2d.shape[0]):
    if y_train[i] == 0:
        c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
    elif y_train[i] == 1:
	c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
    elif y_train[i] == 2:
    	c3 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
pl.legend([c1,c2,c3], ['Setosa', 'Versicolor', 'Virginica']) 
pl.title("Iris training dataset with 3 classes and known outcomes") 
#pl.show() 
pl.savefig('plot.png')

