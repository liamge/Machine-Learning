# Machine-Learning
A repository for the exploration of Machine Learning techniques.

Markov Chains:


	- Text file used is in the format of a list of tokenized sentences
	
	([['This', 'is', 'an', 'example'],['This', 'is', 'another', 'one']])


Linear Regression:


	- Uni/Multivariate Linear Regression algorithm from scratch ft. options for Gradient Descent or Normal Equation
	- Makes fancy plots and stuff
	- Work in progress
	- See more at: https://en.wikipedia.org/wiki/Linear_regression

Principle Component Analysis:
	
	- Allows for dimensionality reduction of any feature set X
	- Auto-selection of K enabled to choose least possible dimensions K s/t variance won't be reduced past a percentage that you can set

K-Means Clustering:

	- Implementation of K-Means Clustering from scratch
	- Graph function takes only two features as arguments currently, will fix that in future commits to allow for 3D scatter plots
	- Syntax Ex:
		km = KMeans()
		km.fit(X)
		centroids, idx = km.cluster(max_iterations, number_of_centroids)
		km.graph(feature_1, feature_2, idx)
