# Kmeans over pokemon dataset
Kmeans is an unsupervised learning algorithm that uses clusters to label training examples.

It works like this:

1. Initialice randomly K clusters
2. For each training example compute euclidean distance for every cluster and choose nearest
3. For each cluster compute mean for every labeled training example with its identifier and move cluster there

Repeat until converge, when centroids are not moving anymore

I used 2 pokemon features, attack and defense; because is easier to plot. Kmeans works with N-dimensional vector.


Thanks to Coursera and Andrew Ng, I encourage you to take this course:
https://www.coursera.org/learn/machine-learning/home/welcome

Regards :)