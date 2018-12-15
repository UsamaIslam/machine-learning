#!/usr/bin/env python
# coding: utf-8

# In[22]:


from skimage import io
import numpy as np
import random
import numpy.matlib
import imageio
import os


# In[23]:


def init_centroids(X, K):
    c = random.sample(list(X), K)
    return c


# In[132]:


def closest_centroids(X, c):
    K = np.size(c, 0)
    idx = np.zeros((np.size(X, 0), 1))
    arr = np.empty((np.size(X, 0), 1))
    for i in range(0, K):
        y = c[i]
        # print('centroid values: ',i,y)
        temp = np.ones((np.size(X, 0), 1)) * y
        b = np.power(np.subtract(X, temp), 2)
        # print(b.shape)
        a = np.sum(b, axis=1)
        # print(a.shape)
        # a = np.asarray(a)
        # print(a)
        a.resize((np.size(X, 0), 1))
        # print(a.shape)
        # print(a)
        # print(np.shape(a))
        arr = np.append(arr, a, axis=1)
        # print(arr)
    arr = np.delete(arr, 0, axis=1)
    # print(arr.shape)
    # print(arr)
    idx = np.argmin(arr, axis=1)
    # print(idx.shape)
    # [print(i) for i in idx]
    return idx


# In[177]:


def compute_centroids(X, idx, K):
    n = np.size(X, 1)
    print(n)
    centroids = np.zeros((K, n))
    for i in range(0, K):
        ci = idx == i
        ci = ci.astype(int)
        total_number = sum(ci)
        # print(ci.shape)
        ci.resize((np.size(X, 0), 1))
        total_matrix = np.matlib.repmat(ci, 1, n)
        print(total_matrix.shape)
        ci = np.transpose(ci)
        total = np.multiply(X, total_matrix)
        centroids[i] = (1 / total_number) * np.sum(total, axis=0)
    return centroids


# In[107]:


def run_kMean(X, initial_centroids, max_iters):
    m = np.size(X, 0)
    n = np.size(X, 1)
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids
    #We can run a counter here to check if previous centroids are same as the new centroids but
    #it's not a good approach. program can stick foreever. 
    #So we are updating our centroids in range(0,max_iterations)
    previous_centroids = centroids
    # print(centroids)
    idx = np.zeros((m, 1))
    for i in range(0, max_iters):
        idx = closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx


# In[110]:
if __name__ == '__main__':
    image = io.imread('test.png')
    # image = image / 255
    X = image.reshape(-1, 3)
    # print(X)
    K = 2  # number of clusters
    max_iters = 20  # number of times the k-mean should run

    initial_centroids = init_centroids(X, K)

    # In[178]:

    centroids, idx = run_kMean(X, initial_centroids, max_iters)

    # In[ ]:

    # idx.resize((np.size(X,0),1))
    print(np.shape(centroids))
    print(np.shape(idx))
    idx = closest_centroids(X, centroids)
    X_recovered = centroids[idx]
    print(np.shape(X_recovered))
    X_recovered = np.reshape(X_recovered, (np.size(image, 0), np.size(image, 1), 3))
    print(np.shape(X_recovered))

    imageio.imwrite('test_output.jpg', X_recovered)
    image_compressed = io.imread('test_output.jpg')
    io.imshow(image_compressed)
    io.show()
