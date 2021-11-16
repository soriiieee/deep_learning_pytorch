# -*- coding: utf-8 -*-
# when   : 2020.03.12
# when   : 2020.05.25 update
# who : [sori-machi]
# pyPIにあった、skelarn_somの中身、距離の測り方を
# distance　="ED"と"SSIM"で区別する
"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
reference:
https://github.com/rileypsmith/sklearn-som/blob/main/sklearn_som/som.py

"""
import sys
import numpy as np
np.random.seed(41)

class SOM():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, m=3, n=3, dim=3, lr=1, sigma=1,distance="ED",max_iter=3000, epoch=1):
        """
        Parameters
        ----------
        m : int, default=3
            The shape along dimension 0 (vertical) of the SOM.
        n : int, default=3
            The shape along dimesnion 1 (horizontal) of the SOM.
        dim : int, default=3
            The dimensionality (number of features) of the input space.
        lr : float, default=1
            The initial step size for updating the SOM weights.
        sigma : float, optional
            Optional parameter for magnitude of change to each weight. Does not
            update over training (as does learning rate). Higher values mean
            more aggressive updates to weights.
        distance:
          sorimachi add-> "ED"= "eucledinan"/ "SSIM" = "S-SOM"
          
        max_iter : int, optional
            Optional parameter to stop training if you reach this many
            interation.
        """
        # Initialize descriptive features of SOM
        self.m = m
        self.n = n
        self.dim = dim
        self.shape = (m, n)
        self.initial_lr = lr
        self.lr = lr
        self.sigma = sigma
        self.distance = distance #2021.03.12 sorimachi add
        self.epoch = epoch #2021.03.16 sorimachi add
        self.max_iter = max_iter

        # Initialize weights
        self.weights = np.random.normal(size=(m * n, dim))
        self._locations = self._get_locations(m, n)
        
        #m*n の単一配列を表示してbmuの参照配列
        # [[0 0][0 1] [0 2] [0 3][1 0] [1 1] [1 2] [1 3] [2 0] [2 1] [2 2] [2 3]]
        # Set after fitting
        self._inertia = None
        self._n_iter_ = None
        self._trained = False

    def _get_locations(self, m, n):
        """
        Return the indices of an m by n array.
        """
        return np.argwhere(np.ones(shape=(m, n))).astype(np.int64)

    def _find_bmu(self, x):
        """
        Find the index of the best matching unit for the input vector x.
        """
        # Stack x to have one row per weight
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        # Calculate distance between x and each weight
        
        #-------------------------------------------------------
        # sorimachi add 2021.03.12--->
        if self.distance == "ED":
          """
          reference -> 
          ユークリッド距離= 残差ベクトルノルム
          """
          distance = np.linalg.norm(x_stack - self.weights, axis=1)
          return np.argmin(distance)
      
        if self.distance == "SSIM":
          """
          reference -> 
          okada introduction -thesis
          """
          x_mean, x_std = np.mean(x_stack,axis=1), np.std(x_stack,axis=1)
          y_mean, y_std = np.mean(self.weights,axis=1),np.std(self.weights,axis=1)
          
          xy_std = np.mean(np.dot((x_stack - x_mean.reshape(-1,1)),(self.weights- y_mean.reshape(-1,1)).T),axis=1)
          
          ssim = (2* x_mean * y_mean * xy_std) / ((x_mean*x_mean + y_mean*y_mean)*(x_std*x_std + y_std*y_std))
          return np.argmax(ssim)
        #-------------------------------------------------------

    def step(self, x):
        """
        Do one step of training on the given input vector.
        """
        # Stack x to have one row per weight
        x_stack = np.stack([x]*(self.m*self.n), axis=0)

        # Get index of best matching unit
        bmu_index = self._find_bmu(x)

        # Find location of best matching unit
        bmu_location = self._locations[bmu_index,:]
        # print(self._locations)
        # print("bmu_location=",bmu_location) [1,3]
        # sys.exit()

        # Find square distance from each weight to the BMU
        stacked_bmu = np.stack([bmu_location]*(self.m*self.n), axis=0)
  
        bmu_distance = np.sum(np.power(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64), 2), axis=1)
        # print("bmu_distance=",bmu_distance)
        # sys.exit()
        # Compute update neighborhood
        neighborhood = np.exp((bmu_distance / (self.sigma ** 2)) * -1)
        local_step = self.lr * neighborhood
        
        # print("neighborhood=",neighborhood)
        # print(self.lr * neighborhood)
        # sys.exit()

        # Stack local step to be proper shape for update
        local_multiplier = np.stack([local_step]*(self.dim), axis=1)
        # print("local_multiplier=",local_multiplier)
        # sys.exit()
        # Multiply by difference between input and weights
        delta = local_multiplier * (x_stack - self.weights)

        # print(delta)
        # sys.exit()
        # Update weights
        self.weights += delta

    def _compute_point_intertia(self, x):
        """
        Compute the inertia of a single point. Inertia defined as squared distance
        from point to closest cluster center (BMU)
        """
        # Find BMU
        bmu_index = self._find_bmu(x) #近傍でmbuを求めるクラス関数2021.03.21
        bmu = self.weights[bmu_index]
        # Compute sum of squared distance (just euclidean distance) from x to bmu
        return np.sum(np.square(x - bmu))

    def fit(self, X, epochs=1, shuffle=True):
        """
        Take data (a tensor of type float64) as input and fit the SOM to that
        data for the specified number of epochs.
        Parameters
        ----------
        X : ndarray
            Training data. Must have shape (n, self.dim) where n is the number
            of training samples.
        epochs : int, default=1
            The number of times to loop through the training data when fitting.
        shuffle : bool, default True
            Whether or not to randomize the order of train data when fitting.
            Can be seeded with np.random.seed() prior to calling fit.
        Returns
        -------
        None
            Fits the SOM to the given data but does not return anything.
        """
        # Count total number of iterations
        global_iter_counter = 0
        n_samples = X.shape[0]
        total_iterations = np.minimum(epochs * n_samples, self.max_iter)

        for epoch in range(self.epoch):
            # Break if past max number of iterations
            if global_iter_counter > self.max_iter:
                break

            if shuffle:
                indices = np.random.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            # Train
            for idx in indices:
                # Break if past max number of iterations
                if global_iter_counter > self.max_iter:
                    break
                input = X[idx]
                # Do one step of training
                self.step(input)
                # Update learning rate
                global_iter_counter += 1
                self.lr = (1 - (global_iter_counter / total_iterations)) * self.initial_lr

        # Compute inertia
        inertia = np.sum(np.array([float(self._compute_point_intertia(x)) for x in X]))
        self._inertia_ = inertia

        # Set n_iter_ attribute
        self._n_iter_ = global_iter_counter

        # Set trained flag
        self._trained = True

        return

    def predict(self, X):
        """
        Predict cluster for each element in X.
        Parameters
        ----------
        X : ndarray
            An ndarray of shape (n, self.dim) where n is the number of samples.
            The data to predict clusters for.
        Returns
        -------
        labels : ndarray
            An ndarray of shape (n,). The predicted cluster index for each item
            in X.
        """
        # Check to make sure SOM has been fit
        if not self._trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimesnion {self.dim}. Received input with dimension {X.shape[1]}'

        labels = np.array([self._find_bmu(x) for x in X])
        return labels

    def transform(self, X):
        """
        Transform the data X into cluster distance space.
        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim) where n is the number of samples. The
            data to transform.
        Returns
        -------
        transformed : ndarray
            Transformed data of shape (n, self.n*self.m). The Euclidean distance
            from each item in X to each cluster center.
        """
        # Stack data and cluster centers
        X_stack = np.stack([X]*(self.m*self.n), axis=1)
        cluster_stack = np.stack([self.weights]*X.shape[0], axis=0)

        # Compute difference
        diff = X_stack - cluster_stack

        # Take and return norm
        return np.linalg.norm(diff, axis=2)

    def fit_predict(self, X, **kwargs):
        """
        Convenience method for calling fit(X) followed by predict(X).
        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim). The data to fit and then predict.
        **kwargs
            Optional keyword arguments for the .fit() method.
        Returns
        -------
        labels : ndarray
            ndarray of shape (n,). The index of the predicted cluster for each
            item in X (after fitting the SOM to the data in X).
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return predictions
        return self.predict(X)

    def fit_transform(self, X, **kwargs):
        """
        Convenience method for calling fit(X) followed by transform(X). Unlike
        in sklearn, this is not implemented more efficiently (the efficiency is
        the same as calling fit(X) directly followed by transform(X)).
        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim) where n is the number of samples.
        **kwargs
            Optional keyword arguments for the .fit() method.
        Returns
        -------
        transformed : ndarray
            ndarray of shape (n, self.m*self.n). The Euclidean distance
            from each item in X to each cluster center.
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return points in cluster distance space
        return self.transform(X)

    @property
    def cluster_centers_(self):
        return self.weights.reshape(self.m, self.n, self.dim)

    @property
    def inertia_(self):
        if self._inertia_ is None:
            raise AttributeError('SOM does not have inertia until after calling fit()')
        return self._inertia_

    @property
    def n_iter_(self):
        if self._n_iter_ is None:
            raise AttributeError('SOM does not have n_iter_ attribute until after calling fit()')
        return self._n_iter_