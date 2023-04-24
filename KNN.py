__authors__ = ['1638117', '1639392', '1550960']
__group__ = 'DM.12'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.train_data = np.array(train_data,float)
        N = np.shape(train_data)[0]
        PxI = np.shape(train_data)[1] * np.shape(train_data)[2]
        self.train_data = np.reshape(self.train_data,(N,PxI))

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.neighbors = []
        self.test_data = np.array(test_data, float)
        N = np.shape(test_data)[0]
        PxI = np.shape(test_data)[1] * np.shape(test_data)[2]
        self.test_data = np.reshape(self.test_data, (N, PxI))
        distances = cdist(self.test_data,self.train_data)
        min_index = np.argsort(distances)
        for index in min_index:
            labels = []
            for i in range(k):
                labels.append(self.labels[index[i]])
            self.neighbors.append(labels)


       #Podria treure's un for si conseguim fer que el min_index només tingui els 3 valors que volem examinar

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        ######################################################
        a =[]
        b =[]
        for i in self.neighbors:
            a.append(max(i, key=i.count))                       #First array completat
            
        diccionario_frecuencias = {}
        for i in range(len(self.neighbors)):
            for j in self.neighbors[i]:
                if j in diccionario_frecuencias:
                    diccionario_frecuencias[j] += 1
                else:
                    diccionario_frecuencias[j] = 1              #Obtenim el total de cada una de les paraules
                    
        m = sum(diccionario_frecuencias.values())               #Suma de totes les paraules
        for i in a:
            b.append((diccionario_frecuencias.get(i)/m)*100)    #Total de cada Classe representant entre totes
        return a,b  
        
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
