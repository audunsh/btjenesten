# Authors: Christian Elias Anderssen Dalan <ceadyy@gmail.com>
# With the help of Audun Skau Hansen <a.s.hansen@kjemi.uio.no>
# April 2022

import numpy as np
import btjenesten.kernels as knls
from scipy.optimize import minimize

class Kernel():
    """
    Kernel class 

    Parameters:
    -----------
    covariance_function:
    The function that will be used to calculate the covariance between our datasets
    """

    def __init__(self, covariance_function):
        self.covariance_function = covariance_function
    
    def K(self, X1, X2):
        """
        Function that returns the covariance matrix given our datasets

        Parameters:
        -----------
        X1: Dataset 1 (Often the training set)

        X2: Dataset 2 (Often the target set)

        Returns:
        ----------
        self.covariance_function(X1, X2) : covariance matrix given our datasets X1 and X2.
        """
        if np.isscalar(X1):
            X1 = np.array([X1])
        if np.isscalar(X2):
            X2 = np.array([X2])

        return self.covariance_function(X1, X2)


class Regressor():
    """
    Gaussian process regressor class

    Parameters:
    -----------
    kernel:
    Specifies the type of covarince function we want for our regressor. 
    If none is provided the default is the radial basis function

    training_data_X: 
    Training data inputs, also called features

    training_data_Y:
    Training data outputs, also called labels
    """

    def __init__(self, training_data_X, training_data_Y, kernel = None):
        if kernel == None:
            self.kernel = Kernel(knls.RBF)
        else:
            self.kernel = Kernel(kernel)
        self.training_data_X = training_data_X
        self.training_data_Y = training_data_Y

    def predict(self, input_data_X, training_data_X = None, training_data_Y = None, return_covariance = False):
        """
        Predicts output values for some input data given a set of training data 

        Parameters:
        -----------
        input_data_X:
        Input features that the gpr will evaluate.

        training_data_X:
        training data inputs.

        training_data_Y:
        training data outputs.
        
        return_variance:
        Returns variance for each prediction if this is true

        Returns:
        -----------
        predicted_y:
        Predicted output data given cooresponding input_data_X and a set of training data
        inputs and outputs (training_data_X, training_data_Y)
        
        predicted_variance:
        Predicted variance for each point of predicted output.
        """
        if training_data_X == None or training_data_Y == None:
            K_11 = self.kernel.K(self.training_data_X, self.training_data_X)
            K_12 = self.kernel.K(self.training_data_X, input_data_X)
            K_21 = K_12.T
            K_22 = self.kernel.K(input_data_X, input_data_X)

            KT = np.linalg.solve(K_11, K_12).T
            
            predicted_y = KT.dot(self.training_data_Y)
            
        else:
            K_11 = self.kernel.K(training_data_X, training_data_X)
            K_12 = self.kernel.K(training_data_X, input_data_X)
            K_21 = self.kernel.K(input_data_X, training_data_X)
            K_22 = self.kernel.K(input_data_X, input_data_X)

            KT = np.linalg.solve(K_11, K_12).T

            predicted_y = KT.dot(training_data_Y)

        if return_covariance:
            predicted_covariance_matrix = K_22 - KT @ K_12
            return predicted_y, predicted_covariance_matrix
        else:
            return predicted_y

    def score(self, input_data_X, input_data_Y):
        """
        Returns the average and maximum error of our predict method.

        Parameters:
        -----------
        input_data_X:
        input data that the gpr will predict corresponding output data to.

        input_data_Y:
        Corresponding true ouput data for input_data_X.

        Returns:
        --------
        avg_error - the average error between the predicted values and the true values
        max_error - the maximum error between the predicted values and the true values
        """

        predicted_y = self.predict(input_data_X)
        avg_error = np.mean(np.abs(predicted_y - input_data_Y))
        max_error = np.max(np.abs(predicted_y - input_data_Y))
        return avg_error
    
    def aquisition(self, minimize_prediction=True):
        """
        Returns the point at which our model function is predicted to have the highest value.

        Parameters:
        -----------
        minimize_prediction:
        If your task is to minimize some model function, this parameter is True. If your task is to maximize the model function
        this parameter is False.

        Returns:
        --------
        p - The predicted point at which an evaluation would yeild the highest/lowest value
        """
        if minimize_prediction:
            x0_index = np.where(self.training_data_Y == np.min(self.training_data_Y))

            x0 = self.training_data_X[x0_index]
            minimization = minimize(self.predict, x0)
            print(minimization)
            p = minimization.x

            return p
        else:
            x0_index = np.where(self.training_data_Y == np.min(self.training_data_Y))

            x0 = self.training_data_X[x0_index]
            objective_function = lambda x, predict = self.predict : -1*predict(x)



            minimization = minimize(objective_function, x0)
            print(minimization)
            p = minimization.x

            return p
        

    def update(self, new_X, new_Y):
        """
        Updates the training data in accordance to the new one
        """
        X_shape = np.array(self.training_data_X.shape)
        y_shape = np.array(self.training_data_Y.shape)

        X_shape[0] += 1
        y_shape += 1

        new_training_data_X = np.zeros(X_shape)
        new_training_data_Y = np.zeros(y_shape)

        new_training_data_X[:-1] = self.training_data_X
        new_training_data_X[-1] = new_X 

        new_training_data_Y[:-1] = self.training_data_Y
        new_training_data_Y[-1] = new_Y

        indexes = np.argsort(new_training_data_X)

        self.training_data_X = new_training_data_X[indexes]
        self.training_data_Y = new_training_data_Y[indexes]

