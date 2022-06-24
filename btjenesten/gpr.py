# Authors: Christian Elias Anderssen Dalan <ceadyy@gmail.com>
# With the help of Audun Skau Hansen <a.s.hansen@kjemi.uio.no>
# April 2022

import numpy as np
import kernels as knls

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
            
            if return_covariance:
                predicted_covariance_matrix = K_22 - KT @ K_12
                return predicted_y, predicted_covariance_matrix
            else:
                return predicted_y
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
        Returns the average error of our predict method.

        Parameters:
        -----------
        input_data_X:
        input data that the gpr will predict corresponding output data to.

        input_data_Y:
        Corresponding true ouput data for input_data_X.

        Returns:
        --------
        avg_error - the average error between the predicted values and the true values
        """

        predicted_y = self.predict(input_data_X)
        avg_error = np.mean(np.abs(predicted_y - input_data_Y))
        return avg_error
