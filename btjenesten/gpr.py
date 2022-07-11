# Authors: Christian Elias Anderssen Dalan <ceadyy@gmail.com>
# With the help of Audun Skau Hansen <a.s.hansen@kjemi.uio.no>
# April 2022

import numpy as np
#from btjenesten import kernels as knls
import btjenesten.kernels as knls
#import kernels as knls
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
    
    def K(self, X1, X2, params):
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

        return self.covariance_function(X1, X2, params)


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

    params:
    
    """

    def __init__(self, training_data_X, training_data_Y, kernel = None, params = 1):
        if kernel == None:
            self.kernel = Kernel(knls.RBF)
        else:
            self.kernel = Kernel(kernel)

        msg = "Expected 2D array. If you only have one feature reshape training data using array.reshape(-1, 1)"
        assert training_data_X.ndim != 1, msg
        
        self.training_data_X = training_data_X
        self.training_data_Y = training_data_Y

        self.params = 1 # 

    def predict(self, input_data_X, training_data_X = None, training_data_Y = None, return_variance = False):
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
            K_11 = self.kernel.K(self.training_data_X, self.training_data_X, self.params)
            K_12 = self.kernel.K(self.training_data_X, input_data_X, self.params)
            K_21 = K_12.T
            K_22 = self.kernel.K(input_data_X, input_data_X, self.params)
            assert (np.linalg.det(K_11) != 0), "Singular matrix. Training data might have duplicates."
            KT = np.linalg.solve(K_11, K_12).T
            
            predicted_y = KT.dot(self.training_data_Y)
            
        else:
            K_11 = self.kernel.K(training_data_X, training_data_X, self.params)
            K_12 = self.kernel.K(training_data_X, input_data_X, self.params)
            K_21 = self.kernel.K(input_data_X, training_data_X, self.params)
            K_22 = self.kernel.K(input_data_X, input_data_X, self.params)

            assert (np.linalg.det(K_11) != 0), "Singular matrix. Training data might have duplicates."
            KT = np.linalg.solve(K_11, K_12).T

            predicted_y = KT.dot(training_data_Y)

        predicted_y = predicted_y.ravel()

        if return_variance:
            predicted_variance = np.diag(K_22 - KT @ K_12)
            
            y_var_negative = predicted_variance < 0
            if np.any(y_var_negative):
                predicted_variance.setflags(write="True")
                predicted_variance[y_var_negative] = 0

            return predicted_y, predicted_variance
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
        return avg_error, max_error
    
    def aquisition(self, minimize_prediction=True, x0 = None, l=1.2, delta=0.1):
        """
        Returns the point at which our model function is predicted to have the highest value.

        Parameters:
        -----------
        minimize_prediction:
        If your task is to minimize some model function, this parameter is True. If your task is to maximize the model function
        this parameter is False.

        l:
        Exploration parameter. Scales how much the standard deviation should impact the function value. l = 1
        means that the function maximized/minimized equals predicted value +/- the standard deviation.
        
        x0:
        Initial guess. If not specified it will use the point at which the training data is the largest/smallest.

        delta:
        Hyperparameter that tunes UCB around measured datapoints.
        
        Returns:
        --------
        p - The predicted point at which an evaluation would yeild the highest/lowest value
        """
        if minimize_prediction: #Minimization process
            if x0 == None:
                x0_index = np.where(self.training_data_Y == np.min(self.training_data_Y))

                x0 = self.training_data_X[x0_index]

            objective_function = lambda x, predict = self.predict : predict(x)
            std_x = lambda x, predict = self.predict : np.sqrt(np.abs(np.diag(predict(x, return_variance = True)[1])))
            objective_noise = lambda x, std = std_x : (1 - std(x))**2 * delta + std(x)

            UCB = lambda x, exploit = objective_function, explore = objective_noise: exploit(x) + l*explore(x) 

            def UCB(x, f = UCB):
                x = x.reshape(1, -1)
                return f(x)

            minimization = minimize(UCB, x0)
            p = minimization.x
            return p

        else: #Maximization process
            if x0 == None:
                x0_index = np.where(self.training_data_Y == np.max(self.training_data_Y))

                x0 = self.training_data_X[x0_index]

            objective_function = lambda x, predict = self.predict : predict(x)
            std_x = lambda x, predict = self.predict : np.sqrt(np.abs(np.diag(predict(x, return_variance = True)[1])))
            objective_noise = lambda x, std = std_x : (1 - std(x))**2 * delta + std(x)

            UCB = lambda x, exploit = objective_function, explore = objective_noise : -1*(exploit(x) + l*explore(x))
            def UCB(x, f = UCB):
                x = x.reshape(1, -1)
                return f(x)

            minimization = minimize(UCB, x0)
            p = minimization.x
            return p

    def update(self, new_X, new_Y, tol=1e-5):
        """
        Updates the training data in accordance to some newly measured data.

        Parameters:
        -----------
        new_X:
        Set of new features that have been measured.

        new_Y:
        Corresponding set of labels to new_X.

        tol:
        Tolerance which the training data set can differ from new points. If this is too low you may encounter singular 
        covariance matrices.
        """

        assert type(new_Y) is np.ndarray, "Data error!!!!! Needs to be array."
        assert type(new_X) is np.ndarray, "Data error!!!!! Needs to be array."

        for measurement in new_X.reshape(-1, self.training_data_X.shape[1]):
            for i in range(len(self.training_data_X)):
                if np.allclose(measurement, self.training_data_X[i], atol = tol):
                    print(f"The model has most likely converged! {measurement} already exists in the training set.")
                    return True
        """
        old_X_shape = self.training_data_X.shape
        old_Y_shape = len(self.training_data_Y)

        new_X_shape = np.array(self.training_data_X.shape)
        new_Y_shape = len(new_Y)

        new_X_shape[0] += new_X.shape[0]
        new_Y_shape += len(new_Y)

        new_training_data_X = np.zeros(new_X_shape)
        new_training_data_Y = np.zeros(new_Y_shape)

        new_training_data_X[:-old_X_shape.shape[0]] = self.training_data_X
        new_training_data_X[-new_X.shape[0]:] = new_X 

        new_training_data_Y[:-old_Y_shape] = self.training_data_Y
        new_training_data_Y[-new_Y_shape:] = new_Y
        """
        #print("X1 shape ",self.training_data_X.shape)
        #print("X2 shape ",.shape)
        new_X = new_X.reshape(-1, self.training_data_X.shape[1])

        new_training_data_X = np.concatenate((self.training_data_X, new_X))
        new_training_data_Y = np.concatenate((self.training_data_Y, new_Y))

        #indexes = np.argsort(new_training_data_X)

        self.training_data_X = new_training_data_X#[indexes]
        self.training_data_Y = new_training_data_Y#[indexes]

        return False

