import numpy as np
from scipy.optimize import minimize
import btjenesten as bt

def parameter_optimization(x_data, y_data, training_fraction, normalize_y = True, params = None):
    """
    Draft for a parameter optimization scheme
    Author: Audun Skau Hansen
    
    Takes as input the dataset (x_data, y_data) and the 
    fraction (a float in the interval 0.0 - 1.0) of datapoints
    to use as training data.
    """
    n = int(training_fraction*x_data.shape[0])
    
    # special first iteration
    if params is None:
        params = np.ones(x_data.shape[1])*-2.0 #*0.001
    #training_subset = np.random.choice(x_data.shape[0], n, replace = False)
    training_subset = np.ones(x_data.shape[0], dtype = bool)
    training_subset[n:] = False
    #print(training_subset)
    y_data_n = y_data*1
    if normalize_y:
        y_data_n*=y_data_n.max()**-1
    
    def residual(params, x_data = x_data, y_data=y_data_n, training_subset = training_subset):
        test_subset = np.ones(x_data.shape[0], dtype = bool)
        test_subset[training_subset] = False
        regressor = bt.gpr.Regressor(x_data[training_subset] , y_data[training_subset]) 
        regressor.params = 10**params
        energy = np.sum((regressor.predict(x_data[test_subset]) - y_data[test_subset])**2)
        return energy
    
    
    ret = minimize(residual, params)
    print(ret)
    return 10**ret["x"]


def remove_redundancy(x_train, y_train, tol = 10e-8):
    """
    extract unique columns of x_train (and corresponding elements in y_train)

    Author: Audun Skau Hansen

    """


    ns = x_train.shape[0] #number of measurements

    # compute the "euclidean distance"
    d = np.sum((x_train[:, None] - x_train[None, :])**2, axis = 2)


    
    active = np.ones(ns, dtype = bool)
    
    
    unique_training_x = []
    unique_training_y = []

    for i in range(ns):

        distances = d[i]
        
        da = distances[active]
        ia = np.arange(ns)[active]
        
        elms = ia[da<tol]
        active[elms] = False
        
        if len(elms)>0:
            unique_training_x.append(x_train[elms[0]])
            unique_training_y.append(np.mean(y_train[elms], axis = 0))


    return np.array(unique_training_x), np.array(unique_training_y)
    
    



# analysis tools for regressor data

def data_projection(regressor, axes = [0], resolution = 20, center = None):
    """
    Project high-dimensional regressor predictions onto smaller 
    spaces.
    
    Author: Audun Skau Hansen
    
    Arguments
    ===

    regressor  = a gpr regressor 
    axies      = indices of axis to sweep over
    resolution = resolution of sweeps along all axes
    center     = (optional) set center point of sweep (default is middle of the region)
    
    Examples
    ===
    
    x, y = data_projection(regressor, axes = [0]) -> 
    all axes except 0 are kept fixed at the mean values (or center values), 
    while 0 is allowed to vary inside the fitting region.
    plot(x[0], y) 
    
    x, y = data_projection(regressor, axes = [1,2]) -> 
    all axes except 1 and 2 are fixed to mean values (or center values)
    while 1 and 2 are allowed to map a surface in the fitting region.
    contourf(x[0], x[1], y)
    
    """
    

    # extract fitting regions (only internal datapoints will be predicted)
    mean  = np.mean(regressor.recover(regressor.training_data_X), axis =0 )
    if center is not None:
        mean = np.array(center)

    #print(regressor.recover(regressor.training_data_X))
    bound = np.max(regressor.recover(regressor.training_data_X), axis = 0)-np.min(regressor.recover(regressor.training_data_X), axis = 0)
    #print(mean, bound)
    lower_bound = mean - bound
    upper_bound = mean + bound


    # create a grid wherein the datapoints are interpolated
    grid = []
    for i in range(len(lower_bound)):
        grid.append(np.linspace(-bound[i], bound[i], resolution))

    #if center is None:
    #    center = np.zeros(len(mean), dtype = float)


    mgrid = list(mean) #[0 for i in range(len(center))]
    for i in range(len(axes)):
        mgrid[axes[i]] = grid[axes[i]]  + mean[axes[i]]
        


    prediction_grid = np.array(np.meshgrid(*mgrid)).reshape(len(mean), -1)

    # return prediction to user
    x = [] # list to contain the relevant grid-points    
    for i in range(len(axes)):
        x.append(mgrid[axes[i]])
    
    return x, regressor.predict(prediction_grid.T).reshape([resolution for i in range(len(axes))])