import numpy as np

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
    mean  = np.mean(regressor.training_data_X, axis =0 )
    bound = np.max(regressor.training_data_X, axis = 0)-np.min(regressor.training_data_X, axis = 0)

    lower_bound = mean - bound
    upper_bound = mean + bound


    # create a grid wherein the datapoints are interpolated
    grid = []
    for i in range(len(lower_bound)):
        grid.append(np.linspace(lower_bound[i], upper_bound[i], resolution))

    if center is None:
        center = list(mean)

    mgrid = [0 for i in range(len(center))]
    for i in range(len(axes)):
        mgrid[i] = grid[i]

    prediction_grid = np.array(np.meshgrid(*mgrid)).reshape(len(mean), -1)

    # return prediction to user
    return np.array(grid)[axes], regressor.predict(prediction_grid.T).reshape([resolution for i in range(len(axes))])
    
    