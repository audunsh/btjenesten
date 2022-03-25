import numpy as np

import ipywidgets as widgets
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class tablewidget():
    """
    Tabular in/out for Box-Behnken widgets
    Author: Audun Skau Hansen, 2022
    """
    def __init__(self, column_headers, row_headers):
        self.tab = np.zeros((len(row_headers)+1,len(column_headers)+1), dtype = object)
        #self.tab[1:,0] = row_headers
        #self.tab[0,1:] = column_headers
        self.row_headers = row_headers
        self.column_headers = column_headers
        
        self.tab[0,0] = widgets.Label("")
        
        for i in range(len(column_headers)):
            self.tab[0, i+1] = widgets.Label(column_headers[i], align = "right")
        for j in range(len(row_headers)):
            self.tab[j+1, 0] = widgets.Label(row_headers[j], align = "center")
        
        self.items = []
        for i in range(len(row_headers)):
            for j in range(len(column_headers)):
                self.tab[i+1,j+1] = widgets.BoundedFloatText(
                                        value=0,
                                        min=-1e15,
                                        max=1e15,
                                        step=0.1,
                                        description='',
                                        disabled=False
                                    )
                
        
        self.widget = widgets.GridBox(list(self.tab.ravel()), layout=widgets.Layout(grid_template_columns="repeat(%i, 100px)" % (len(column_headers)+1)))
        
    def as_numpy_array(self):
        """
        Returns the table (excluding headers) as a numpy array
        """
        ret = np.zeros(self.tab[1:,1:].shape, dtype = float)
        for i in range(self.tab.shape[0]-1):
            for j in range(self.tab.shape[1]-1):
                ret[i,j] = float(self.tab[i+1,j+1].value)
        return ret
    
    def set_from_array(self, input_array):
        """
        Set the table (excluding headers) from an input array
        """
        for i in range(self.tab.shape[0]-1):
            for j in range(self.tab.shape[1]-1):
                self.tab[i+1,j+1].value = str(input_array[i,j])
                
                
    
    def _repr(self):
        """
        Returns a latex-formatted string to display the mathematical expression of the basisfunction. 
        """
        return self.widget
    
def visualize_surfaces(bbwidget, Nx = 30):
    """
    Visualize response surfaces the regressor model
    **Author**: Audun Skau Hansen, Department of Chemistry, UiO (2022)

    ## Keyword arguments:

    | Argument      | Description |
    | ----------- | ----------- |
    | sheet      | Box-Benhken data sheet       |
    | regressor   | sklearn LinearRegression instance       |
    | Nx   | mesh resolution along each axis        |
    """
    data = bbwidget.as_numpy_array()
    bounds = np.zeros((3,2), dtype = float)
    bounds[:,0] = np.min(data[:,1:4], axis = 0)
    bounds[:,1] = np.max(data[:,1:4], axis = 0)
    
    #if regressor is None:
    # crop data from the sheet above 
        
    X_train = data[:,1:4]
    y_train = data[:, 4]
    
    #print(X_train, y_train)

    # perform a second order polynomial fit 
    # (linear in the matrix elements)
    degree=2 # second order
    poly= PolynomialFeatures(degree) # these are the matrix elements
    regressor=make_pipeline(poly,LinearRegression()) #set up the regressor
    regressor.fit(X_train,y_train) # fit the model



    # first, we extract all relevant information
    coefficients = regressor.steps[1][1].coef_
    names = poly.get_feature_names()
    predicted = regressor.predict(X_train)
    measured =  y_train
    score    = regressor.score(X_train, y_train)


    # we then compute and tabulate various statistics 
    squared_error = (predicted - measured)**2
    #print(predicted, measured)
    mean_squared_error = np.mean(squared_error)
    variance_error = np.var(squared_error)
    std_error = np.std(squared_error)


    # find max and min inside bounds using scipy.optimize
    from scipy.optimize import minimize
    mx = minimize(lambda x : -1*regressor.predict(np.array([x])), X_train[0], bounds = bounds)
    max_point = mx.x
    max_fun = -1*mx.fun
    mn = minimize(lambda x : regressor.predict(np.array([x])), X_train[0], bounds = bounds)
    min_point = mn.x
    min_fun = mn.fun




    #mse = mean_squared_error(predicted, measured) # mean squared error
    #mse = np.sum((predicted - measured)**2)/len(predicted) #alternative calculation
    print("Mean squared error :", mean_squared_error)
    print("Variance of error  :", variance_error)
    print("Standard dev. error:", std_error)
    print("Fitting score.     :", score)
    print("Maximum coords     :", max_point)
    print("Maximum value.     :", max_fun[0])
    print("Minimum coords     :", min_point)
    print("Minimum value.     :", min_fun[0])
        
    
    
    

    xa = np.linspace(bounds[0,0], bounds[0,1],Nx)
    xb = np.linspace(bounds[1,0], bounds[1,1],Nx)
    xc = np.linspace(bounds[2,0], bounds[2,1],Nx)

    va, vb, vc = bbwidget.column_headers[1], bbwidget.column_headers[2], bbwidget.column_headers[3]
    
    
    # displaying the fitting parameters

    fnames = relabel_defaults(poly.get_feature_names(), [va,vb,vc])
    ax, fig = plt.subplots(figsize=(9,5))
    plt.plot(regressor.steps[1][1].coef_, "s")
    #fig.set_xticklabels(poly.get_feature_names())
    for i in range(len(regressor.steps[1][1].coef_)):
        plt.text(i+.1,regressor.steps[1][1].coef_[i], fnames[i], ha = "left" , va = "center")
    plt.axhline(0)
    plt.title("Fitting parameters")
    plt.show()
    # print a table of the fitting parameters
    print(np.array([fnames, regressor.steps[1][1].coef_]).T)
    print("Intercept:", regressor.steps[1][1].intercept_)
    
    
    
    
    """
    plt.figure(figsize=(9.5,8))
    plt.title(va + " vs " + vb)
    plt.contourf(xa,xb,yab)
    plt.xlabel(va)
    plt.ylabel(vb)
    plt.colorbar()
    plt.show()
    """
    
    
    
    

    """
    plt.figure(figsize=(9.5,8))
    plt.title(va + " vs " + vc)
    plt.contourf(xa,xc,yac)
    plt.xlabel(va)
    plt.ylabel(vc)
    plt.colorbar()
    plt.show()
    """
    
    
    
    

    """
    plt.figure(figsize=(9.5,8))
    plt.title(vb + " vs " + vc)
    plt.contourf(xb,xc,ybc)
    plt.xlabel(vb)
    plt.ylabel(vc)
    plt.colorbar()
    plt.show()
    """
    #fig = plt.figure(figsize=(9,3))
    
    
    
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (9,4), sharey = True)
    #fig.
    

    
    #ax = fig.add_subplot(1, 3, 1)
    Xa = np.zeros((Nx,3), dtype = float)
    Xa[:,0] = xa
    Xa[:,1:] = min_point[1:]
    ax1.plot(xa, regressor.predict(Xa))
    ax1.set_xlabel(va)
    
    #ax = fig.add_subplot(1, 3, 2)
    Xb = np.zeros((Nx,3), dtype = float)
    Xb[:,1] = xb
    Xb[:,0] = min_point[0]
    Xb[:,2] = min_point[2]
    ax2.plot(xb, regressor.predict(Xb))
    ax2.set_xlabel(vb)
    ax2.set_title("Fitted means")
    
    
    #ax = fig.add_subplot(1, 3, 3)
    Xc = np.zeros((Nx,3), dtype = float)
    Xc[:,2] = xc
    Xc[:,0] = min_point[0]
    Xc[:,1] = min_point[1]
    ax3.plot(xc, regressor.predict(Xc))
    ax3.set_xlabel(vc)
    
    #ax.show()
    
    #plt.show()
    
    
    xab3 = np.vstack((np.array(np.meshgrid(xa, xb)).reshape(2,-1), np.zeros(Nx**2))).T
    yab = regressor.predict(xab3).reshape((Nx,Nx))
    
    #fig, ax = plt.subplots() #subplot_kw={"projection": "3d"})
    fig = plt.figure(figsize=(9,3))
    
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    X,Y = np.meshgrid(xa, xb)
    surf = ax.plot_surface(X,Y,yab, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    #fig.colorbar(surf, shrink=0.3, aspect=5)
    ax.contour(X, Y, yab, zdir='z', offset=yab.min(), cmap=cm.coolwarm)
    
    plt.xlabel(va)
    plt.ylabel(vb)

    #plt.show()
    
    
    xac3 = np.vstack((np.array(np.meshgrid(xa, xc)).reshape(2,-1), np.zeros(Nx**2))).T
    xac3[:,[1,2]] = xac3[:, [2,1]]
    yac = regressor.predict(xac3).reshape((Nx,Nx))
    
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    X,Y = np.meshgrid(xa, xc)
    surf = ax.plot_surface(X,Y,yac, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    #fig.colorbar(surf, shrink=0.3, aspect=5)
    ax.contour(X, Y, yac, zdir='z', offset=yac.min(), cmap=cm.coolwarm)
    
    plt.xlabel(va)
    plt.ylabel(vc)

    #plt.show()
    
    
    xbc3 = np.vstack((np.array(np.meshgrid(xb, xc)).reshape(2,-1), np.zeros(Nx**2))).T
    xbc3[:,[0,1,2]] = xac3[:, [1,0,2]]
    ybc = regressor.predict(xbc3).reshape((Nx,Nx))
    
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    X,Y = np.meshgrid(xb, xc)
    
    surf = ax.plot_surface(X,Y,ybc, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    #fig.colorbar(surf, shrink=0.3, aspect=5)
    ax.contour(X, Y, ybc, zdir='z', offset=ybc.min(), cmap=cm.coolwarm)
    
    plt.xlabel(vb)
    plt.ylabel(vc)

    plt.show()
    
    
    
    
    
    
    
    
def relabel_defaults(titles, new_names):
    """
    Rename default variable ("x0", "x1", "x2") to 
    variable names from sheet[1:,0]
    **Author**: Audun Skau Hansen, Department of Chemistry, UiO
    """
    #new_names = to_array(sheet)[1:4,0]
    new_titles = []
    for i in titles:
        new_titles.append( i.replace("x0", new_names[0]).replace("x1", new_names[1]).replace("x2", new_names[2]) )
    return new_titles

def minitable(titles, values, sheet):
    """
    Generate a mini-table for displaying inter-variable 
    dependencies as indicated by the model
    Author: Audun Skau Hansen, Department of Chemistry, UiO

    ## Keyword arguments:

    | Argument      | Description |
    | ----------- | ----------- |
    | titles      | default variables      |
    | values   | coefficients        |
    | sheet   | the Box-Behnken sheet        |
    """
    arr = np.zeros((len(titles),2), dtype = object)
    arr[:,0] =relabel_defaults(titles, sheet)
    arr[:,1] = values
    return from_array(arr)


def bbdesign(n_center = 3, randomize = True, sheet = None):
    """
    Returns a Box-Benhken experimental design for 3 variables
    Author: Audun Skau Hansen, Department of Chemistry, UiO
    

    ## Keyword arguments:

    | Argument      | Description |
    | ----------- | ----------- |
    | n_center      |  number of samples in the center     |
    | randomize   | whether or not to randomize the ordering (bool)        |
    | sheet   | heet containing the min/max values of variables        |

    """
    a = np.arange(-1,2)
    A = np.array(np.meshgrid(a,a,a)).reshape(3,-1).T
    A = np.concatenate([A[np.sum(A**2, axis = 1)==2, :], np.zeros((n_center,3))])

    ai = np.arange(len(A))

    if randomize == True:
        # randomize run order
        np.random.shuffle(ai)
        
        
    if sheet is not None:
        # Transform coordinates
        tm = sheet.as_numpy_array()
                 
        for i in range(3):
            A[:,i] = interp1d(np.linspace(-1,1,2),tm[i] )(A[:,i])
        #A = A.dot(tm)
        


    return A[ai, :], ai

def bbsheet(sheet):
    """
    Returns a Box-Behnken sheet for gathering experimental results
    Author: Audun Skau Hansen, Department of Chemistry, UiO
    
    ## Keyword arguments

    sheet = setup from bbsetup
    """
    bd, ai = bbdesign(sheet = sheet)
    #global bb_sheet
    
    #sh = sheet.as_numpy_array()
    #arr = np.zeros(sh.shape + np.array([1,2]), dtype = object)
    
    #print("sh")
    #print(sh)
    #print(bd)
    
    column_headers = ["Run",
                      sheet.row_headers[0],
                      sheet.row_headers[1],
                      sheet.row_headers[2],
                      "Result"
                     ]
    
    row_headers = ["" for i in range(bd.shape[0])]

    
    bb_widget = tablewidget(column_headers,row_headers)
    
    arr = np.zeros( (len(row_headers), len(column_headers)), dtype = float)
    
    arr[:,0] = np.arange(len((row_headers)))+1
    arr[:,0] = ai+1
    arr[:,1:4] = bd
    bb_widget.set_from_array(arr)
    
    
    
    return bb_widget
    

def bbsetup():
    """
    Returns an interactive sheet (ipysheet)
    for setting up a Box-Benkhen design session
    
    **Author**: Audun Skau Hansen, Department of Chemistry, UiO
    """
    global sheet
    arr = np.zeros((4,3), dtype = object)
    arr[0,0] = ""
    arr[1,0] = "Variable A"
    arr[2,0] = "Variable B"
    arr[3,0] = "Variable C"
    arr[0,1] = "Minimum"
    arr[0,2] = "Maximum"
    arr[1:,1] = -1
    arr[1:,2] = 1
    sheet = from_array(arr)
    
    sheet.column_headers = False
    sheet.row_headers = False
    
    return sheet