import numpy as np
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ipysheet import from_array, to_array, cell

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def visualize_surfaces(sheet, regressor, Nx = 300):
    """
    Visualize response surfaces the regressor model
    Author: Audun Skau Hansen, Department of Chemistry, UiO (2022)
    
    sheet      = Box-Benhken data sheet
    regressor  = sklearn LinearRegression instance
    Nx         = mesh resolution along each axis
    """
    bounds = np.array(to_array(sheet)[1:,1:], dtype = float)
    xa = np.linspace(bounds[0,0], bounds[0,1],Nx)
    xb = np.linspace(bounds[1,0], bounds[1,1],Nx)
    xc = np.linspace(bounds[2,0], bounds[2,1],Nx)



    xab3 = np.vstack((np.array(np.meshgrid(xa, xb)).reshape(2,-1), np.zeros(Nx**2))).T
    yab = regressor.predict(xab3).reshape((Nx,Nx))

    plt.figure(figsize=(9.5,8))
    plt.title(to_array(sheet)[1,0] + " vs " + to_array(sheet)[2,0])
    plt.contourf(xa,xb,yab)
    plt.xlabel(to_array(sheet)[1,0])
    plt.ylabel(to_array(sheet)[2,0])
    plt.colorbar()
    plt.show()
    
    
    
    xac3 = np.vstack((np.array(np.meshgrid(xa, xc)).reshape(2,-1), np.zeros(Nx**2))).T
    yac = regressor.predict(xac3).reshape((Nx,Nx))

    plt.figure(figsize=(9.5,8))
    plt.title(to_array(sheet)[1,0] + " vs " + to_array(sheet)[3,0])
    plt.contourf(xa,xc,yac)
    plt.xlabel(to_array(sheet)[1,0])
    plt.ylabel(to_array(sheet)[3,0])
    plt.colorbar()
    plt.show()
    
    
    
    xbc3 = np.vstack((np.array(np.meshgrid(xb, xc)).reshape(2,-1), np.zeros(Nx**2))).T
    ybc = regressor.predict(xbc3).reshape((Nx,Nx))

    plt.figure(figsize=(9.5,8))
    plt.title(to_array(sheet)[2,0] + " vs " + to_array(sheet)[3,0])
    plt.contourf(xb,xc,ybc)
    plt.xlabel(to_array(sheet)[2,0])
    plt.ylabel(to_array(sheet)[3,0])
    plt.colorbar()
    plt.show()
    
def relabel_defaults(titles, sheet):
    """
    Rename default variable ("x0", "x1", "x2") to 
    variable names from sheet[1:,0]
    Author: Audun Skau Hansen, Department of Chemistry, UiO
    """
    new_names = to_array(sheet)[1:4,0]
    new_titles = []
    for i in titles:
        new_titles.append( i.replace("x0", new_names[0]).replace("x1", new_names[1]).replace("x2", new_names[2]) )
    return new_titles

def minitable(titles, values, sheet):
    """
    Generate a mini-table for displaying inter-variable 
    dependencies as indicated by the model
    Author: Audun Skau Hansen, Department of Chemistry, UiO
    """
    arr = np.zeros((len(titles),2), dtype = object)
    arr[:,0] =relabel_defaults(titles, sheet)
    arr[:,1] = values
    return from_array(arr)


def bbdesign(n_center = 3, randomize = True, sheet = None):
    """
    Returns a Box-Benhken experimental design for 3 variables
    Author: Audun Skau Hansen, Department of Chemistry, UiO
    
    n_center  = number of samples in the center
    randomize = whether or not to randomize the ordering
    sheet     = sheet containing the min/max values of variables
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
        tm = np.array(to_array(sheet)[1:,1:], dtype = float)
                 
        for i in range(3):
            A[:,i] = interp1d(np.linspace(-1,1,2),tm[i] )(A[:,i])
        #A = A.dot(tm)
        


    return A[ai, :] 

def bbsheet(sheet):
    """
    Returns a Box-Behnken sheet for gathering experimental results
    Author: Audun Skau Hansen, Department of Chemistry, UiO
    
    sheet = setup from bbsetup
    """
    bd = bbdesign(sheet = sheet)
    global bb_sheet
    
    sh = to_array(sheet)
    arr = np.zeros(bd.shape + np.array([1,2]), dtype = object)
    arr[0,0] = "Run"
    arr[0,1] = sh[1,0] #variable A
    arr[0,2] = sh[2,0] #variable A
    arr[0,3] = sh[3,0] #variable A
    arr[0,4] = "Result" #variable A
    arr[1:,0] = np.arange(bd.shape[0])+1 # Experiment number
    arr[1:,1:4] = bd
    
    bb_sheet = from_array(arr)
    
    bb_sheet.column_headers = False
    bb_sheet.row_headers = False
    
    return bb_sheet
    

def bbsetup():
    """
    Returns an interactive sheet (ipysheet)
    for setting up a Box-Benkhen design session
    
    Author: Audun Skau Hansen, Department of Chemistry, UiO
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
