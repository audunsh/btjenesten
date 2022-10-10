import numpy as np
from scipy.optimize import minimize
#import btjenesten as bt

from btjenesten.gpr import Regressor

import matplotlib.pyplot as plt
from IPython.display import HTML

def parameter_optimization(x_data, y_data, training_fraction = 0.8, normalize_y = True, params = None, training_subset = None):
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
    if training_subset is None:
        training_subset = np.ones(x_data.shape[0], dtype = bool)
        training_subset[n:] = False
    else:
        if len(training_subset)<len(y_data):
            # assume index element array
            ts = np.zeros(len(y_data), dtype = bool)
            ts[training_subset] = True
            training_subset = ts
            
            
    #print(training_subset)
    #print(x_data[training_subset])
        
    #print(training_subset)
    y_data_n = y_data*1
    if normalize_y:
        y_data_n*=y_data_n.max()**-1
    
    def residual(params, x_data = x_data, y_data=y_data_n, training_subset = training_subset):
        test_subset = np.ones(x_data.shape[0], dtype = bool)
        test_subset[training_subset] = False
        regressor = Regressor(x_data[training_subset] , y_data[training_subset]) 
        regressor.params = 10**params
        energy = np.sum((regressor.predict(x_data[test_subset]) - y_data[test_subset])**2)
        return energy
    
    
    ret = minimize(residual, params)
    #print(ret)
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


def show_3d_sample_box(all_x):
    """
    a "hack" for showing the measurement setup in 3D using the bubblebox module / evince
    Author: Audun
    """
    import bubblebox as bb
    b = bb.mdbox(n_bubbles = all_x.shape[0], size = (4,4,4))
    pos = all_x - np.min(all_x, axis = 0)[None, :]
    pos = pos*np.max(pos, axis = 0)[None, :]**-1
    b.pos = 8*(pos.T - np.ones(3)[:, None]*.5)
    return b

def choose_n_most_distant(all_x, n):
    """
    choose n measurements such that the total distance between the measurements is at a maximum
    Author: Audun
    """
    d = np.sum((all_x[:, None] - all_x[None, :])**2, axis = 2)
    
    total_d = np.sum(d, axis = 0)
    
    return np.argsort(total_d)[-n:]

import numpy as np

def parameter_tuner_3d(all_x, all_y, n):
    """
    Interactive (widget for Jupyter environments) parameter tuner 
    for the gpr module 

    Authors: Audun Skau Hansen and Ayla S. Coder 
    """


    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button



    training_x = all_x[n]
    training_y = all_y[n]
    
    regressor = Regressor(training_x, training_y)


    # The parametrized function to be plotted
    def f(params1, params2, params3):
        regressor.params = 10**np.array([params1, params2, params3])
        return regressor.predict(all_x)



    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    plt.plot(np.arange(len(all_y))[n], all_y[n], "o", markersize = 10, label = "training data", color = (0,0,.5))
    plt.plot(all_y, "o", label = "true values", color = (.9,.2,.4))
    plt.legend()
    line, = plt.plot( f(1,1,1), ".-", lw=1, color = (.9,.9,.2))

    ax.set_xlabel('Time [s]')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.4, bottom=0.25)
    """
    # Make a horizontal slider to control the frequency.
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label='Frequency [Hz]',
        valmin=0.1,
        valmax=30,
        valinit=init_frequency,
    )
    """

    # Make a vertically oriented slider to control the amplitude
    param1 = plt.axes([0.1, 0.3, 0.02, 0.5])
    param_slider1 = Slider(
        ax=param1,
        label="log(P1)",
        valmin=-10,
        valmax=10,
        valinit=1.0,
        orientation="vertical"
    )

    # Make a vertically oriented slider to control the amplitude
    param2 = plt.axes([0.2, 0.3, 0.02, 0.5])
    param_slider2 = Slider(
        ax=param2,
        label="log(P2)",
        valmin=-10,
        valmax=10,
        valinit=1.0,
        orientation="vertical"
    )

    # Make a vertically oriented slider to control the amplitude
    param3 = plt.axes([0.3, 0.3, 0.02, 0.5])
    param_slider3 = Slider(
        ax=param3,
        label="log(P3)",
        valmin=-10,
        valmax=10,
        valinit=1.0,
        orientation="vertical"
    )



    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_ydata( f(param_slider1.val,param_slider2.val,param_slider3.val)) 
        fig.canvas.draw_idle()



    # register the update function with each slider
    #freq_slider.on_changed(update)
    param_slider1.on_changed(update)
    param_slider2.on_changed(update)
    param_slider3.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        param_slider1.reset()
        param_slider2.reset()
        param_slider3.reset()
    
    button.on_clicked(reset)
    
    
 
    plt.show()
    
    
# This function plots all the dimensions chosen in rows of three in the same figure. 
# That way it can be used to visualize experiments where the analysis involves more than three dimensions. 

def one_dimensional_plots(x_data, proposed_new_x, reg, y_labels, x_labels):
    """
    Author: Ayla S. Coder
    """
    import matplotlib.pyplot as plt
    # Where y_labels and x_labels are arrays of strings 
    
    plt.figure(figsize=(8, 5)) # Sets the figure size
    
    rows = int(len(x_data[0])/3) # The amount of rows of plots in the figure
    
    dimensions = len(x_data[0]) # The amount of dimensions evaluated.
    
    # For loop to set up each subplot
    for i in range (dimensions):
        
        # This function is documented on Btjeneste website. If you're curious, run the command help(bt.analysis.data_projection)
        x,y = data_projection(reg, axes = [i], center = proposed_new_x, resolution = 100)
        
        # Sets up the i-th subplot
        ax = plt.subplot(rows, dimensions, i+1)
    
        ax.set_ylabel(y_labels[i], fontsize = 15)
        ax.set_xlabel(x_labels[i], fontsize = 15)
        ax.grid()
       # What is plotted in each plot:
        ax.plot(x[0], y)
        
    plt.show()


def html_table(values, columns = None, rows = None):
    """
    Simple HTML table generator
    Author: Audun Skau Hansen
    """
    ret = """<table>\n"""
    
    nr = values.shape[0]
    nc = values.shape[1]
    
    if columns is not None:
        ret += "<tr>\n"
        ret += "<th></th>\n"
        
        for c in range(len(columns)):
            ret += "<th>%s</th>\n" % columns[c]
        
        ret += "</tr>\n"
        
    for r in range(nr):
        ret += "<tr>\n"
        if columns is not None:
            if rows is not None:
                ret += "<th>%s</th>\n" % rows[r]
            else:
                ret += "<th></th>\n"
        else:
            if rows is not None:
                ret += "<th>%s</th>\n" % rows[r]
            else:
                ret += "<th></th>\n"

        for c in range(nc):
            ret += "<th>%s</th>\n" % values[r, c]
        ret += "</tr>\n"
    ret += """</table>\n"""
    
    return HTML(ret)
    

def regressor_summary(all_x_, all_y_, reg, n, error_margin = None):
    # Figure where the response is plotted against the samples. Figure 1
    plt.figure(figsize = (8,5))
    plt.title("Basic overview of GPR training vs true values", fontsize = "25")

    y_pred = reg.predict(all_x_)
    plt.plot(n, all_y_[n], "o", color = (0,1,0), markersize = 14, label ="training")
    plt.plot(all_y_ , "o-", color = (0,0,0), label = "exact", markersize = 8, linewidth = 5)
    plt.plot(y_pred, "o-", color = (1.0,0,0), label = "prediction", linewidth = 2)

    plt.fill_between(np.arange(len(all_y_)), all_y_ - error_margin, all_y_ + error_margin, label = "margin of error", zorder = -10)

    plt.xlabel("Samples", fontsize = "20")
    plt.ylabel("Response", fontsize = "20")

    plt.legend()

    # # # to add a table to Figure 1:
    columns = ("MgSO4 content", "Acetonitril Ratio", "HCl volume", "Real Response", "Predicted Response")
    rows = ['Sample %d' % x for x in np.arange(1, len(all_y_)+1, 1)]

    n_rows = len(all_y_)
    plt.text(len(all_x_)/3.5, -4e7, "Sample compositions", fontsize = 25, fontweight='bold', color="red")


    # Reshaping (stacking) data for the table
    x = all_x_.T
    stacked = np.vstack([x, all_y_, y_pred]).T

    # Add a table at the bottom of the axes
    """
    the_table = plt.table(cellText= stacked,
                          cellLoc='center',
                          rowLoc='center',
                          rowLabels=rows,
                          colLabels=columns,
                          bbox=[0.0, -2.4, 1.2,2.1]) # table.scale doesn't work with bbox 

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(16)
    """
    plt.show()

    the_table = html_table(stacked, rows = rows, columns = columns )

    return the_table
    