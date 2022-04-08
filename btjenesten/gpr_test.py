import numpy as np
import gpr as gpr 
import kernels as knls 
import matplotlib.pyplot as plt

constant = knls.Constant
crazy = knls.Funny_trigonometric
rbf = knls.RBF

f = lambda x : np.exp(x) + x

x_train = np.linspace(0, 10, 11)
y_train = f(x_train)

regressor = gpr.Regressor(x_train, y_train, rbf)

x_val = np.linspace(0, 10, 110)
y_val = regressor.predict(x_val)

print(regressor.score(x_val, f(x_val)))

plt.plot(x_val, y_val, label="Predicted values")
plt.scatter(x_train, y_train, label="Training points", color="red")
plt.plot(x_val, f(x_val), label="True values", color="green", zorder=0.5, linewidth=5, alpha=0.3)
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
