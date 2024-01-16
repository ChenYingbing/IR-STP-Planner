import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# Define kernel parameters. 
l = 0.1
sigma_f = 2

# Define kernel object. 
kernel = ConstantKernel(constant_value=sigma_f,constant_value_bounds=(1e-3, 1e3)) \
            * RBF(length_scale=l, length_scale_bounds=(1e-3, 1e3))

# Set dimension. 
d = 1
# Number of training points.
n = 1000
# Length of the training set. 
L = 2
# Generate training features.
x = np.linspace(start=0, stop=L, num=n)
X = x.reshape(n, d)
# Error standar deviation. 
sigma_n = 0.4
# Errors.
epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)

# Generate non-linear function.
def f(x):
    f = np.sin((4*np.pi)*x) + np.sin((7*np.pi)*x) + np.sin((3*np.pi)*x) 
    return(f)

f_x = f(x)

# Observed target variable. 
y = f_x + epsilon

# Construct test set
sample_num = 10

n_star = n + 300
x_star = np.linspace(start=0, stop=(L + 0.5), num=n_star)

X_star = x_star.reshape(n_star, d)

# Define GaussianProcessRegressor object
gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2, n_restarts_optimizer=10, )

print("x", x.shape)
print("X", X.shape)
print("y", y.shape)

# Fit to data using Maximum Likelihood Estimation of the parameters.
gp.fit(X, y)
# Make the prediction on test set.
y_pred = gp.predict(X_star)

# Generate samples from posterior distribution. 
y_hat_samples = gp.sample_y(X_star, n_samples=sample_num)
# Compute the mean of the sample. 
y_hat = np.apply_over_axes(func=np.mean, a=y_hat_samples, axes=1).squeeze()
# Compute the standard deviation of the sample. 
y_hat_sd = np.apply_over_axes(func=np.std, a=y_hat_samples, axes=1).squeeze()

print("y_hat_samples", y_hat_samples.shape)
print("y_hat", y_hat.shape)
print("y_hat_sd", y_hat_sd.shape)

ig, ax = plt.subplots(figsize=(15, 8))
# Plot training data.
plt.scatter(x=x, y=y, label='training data')
# Plot "true" linear fit.
plt.plot(x_star, f(x_star), 'r', label='f(x)')

# Plot corridor. 
ax.fill_between(
    x=x_star, 
    y1=(y_hat - 2*y_hat_sd), 
    y2=(y_hat + 2*y_hat_sd), 
    color='green',
    alpha=0.3, 
    label='Credible Interval'
)

# Plot prediction. 
plt.plot(x_star, y_pred, color='g', label='pred')
ax.set(title='Prediction & Credible Interval')
ax.legend(loc='lower left')

plt.show()