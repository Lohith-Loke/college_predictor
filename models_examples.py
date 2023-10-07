
#! **Linear Regression:**
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Fit linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Make predictions
y_pred = lin_reg.predict(X)


#! **Logistic Regression:**

import numpy as np
from sklearn.linear_model import LogisticRegression

# Generate random data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Fit logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Make predictions
y_pred = log_reg.predict(X)


#! **Ridge Regression:**

import numpy as np
from sklearn.linear_model import Ridge

# Generate random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Fit ridge regression model
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y)

# Make predictions
y_pred = ridge_reg.predict(X)


#! **Lasso Regression:**

import numpy as np
from sklearn.linear_model import Lasso

# Generate random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Fit lasso regression model
lasso_reg = Lasso(alpha=1)
lasso_reg.fit(X, y)

# Make predictions
y_pred = lasso_reg.predict(X)


#! **Polynomial Regression:**

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + X**2 + np.random.rand(100, 1)

# Transform input data to polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Fit linear regression model on polynomial features
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Make predictions
y_pred = lin_reg.predict(X_poly)


#! **Bayesian Linear Regression:**

import numpy as np
from sklearn.linear_model import BayesianRidge

# Generate random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Fit Bayesian ridge regression model
bayes_ridge_reg = BayesianRidge()
bayes_ridge_reg.fit(X, y)

# Make predictions
y_pred = bayes_ridge_reg.predict(X)
