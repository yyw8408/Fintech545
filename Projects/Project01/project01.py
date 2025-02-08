#Problem1
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('Projects/Project01/problem1.csv')
#print(df.head())
#print(df.describe())

mean = np.mean(df['X'])
variance = np.var(df['X'])
print(f"mean: {mean}")
print(f"variance: {variance}")

skewness = stats.skew(df['X'])
kurt = stats.kurtosis(df['X'])
print(f"skewness: {skewness}")
print(f"kurtosis: {kurt}")

#fit the normal distribution and student-t distribution model
mu, sigma = stats.norm.fit(df['X'])
degree_freedom_t, location_t, scale_t = stats.t.fit(df['X'])
print(f"normal distribution:\nmu: {mu}, sigma: {sigma}")
print(f"student-t distribution:\ndegree of freedom: {degree_freedom_t}, location: {location_t}, scale: {scale_t}")

x = np.linspace(min(df['X']), max(df['X']), 1000)
pdf_normal = stats.norm.pdf(x, mu, sigma) 
pdf_t = stats.t.pdf(x, degree_freedom_t, location_t, scale_t) 

#data visiualization
plt.figure(figsize=(10, 6))
plt.hist(df['X'], bins=30, density=True, alpha=0.6, color='gray', label="df")
plt.plot(x, pdf_normal, 'r-', label="normal distribution")
plt.plot(x, pdf_t, 'b-', label="student-t distribution")
plt.legend()
plt.show()

#calculate maximum likehood
log_ml_norm = np.sum(stats.norm.logpdf(df['X'], mu, sigma))
log_ml_t = np.sum(stats.t.logpdf(df['X'], degree_freedom_t, location_t, scale_t))
print(f"normal distribution maximum likehood: {log_ml_norm}")
print(f"student-t distribution maximum likehood: {log_ml_t}")

n = len(df['X'])
AIC_norm = 2*2 - log_ml_norm
AIC_t = 2*3 - log_ml_t
BIC_norm = 2*np.log(n) - 2*log_ml_norm
BIC_t = 3*np.log(n) - 2*log_ml_t
print(f"normal distribution:\nAIC: {AIC_norm}, BIC: {BIC_norm}")
print(f"student-t distribution:\nAIC: {AIC_t}, BIC: {BIC_t}")

#Problem2
import numpy as np
import pandas as pd
from scipy import stats
df2 = pd.read_csv("Projects/Project01/problem2.csv")
#print(df2.head())
cov_matrix = df2.cov()
print(cov_matrix)

#semi-definite examination
eigenvalues = np.linalg.eigvalsh(cov_matrix)
print(eigenvalues)

#find the nearest PSD
def near_PSD(A, epsilon = 0.0):
    n = A.shape[0]
    out = A.copy()
    invSD = None
    #transform into corr-matrix
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0/np.sqrt((np.maximum(1e-8,np.diag(out)))))
        out = invSD @ out @ invSD
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon + 1e-8)
    denom = np.abs(vecs @ vecs @ vals)
    denom[denom == 0] = 1e-8
    T = 1 / denom
    T = np.diag(np.sqrt(T))
    L = np.diag(np.sqrt(vals))
    B = T @ vecs @ L
    out = B @ np.transpose(B)
    #Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out
    
near_pairwise = near_PSD(cov_matrix)
print(near_pairwise)
    
#calculate cov matrix using overlapping data
overlap_df = df2.dropna()
overlap_cov_matrix = overlap_df.cov()
print(overlap_cov_matrix)

overlap_eigenvalues = np.linalg.eigvalsh(overlap_cov_matrix)
print(overlap_eigenvalues)
diff_norm = np.linalg.norm(near_pairwise - overlap_cov_matrix, 'fro')
print(f"Frobenius Norm of Difference: {diff_norm}")

eigenvalues1 = np.linalg.eigvalsh(near_pairwise)
eigenvalues2 = np.linalg.eigvalsh(overlap_cov_matrix)
print(eigenvalues1)
print(eigenvalues2)

#Problem3
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


#method 1: based on conditional distribution formula
df3 = pd.read_csv('Projects/Project01/problem3.csv')
mu = df3.mean().values
sigma = df3.cov().values
print(mu)
print(sigma)

mu_X1 = mu[0]
mu_X2 = mu[1]
sigma_11 = sigma[0, 0]
sigma_22 = sigma[1, 1]
sigma_12 = sigma[1, 0]

X1 = 0.6
mu_X21 = mu_X2 + (sigma_12/sigma_11) * (X1 - mu_X1)
sigma_X21 = np.sqrt(sigma_22 - (sigma_12 ** 2)/sigma_11)
print(f"Conditional distribution: X2|X1=0.6 ~ N({mu_X21}, {sigma_X21})")

#method 2: linear regression
import statsmodels.api as sm
y = df3['x2'].values
X = df3['x1'].values
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()

alpha, beta = model.params
mu_X21_LR = alpha + beta * X1
residuals = y - model.predict(X)
sigma_X21_LR = np.var(residuals)
print(f"Linear Regression: X2|X1=0.6 ~ N({mu_X21_LR}, {sigma_X21_LR})")

#calculate cholesky root
L = np.linalg.cholesky(sigma)
Z1 = np.random.normal(0,1,10000)
Z2 = np.random.normal(0,1,10000)
X1 = mu_X1 + L[0,0]*Z1
X2 = mu_X2 + L[1,0]*Z1 + L[1,1]*Z2

X2_given_X1 = mu_X21 + np.sqrt(sigma_X21) * Z2
simulate_mean = np.mean(X2_given_X1)
simulate_var = np.var(X2_given_X1)
print(f"simulated X2|X1=0.6 ~ N({simulate_mean}, {simulate_var})")


#Problem4
import statsmodels as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.api as smt

df4 = pd.read_csv('Projects/Project01/problem4.csv')

def tsplot(y, lags=None, figsize=(15, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)


        plt.tight_layout()
    return 

#MA(1)
MA1 = ARIMA(df4, order=(0,0,1))
MA1_model = MA1.fit()
print(MA1_model.summary())
tsplot(MA1_model.fittedvalues,lags=10)
plt.show()

#MA(2)
MA2 = ARIMA(df4, order=(0,0,2))
MA2_model = MA2.fit()
print(MA2_model.summary())
tsplot(MA2_model.fittedvalues,lags=10)
plt.show()

#MA(3)
MA3 = ARIMA(df4, order=(0,0,3))
MA3_model = MA3.fit()
print(MA3_model.summary())
tsplot(MA3_model.fittedvalues,lags=10)
plt.show()

#AR(1)
AR1 = ARIMA(df4, order=(1,0,0))
AR1_model = AR1.fit()
print(AR1_model.summary())
tsplot(AR1_model.fittedvalues,lags=10)
plt.show()

#AR(2)
AR2 = ARIMA(df4, order=(2,0,0))
AR2_model = AR2.fit()
print(AR2_model.summary())
tsplot(AR2_model.fittedvalues,lags=10)
plt.show()

#AR(3)
AR3 = ARIMA(df4, order=(3,0,0))
AR3_model = AR3.fit()
print(AR3_model.summary())
tsplot(AR3_model.fittedvalues,lags=10)
plt.show()

tsplot(df4['y'],lags=10)
plt.show()

import pmdarima as pm

arma_model = pm.auto_arima(df4,
                      start_p=0, max_p=9, 
                      start_q=0, max_q=9,
                      d=0, 
                      seasonal=False, 
                      information_criterion='aic', 
                      trace=True, 
                      stepwise=True) 

print(arma_model.summary())


#Problem5
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#calculate exp covariance matrix
def exp_weighted_cov_matrix(X: np.ndarray, lam: float):
    n, d = X.shape
    w = np.array([(1 - lam) * lam ** i for i in range(n)])
    w = w / w.sum()

    mu = np.sum(X * w[:, None], axis=0)
    S = np.zeros((d, d))
    for i in range(n):
        diff = X[i] - mu
        S += w[i] * np.outer(diff, diff)

    return S

df5 = pd.read_csv("Projects/Project01/DailyReturn.csv",skiprows=0)
df5_numeric = df5.select_dtypes(include=[np.number])
df5_numpy = df5_numeric.to_numpy()
#lam = 0.75
#cov_matrix = exp_weighted_cov_matrix(df5_numpy, lam)
#print(cov_matrix)

#divide lambda into 10 values
lambda_values = np.linspace(0.01, 0.99, 10)

explained_variances = {}

#simulate pca for each lambda
for lam in lambda_values:
    cov_matrix = exp_weighted_cov_matrix(df5_numpy, lam)
    
    pca = PCA()
    pca.fit(cov_matrix)

    print(f"λ={lam:.2f}, PCA Explain variance:", pca.explained_variance_ratio_)
    
    if len(pca.explained_variance_ratio_) > 0:
        explained_variances[lam] = np.cumsum(pca.explained_variance_ratio_)

if explained_variances:
    plt.figure(figsize=(10, 6))
    for lam, var in explained_variances.items():
        if len(var) > 0:
            plt.plot(range(1, len(var) + 1), var, label=f"λ={lam:.2f}")

    
    plt.title("cumulative weights")
    plt.legend()
    plt.grid()
    plt.show()


#Problem6
from scipy.linalg import cholesky
from sklearn.decomposition import PCA
import time

covariance_matrix = pd.read_csv("Projects/Project01/problem6.csv", skiprows=1,header=None, dtype=float).values
print(covariance_matrix)
mean_vector = np.zeros(covariance_matrix.shape[0])

#simulate 10000 draws using cholesky method
num_samples = 10000
random_normals = np.random.randn(num_samples, covariance_matrix.shape[0])
start_time = time.time()

def cholesky_root(a):
    n = a.shape[0]
    L = np.zeros_like(a)
    for j in range(n):
        s = 0.0
        if j > 1:
            s = np.dot(L[j,:j], L[j,:j])
        L[j,j] = np.sqrt(a[j,j] - s)
    for i in range(j+1,n):
        s = np.dot(L[i, :j], L[j, :j])
        L[i,j] = (a[i,j] - s)/L[j,j]
    
    return L

#cholesky factorization
L = cholesky_root(covariance_matrix)
print(L.shape)
print(random_normals.shape)
samples_cholesky = random_normals @ np.transpose(L) + mean_vector
end_time = time.time()
cholesky_time_cost = end_time - start_time

def simulate_pca(a, num_sim, nval=None):
    vals, vecs = np.linalg.eigh(a)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    total_variance = np.sum(vals)

    posv = vals >= 1e-8
    vals = vals[posv]
    vecs = vecs[:, posv]
    
    if nval is not None:
        vals = vals[:nval]
        vecs = vecs[:, :nval]

    print(f"Simulating with {len(vals)} PC Factors: {np.sum(vals) / total_variance * 100:.2f}% total variance explained")

    B = vecs @ np.diag(np.sqrt(vals))

    r = np.random.randn(len(vals), num_sim)
    return (B @ r).T

#calculate time spend
start_time2 = time.time()
samples_pca = simulate_pca(covariance_matrix, num_samples)
end_time2 = time.time()
pca_time = end_time2 - start_time2

cov_cholesky = np.cov(samples_cholesky, rowvar=False)

cov_pca = np.cov(samples_pca, rowvar=False)

print(f"Cholesky simulation time: {cholesky_time_cost:.4f} seconds")
print(f"PCA simulation time: {pca_time:.4f} seconds")

frobenius_cholesky = np.linalg.norm(covariance_matrix - cov_cholesky, 'fro')
frobenius_pca = np.linalg.norm(covariance_matrix - cov_pca, 'fro')
print(f"Frobenius norm (Cholesky method): {frobenius_cholesky:.4f}")
print(f"Frobenius norm (PCA method): {frobenius_pca:.4f}")


eigenvalues_original = np.linalg.eigvalsh(covariance_matrix)[::-1]
cumulative_variance_original = np.cumsum(eigenvalues_original) / np.sum(eigenvalues_original)

eigenvalues_pca = np.linalg.eigvalsh(cov_pca)[::-1]
eigenvalues_cholesky = np.linalg.eigvalsh(cov_cholesky)[::-1]
cumulative_variance_pca = np.cumsum(eigenvalues_pca) / np.sum(eigenvalues_pca)
cumulative_variance_cholesky = np.cumsum(eigenvalues_cholesky) / np.sum(eigenvalues_cholesky)

print(f"Cumulative variance explained (original): {cumulative_variance_original[:10]}")
print(f"Cumulative variance explained (PCA): {cumulative_variance_pca[:10]}")
print(f"Cumulative variance explained (Cholesky): {cumulative_variance_cholesky[:10]}")

