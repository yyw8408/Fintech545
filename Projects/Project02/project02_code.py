import pandas as pd
import numpy as np

#Problem1
data = pd.read_csv("Projects/Project02/DailyPrices.csv")
data.drop(columns=['Date'], inplace=True)
data.head()
#A
data1 = data[['SPY','AAPL','EQIX']]
data1_returns = data1.pct_change()
data1_returns = data1_returns.dropna()
data1_arithmetic = data1_returns - data1_returns.mean()
data1_arithmetic_std = data1_arithmetic.std()
data1_arithmetic.tail(5)
print(data1_arithmetic_std)
#B
data1_log_returns = np.log(data1/data1.shift(1))
data1_log_returns = data1_log_returns.dropna()
data1_log_returns_zero_mean = data1_log_returns - data1_log_returns.mean()
data1_log_returns_std = data1_log_returns_zero_mean.std()
data1_log_returns_zero_mean.tail(5)
print(data1_log_returns_std)

#Problem2
#A
import pandas as pd
import numpy as np
from scipy.stats import t, norm, multivariate_normal

data = pd.read_csv("Projects/Project02/DailyPrices.csv")
data.drop(columns=['Date'], inplace=True)
data1 = data[['SPY','AAPL','EQIX']]
data1_returns = data1.pct_change().dropna()

current_prices = data1.iloc[-1]
print(current_prices)
holdings = {"SPY": 100, "AAPL": 200, "EQIX": 150}

tickers = list(set(data1.columns) & set(holdings.keys()))
current_prices = current_prices[tickers]
data1_returns = data1_returns[tickers].dropna()

PV = 0.0
delta = np.zeros(len(tickers))

for i,j in enumerate(tickers):
    PV += current_prices[j] * holdings[j]
    delta[i] = current_prices[j] * holdings[j]
    
delta = delta / PV
print(PV)
print(delta)

#B.a
from code_library import compute_ew_cov
ewma_cov = compute_ew_cov(data1_returns, 0.97)

p_sigma = np.sqrt(delta.T @ ewma_cov.values @ delta)
VAR_ewma = -PV * norm.ppf(0.05) * p_sigma
print(VAR_ewma)

ES_normal = PV * (norm.pdf(norm.ppf(0.05)) / 0.05) * p_sigma
print(ES_normal)

alpha = 0.05
z_alpha = norm.ppf(alpha)
pdf_z_alpha = norm.pdf(z_alpha)

individual_VAR_ewma = {}
individual_ES_ewma = {}
for i, stock in enumerate(tickers):
    sigma_i = np.sqrt(ewma_cov.iloc[i, i])
    stock_value = holdings[stock] * current_prices[stock]
    individual_VAR_ewma[stock] = -stock_value * z_alpha * sigma_i
    individual_ES_ewma[stock] = stock_value * sigma_i * pdf_z_alpha / alpha

for stock in tickers:
    print(f"{stock} - VaR: {individual_VAR_ewma[stock]:.2f}, ES: {individual_ES_ewma[stock]:.2f}")

#B.b
params = {stock: t.fit(data1_returns.iloc[:, i].values) for i, stock in enumerate(tickers)}
t_dists = {stock: t(df=params[stock][0], loc=params[stock][1], scale=params[stock][2]) for stock in tickers}

U = np.column_stack([t_dists[stock].cdf(data1_returns.iloc[:,i].values) for i, stock in enumerate(tickers)])
Z = norm.ppf(U)
R_pearson = np.corrcoef(Z, rowvar=False)

num_simulate = 10000
copula_pearson = multivariate_normal(mean=np.zeros(len(tickers)), cov=R_pearson)
pearson_samples = copula_pearson.rvs(num_simulate)
for i,t in enumerate(tickers):
    pearson_samples[:,i] = t_dists[t].ppf(norm.cdf(pearson_samples[:,i]))
print(pearson_samples)
portfolio_returns_sim = pearson_samples @ delta
VAR_copula = -np.percentile(portfolio_returns_sim * PV, 5)
print(VAR_copula)
ES_t_copula = -np.mean(portfolio_returns_sim[portfolio_returns_sim * PV <= -VAR_copula]) * PV
print(ES_t_copula)

individual_VAR_copula = {}
individual_ES_copula = {}
for i, stock in enumerate(tickers):
    stock_sim_returns = pearson_samples[:, i]
    stock_losses = -stock_sim_returns * holdings[stock] * current_prices[stock]
    individual_VAR_copula[stock] = np.percentile(stock_losses, 100 * (1 - alpha))
    individual_ES_copula[stock] = stock_losses[stock_losses >= individual_VAR_copula[stock]].mean()

for stock in tickers:
    print(f"{stock} - VaR: {individual_VAR_copula[stock]:.2f}, ES: {individual_ES_copula[stock]:.2f}")

#B.c
portfolio_losses = data1_returns @ delta * PV
VAR_historical = -np.percentile(portfolio_losses, 5)
ES_historical = -portfolio_losses[portfolio_losses <= -VAR_historical].mean()

print(VAR_historical)
print(ES_historical)


individual_VAR_historical = {}
individual_ES_historical = {}
for stock in tickers:
    stock_losses_hist = -data1_returns[stock] * holdings[stock] * current_prices[stock]
    individual_VAR_historical[stock] = np.percentile(stock_losses_hist, 100 * (1 - alpha))
    individual_ES_historical[stock] = stock_losses_hist[stock_losses_hist >= individual_VAR_historical[stock]].mean()

for stock in tickers:
    print(f"{stock} - VaR: {individual_VAR_historical[stock]:.2f}, ES: {individual_ES_historical[stock]:.2f}")

#Problem3
import pandas as pd
import numpy as np
from scipy.stats import t, norm, multivariate_normal

def bs_price(S, X, T, sigma, r, option_type = "C"):
    d1 = (np.log(S/X) + (r + (sigma**2)/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "C":
        price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "P":
        price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def cal_iv(S, X, T, price, r, option_type = "C", epsilon = 1e-6, max_iter = 10000):
    sigma_low = 0.01
    sigma_high = 2.0
    for i in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2.0
        price_mid = bs_price(S, X, T, sigma_mid, r)
        if abs(price_mid - price) < epsilon:
            return sigma_mid
        if price_mid < price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    return sigma_mid

#A
call_price = 3.0
S = 31.0
X = 30.0
r = 0.1
T = 0.25
iv = cal_iv(S, X, T, call_price, r)
print(iv)

#B
d1 = (np.log(S/X) + (r + (iv**2)/2) * T) / (iv * np.sqrt(T))
d2 = d1 - iv * np.sqrt(T)
delta = norm.cdf(d1)
vega = S * np.sqrt(T) * norm.pdf(d1)
theta = (-S * norm.pdf(d1) * iv / (2*np.sqrt(T)) - r*X*np.exp(-r*T)*norm.cdf(d2))
print("delta: ", delta, "\nvega: ", vega, "\ntheta: ", theta)

#C
put_price = bs_price(S, X, T, iv, r, option_type="P")
print(put_price)

rhs = S + put_price
lhs = call_price + X * np.exp(-r * T)
print(lhs, rhs)

#D.d
holding_days = 20
trading_days = 255
dt = holding_days / trading_days
delta_put = norm.cdf(d1) - 1 
vega_put = S * np.sqrt(T) * norm.pdf(d1)
theta_put = (-(S * norm.pdf(d1) * iv) / (2 * np.sqrt(T))) + r * X * np.exp(-r*T) * norm.cdf(-d2)
delta_portfolio = delta + delta_put + 1
vega_portfolio = vega + vega_put
theta_portfolio = theta + theta_put
print(delta_portfolio)
print(theta_portfolio)
PV = call_price + put_price + S
sigma = 0.25 * np.sqrt(holding_days/trading_days)

VAR_delta_normal = -(delta_portfolio * PV * sigma * norm.ppf(0.05) + theta_portfolio * (holding_days/trading_days))
ES_delta_normal = delta_portfolio * PV * sigma  * norm.pdf(norm.ppf(0.05)) / 0.05 + theta_portfolio * holding_days / trading_days

print(VAR_delta_normal)
print(ES_delta_normal)


#D.e
n = 10000

np.random.seed(42)
S_T = S * np.exp((r - 0.5 * 0.25**2)*dt + 0.25 * np.sqrt(dt) * np.random.randn(n))
time_to_expire = T - dt

call_prices_future = bs_price(S_T, X, time_to_expire, iv, r, option_type="C")
put_prices_future = bs_price(S_T, X, time_to_expire, iv, r, option_type="P")

FV = call_prices_future + put_prices_future + S_T
loss = PV - FV

sorted_loss = np.sort(loss)
index_VaR = int(0.05 * n)
VaR_MC = -sorted_loss[index_VaR]
ES_MC = -np.mean(sorted_loss[:index_VaR])

print(VaR_MC)
print(ES_MC)

#E
import matplotlib.pyplot as plt

stock_prices = np.linspace(20, 40, 100)
portfolio_values_delta = PV + delta_portfolio * (stock_prices - S)
portfolio_values_mc = stock_prices + bs_price(stock_prices, X, T, iv, r, option_type="C") + bs_price(stock_prices, X, T, iv, r, option_type="P")

plt.figure(figsize=(10, 6))
plt.plot(stock_prices, portfolio_values_mc, label='Monte Carlo (Non-linear)', linewidth=2)
plt.plot(stock_prices, portfolio_values_delta, label='Delta-Normal Approximation (Linear)', linestyle='--', linewidth=2)

plt.axvline(S, color='grey', linestyle=':', label='Current Stock Price')
plt.axhline(PV, color='grey', linestyle=':')

plt.xlabel('Stock Price')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value vs Stock Price: Delta-Normal vs Monte Carlo')
plt.legend()
plt.grid(True)
plt.show()