import pandas as pd
import numpy as np
import statsmodels.api as sm

#Part1

df_portfolio = pd.read_csv('Projects/Final project/initial_portfolio.csv')
df_rf = pd.read_csv('Projects/Final project/rf.csv')
df_prices = pd.read_csv('Projects/Final project/DailyPrices.csv')

df_prices['Date'] = pd.to_datetime(df_prices['Date'])
df_rf['Date'] = pd.to_datetime(df_rf['Date'])
df_prices.set_index('Date', inplace=True)
df_rf.set_index('Date', inplace=True)

df_returns = df_prices.pct_change().dropna()

train_end = pd.Timestamp('2023-12-31')
train_returns = df_returns[df_returns.index <= train_end]
hold_returns = df_returns[df_returns.index > train_end]

train_rf = df_rf[df_rf.index <= train_end]
hold_rf = df_rf[df_rf.index > train_end]

def culmulative_returns(returns):
    return np.prod(1 + returns) - 1

def carino_k_attribution(weights, returns, rf_rates):
    port_daily_return = (weights * returns).sum(axis=1)
    port_excess_return = port_daily_return - rf_rates
    
    cumulative_arith_return = np.prod(1 + port_excess_return) - 1
    cumulative_geom_return = np.log(1 + cumulative_arith_return)
    
    k = cumulative_geom_return / cumulative_arith_return if cumulative_arith_return != 0 else 1
    
    daily_k = np.zeros_like(port_excess_return)
    non_zero_mask = port_excess_return != 0
    daily_k[non_zero_mask] = np.log(1 + port_excess_return[non_zero_mask]) / (k * port_excess_return[non_zero_mask])
    daily_k[~non_zero_mask] = 11
    
    contribution = {}
    for ticker in returns.columns:
        ticker_excess = returns[ticker] - rf_rates
        ticker_contrib = weights[ticker] * ticker_excess * daily_k
        contribution[ticker] = ticker_contrib.sum()
    
    return contribution, cumulative_arith_return


tickers = df_portfolio['Symbol'].unique()
capm_params = {}

for ticker in tickers:
    if ticker == "SPY":
        capm_params[ticker] = {'alpha': 0, 'beta': 1, 'residuals': 0}
        continue
    
    stock_train = train_returns[ticker].dropna()
    df_temp = pd.DataFrame({
        'Stock': stock_train,
        'market': train_returns['SPY'],
        'rf': train_rf['rf']
    }).dropna()
    
    df_temp['excess_stock'] = df_temp['Stock'] - df_temp['rf']
    df_temp['excess_market'] = df_temp['market'] - df_temp['rf']
    
    X = sm.add_constant(df_temp['excess_market'])  # Add constant for intercept
    y = df_temp['excess_stock']
    model = sm.OLS(y, X).fit()  # Ordinary Least Squares regression
    
    capm_params[ticker] = {
        'alpha': model.params['const'],  # Intercept
        'beta': model.params['excess_market'],   # Slope (beta)
        'residuals': model.mse_resid    # Residuals
    }

df_capm = pd.DataFrame.from_dict(capm_params, orient='index')
print(df_capm)
    
market_excess_hold = hold_returns['SPY'] - hold_rf['rf']
market_culmulative_excess = culmulative_returns(market_excess_hold)

results = []
for ticker in tickers:
    stock_excess_hold = hold_returns[ticker] - hold_rf['rf']
    stock_cumulative_excess = culmulative_returns(stock_excess_hold)
    
    if ticker == "SPY":
        sys_return = market_culmulative_excess
        idio_return = 0
        sys_std = np.std(stock_excess_hold)  # Variance of the stock excess returns
        idio_std = 0
    else:
        beta = capm_params[ticker]['beta']
        sys_return = beta * market_culmulative_excess
        idio_return = stock_cumulative_excess - sys_return
        sys_std = beta * np.std(market_excess_hold)  # Variance of the systematic component
        idio_std = np.std(stock_excess_hold - beta * market_excess_hold)  # Variance of the idiosyncratic component
    
    results.append({
        'Symbol': ticker,
        'cumulative_excess_return': stock_cumulative_excess,
        'systematic_return': sys_return,
        'idiosyncratic_return': idio_return,
        'systematic_std': sys_std,
        'idiosyncratic_std': idio_std,
        'beta': capm_params[ticker]['beta']
    })
    
df_result = pd.DataFrame(results)
print(df_result)

hold_prices = df_prices.loc[hold_returns.index]
portfolios = df_portfolio['Portfolio'].unique()
portfolio_weights = {}
portfolio_returns = {}

for portfolio in portfolios:
    df_temp = df_portfolio[df_portfolio['Portfolio'] == portfolio].copy()
    holdings = df_temp.set_index('Symbol')['Holding']
    symbols = holdings.index.tolist()
    
    df_port_prices = hold_prices[symbols].copy()
    
    df_market_value = df_port_prices.multiply(holdings, axis='columns')
    portfolio_value = df_market_value.sum(axis=1)
    
    dynamic_weights = df_market_value.div(portfolio_value, axis=0)
    portfolio_weights[portfolio] = dynamic_weights
    
    df_port_returns = hold_returns[symbols]
    daily_port_return = (dynamic_weights * df_port_returns).sum(axis=1)
    portfolio_returns[portfolio] = daily_port_return
    cum_port_return = np.prod(1 + daily_port_return) - 1
    
portfolio_results = []  
for portfolio in portfolios:
    dynamic_weights = portfolio_weights[portfolio]
    symbols = dynamic_weights.columns
    
    average_weight = dynamic_weights.mean(axis=0)
    
    port_returns = hold_returns[symbols]
    
    return_contrib, cum_excess = carino_k_attribution(
        dynamic_weights, port_returns, hold_rf['rf'].values
    )
    
    df_port = pd.DataFrame({
        'Symbol': list(symbols),
        'Weight': average_weight.values,
        'Return_Contrib': [return_contrib[t] for t in symbols]
    })
    
    df_merged = pd.merge(df_port, df_result, on='Symbol', how='left')

    port_sys_return = np.sum(df_merged['Weight'] * df_merged['systematic_return'])
    port_idio_return = np.sum(df_merged['Weight'] * df_merged['idiosyncratic_return'])
    
    port_beta = np.sum(df_merged['Weight'] * df_merged['beta'])
    
    port_sys_std = port_beta * np.std(market_excess_hold)

    port_daily_return = portfolio_returns[portfolio]
    port_excess_return = port_daily_return - hold_rf['rf'].values
    
    port_total_std = np.std(port_excess_return)

    port_idio_var = max(0, port_total_std**2 - port_sys_std**2)
    port_idio_std = np.sqrt(port_idio_var)
    
    risk_attributions = {}
    for ticker in symbols:
        try:
            weighted_return = dynamic_weights[ticker] * hold_returns[ticker]
            
            X = sm.add_constant(port_daily_return)
            model = sm.OLS(weighted_return, X).fit()
            
            if 'const' in model.params.index:
                beta_coef = model.params.get('x1', model.params.iloc[1] if len(model.params) > 1 else 0)
            else:
                beta_coef = model.params.iloc[1] if len(model.params) > 1 else model.params.iloc[0]
            
            risk_contrib = beta_coef * port_total_std
            risk_attributions[ticker] = risk_contrib
            
        except Exception as e:
            weight = average_weight[ticker]
            stock_std = df_merged.loc[df_merged['Symbol'] == ticker, 'total_std'].values[0]
            risk_attributions[ticker] = weight * stock_std
    
    portfolio_results.append({
        'Portfolio': portfolio,
        'cumulative_excess_return': cum_excess,
        'systematic_return': port_sys_return,
        'idiosyncratic_return': port_idio_return,
        'systematic_std': port_sys_std,
        'idiosyncratic_std': port_idio_std,
        'total_std': port_total_std,
        'beta': port_beta
    })

df_portfolio_results = pd.DataFrame(portfolio_results)
df_risk_attributions = pd.DataFrame([risk_attributions]).T
print(df_portfolio_results)
print(df_risk_attributions)

df_total = df_portfolio.groupby('Symbol').agg({'Holding': 'sum'}).reset_index()
tickers_total = df_total['Symbol'].tolist()
df_total_prices = hold_prices[tickers_total].copy()
total_holdings = df_total.set_index('Symbol')['Holding']


df_total_value = df_total_prices.multiply(total_holdings, axis='columns')
df_portfolio_value = df_total_value.sum(axis=1)
dynamic_weights_total = df_total_value.div(df_portfolio_value, axis=0)
total_daily_return = (dynamic_weights_total * hold_returns[tickers_total]).sum(axis=1)
total_excess_return = total_daily_return - hold_rf['rf'].values

total_contrib, total_cum_excess = carino_k_attribution(
    dynamic_weights_total, hold_returns[tickers_total], hold_rf['rf'].values
)

average_weight_total = dynamic_weights_total.mean(axis=0).reset_index()
average_weight_total.columns = ['Symbol', 'Weight']

df_total = pd.merge(df_total, average_weight_total, on='Symbol', how='left')
df_total = pd.merge(df_total, df_result, on='Symbol', how='left')
print(df_total)

total_cum_excess = np.sum(df_total['Weight'] * df_total['cumulative_excess_return'])
total_sys_return = np.sum(df_total['Weight'] * df_total['systematic_return'])
total_idio_return = np.sum(df_total['Weight'] * df_total['idiosyncratic_return'])
total_beta = np.sum(df_total['Weight'] * df_total['beta'])
total_sys_std = total_beta * np.std(market_excess_hold)
total_std = np.std(total_excess_return)
total_idio_std = np.std(total_idio_var = max(0, total_std**2 - total_sys_std**2))


df_total_results = pd.DataFrame([{
    'Portfolio': 'Total',
    'cumulative_excess_return': total_cum_excess,
    'systematic_return': total_sys_return,
    'idiosyncratic_return': total_idio_return,
    'systematic_std': total_sys_std,
    'idiosyncratic_std': total_idio_std,
    'total_std': total_std,
    'beta': total_beta
}])

print(df_total_results)


#Part2
from scipy.optimize import minimize
average_market_return = train_returns['SPY'].mean() 
average_rf = train_rf['rf'].mean()
average_excess_market_return = average_market_return - average_rf

expected_returns = {}
for ticker in tickers:
    if ticker in capm_params:
        expected_returns[ticker] = average_rf + capm_params[ticker]['beta'] * average_excess_market_return
cov_matrix = np.cov(train_returns[tickers])

optimal_portfolio = {}
for portfolio in portfolios:
    portfolio_tickers = df_portfolio[df_portfolio['Portfolio'] == portfolio]['Symbol'].unique()
    portfolio_expected_returns = np.array([expected_returns[ticker] for ticker in portfolio_tickers])
    portfolio_cov_matrix = np.cov(train_returns[portfolio_tickers].T)
    
    def sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return -(portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)

    num_assets = len(portfolio_tickers)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    initial_weights = np.ones(num_assets) / num_assets

    result = minimize(sharpe_ratio, initial_weights, 
                        args=(portfolio_expected_returns, portfolio_cov_matrix, average_rf),
                        method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result['x']
    optimal_portfolio[portfolio] = {
            'tickers': portfolio_tickers,
            'weights': dict(zip(portfolio_tickers, optimal_weights))
        }
    
    for ticker, weight in zip(portfolio_tickers, optimal_weights):
        print(f"  {ticker}: {weight:.4f}")
all_tickers = set()
for portfolio in portfolios:
    all_tickers.update(optimal_portfolio[portfolio]['tickers'])

all_tickers = list(all_tickers)
all_expected_returns = np.array([expected_returns[ticker] for ticker in all_tickers])
all_cov_matrix = np.cov(train_returns[all_tickers].T)

num_all_assets = len(all_tickers)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_all_assets))
initial_weights_all = np.ones(num_all_assets) / num_all_assets

result_all = minimize(sharpe_ratio, initial_weights_all,args=(all_expected_returns, all_cov_matrix, average_rf),
                      method='SLSQP', bounds=bounds, constraints=constraints)

all_optimal_weights = result_all['x']
total_optimal_weights = dict(zip(all_tickers, all_optimal_weights))

for ticker, weight in total_optimal_weights.items():
    print(f"  {ticker}: {weight:.4f}")

optimal_portfolio_results = []
for portfolio in portfolios:
    optimal_weights_dict = optimal_portfolio[portfolio]['weights']
    portfolio_tickers = optimal_portfolio[portfolio]['tickers']
    
    port_beta = np.sum([optimal_weights_dict[ticker] * capm_params[ticker]['beta'] for ticker in portfolio_tickers])
    sys_return = port_beta * market_culmulative_excess
    
    expected_idio_var = 0
    for ticker in portfolio_tickers:
        expected_idio_var += (optimal_weights_dict[ticker] ** 2) * capm_params[ticker]['residuals']
        
    weighted_excess_return = 0
    for ticker in portfolio_tickers:
        stock_excess_hold = hold_returns[ticker] - hold_rf['rf']
        stock_cumulative_excess = culmulative_returns(stock_excess_hold)
        weighted_excess_return += optimal_weights_dict[ticker] * stock_cumulative_excess
    
    idio_return = weighted_excess_return - sys_return
    sys_var = (port_beta ** 2) * market_excess_hold.var()  # Variance of the systematic component
    
    portfolio_weights = np.array([optimal_weights_dict.get(ticker, 0) for ticker in portfolio_tickers])
    portfolio_excess_returns = hold_returns[portfolio_tickers] - hold_rf['rf'].values.reshape(-1, 1)
    weighted_port_returns = portfolio_excess_returns.dot(portfolio_weights)
    
    idio_var = 0
    for ticker in portfolio_tickers:
        stock_excess_hold = hold_returns[ticker] - hold_rf['rf']
        stock_sys = capm_params[ticker]['beta'] * market_excess_hold
        stock_idio = stock_excess_hold - stock_sys
        idio_var += (optimal_weights_dict[ticker] ** 2) * stock_idio.var()
        
    total_risk = sys_var + idio_var
    optimal_portfolio_results.append({
        'Portfolio': portfolio,
        'cum_excess_return': weighted_excess_return,
        'sys_return': sys_return,
        'idio_return': idio_return,
        'sys_var': sys_var,
        'idio_var': idio_var,
        'total_var': total_risk,
        'beta': port_beta,
        'exp_idio_var': expected_idio_var
    })
df_optimal_portfolio_results = pd.DataFrame(optimal_portfolio_results)
print(df_optimal_portfolio_results)
    
combined_beta = np.sum([total_optimal_weights[ticker] * capm_params[ticker]['beta'] for ticker in all_tickers])
combined_sys_return = combined_beta * market_culmulative_excess
combined_expected_idio_var = 0
for ticker in all_tickers:
    combined_expected_idio_var += (total_optimal_weights[ticker] ** 2) * capm_params[ticker]['residuals']
    
combined_weighted_excess_return = 0
for ticker in all_tickers:
    stock_excess_hold = hold_returns[ticker] - hold_rf['rf']
    stock_cumulative_excess = culmulative_returns(stock_excess_hold)
    combined_weighted_excess_return += total_optimal_weights[ticker] * stock_cumulative_excess
    
combined_idio_return = combined_weighted_excess_return - combined_sys_return
combined_sys_var = (combined_beta ** 2) * market_excess_hold.var()  # Variance of the systematic component

combined_idio_var = 0
for ticker in all_tickers:
    stock_excess_hold = hold_returns[ticker] - hold_rf['rf']
    stock_sys = capm_params[ticker]['beta'] * market_excess_hold
    stock_idio = stock_excess_hold - stock_sys
    combined_idio_var += (total_optimal_weights[ticker] ** 2) * stock_idio.var()
    
combined_total_risk = combined_sys_var + combined_idio_var
optimal_portfolio_results.append({
    'Portfolio': 'Total',
    'cum_excess_return': combined_weighted_excess_return,
    'sys_return': combined_sys_return,
    'idio_return': combined_idio_return,
    'sys_var': combined_sys_var,
    'idio_var': combined_idio_var,
    'total_var': combined_total_risk,
    'beta': combined_beta,
    'exp_idio_var': combined_expected_idio_var
})

df_optimal_results = pd.DataFrame(optimal_portfolio_results)
print(df_optimal_results)

# Part3

# Part4
from scipy.stats import norm, t, skewnorm
from scipy.special import gammaln
from scipy.special import k1e
from scipy.optimize import differential_evolution
from copulae import GaussianCopula

#normal distribution
def normal_log_likelihood(params, data):  
    mu, sigma = params  
    if sigma <= 0:  
        return np.inf  
    return -np.sum(norm.logpdf(data, loc=mu, scale=sigma))  

#generalized t distribution
def gen_t_log_likelihood(params, data):  
    mu, sigma, nu = params  
    if sigma <= 0 or nu <= 0:  
        return np.inf  
    const = gammaln((nu + 1)/2) - gammaln(nu/2) - 0.5 * np.log(nu * np.pi * sigma**2)  
    log_pdf = const - (nu + 1)/2 * np.log(1 + (data - mu)**2 / (nu * sigma**2))  
    return -np.sum(log_pdf)  

#NIG distribution
def nig_pdf(x, alpha, beta, mu, delta):  
    gamma = np.sqrt(alpha**2 - beta**2)  
    z = (x - mu) / delta  
    bessel_arg = alpha * delta * np.sqrt(1 + z**2)  
    pdf = (alpha * delta / np.pi) * np.exp(delta * gamma + beta * (x - mu)) * k1e(bessel_arg) / np.sqrt(1 + z**2)  
    return np.clip(pdf, 1e-10, None)

def nig_log_likelihood(params, data):
    alpha, beta, mu, delta = params
    if alpha <= 0 or delta <= 0 or np.abs(beta) >= alpha:
        return np.inf
    return -np.sum(np.log(nig_pdf(data, alpha, beta, mu, delta)))

def nig_initial_params(data):
    mean = np.mean(data)
    var = np.var(data)
    skew = pd.Series(data).skew()
    alpha_init = 1.0 / np.sqrt(var)
    beta_init_raw = skew / (3 * np.sqrt(var))
    beta_init = np.clip(beta_init_raw, -0.9 * alpha_init, 0.9 * alpha_init)
    delta_init = np.sqrt(var)
    mu_init = mean - delta_init * beta_init / np.sqrt(alpha_init**2 - beta_init**2)
    return [alpha_init, beta_init, mu_init, delta_init]

def fit_nig(data):
    initial_params = nig_initial_params(data)
    alpha_init = initial_params[0]
    bounds = [
        (1e-5, 100),                    # α ∈ (0, 100)  
        (-0.9*alpha_init, 0.9*alpha_init),  # |β| < 0.9α  
        (None, None),                    # μ无约束  
        (1e-5, 10)                       # δ ∈ (0, 10)  
    ]
    result = minimize(nig_log_likelihood, initial_params, args=(data,),
                     method='L-BFGS-B', bounds=bounds)
    alpha, beta, mu, delta = result.x
    # 检查参数合法性
    if alpha <= 0 or delta <= 0 or np.abs(beta) >= alpha:
        return [np.nan] * 4  # 标记为无效
    return result.x

def nig_log_pdf(x, alpha, beta, mu, delta):
    gamma = np.sqrt(alpha**2 - beta**2)
    z = (x - mu) / delta
    bessel_arg = alpha * delta * np.sqrt(1 + z**2)
    log_pdf = np.log(alpha * delta / np.pi) + delta * gamma + beta * (x - mu) + np.log(k1e(bessel_arg)) - 0.5 * np.log(1 + z**2)
    return log_pdf

def nig_log_likelihood(params, data):
    alpha, beta, mu, delta = params
    if alpha <= 0 or delta <= 0 or np.abs(beta) >= alpha:
        return np.inf
    return -np.sum(nig_log_pdf(data, alpha, beta, mu, delta))


def fit_nig_robust(data):
    result = fit_nig(data)
    if not np.isnan(result[0]):
        return result
    bounds = [
        (1e-5, 100),          # α  
        (-50, 50),             # β（根据数据范围调整）  
        (np.min(data), np.max(data)),  # μ  
        (1e-5, 10)             # δ  
    ]
    result_global = differential_evolution(
        lambda params: nig_log_likelihood(params, data),
        bounds=bounds,
        maxiter=100
    )
    return result_global.x
#skew normal distribution
def skew_normal_pdf(x, xi, omega, alpha):  
    z = (x - xi) / omega  
    return (2 / omega) * norm.pdf(z) * norm.cdf(alpha * z)  

def skew_normal_log_likelihood(params, data):  
    xi, omega, alpha = params  
    if omega <= 0:  
        return np.inf  
    return -np.sum(np.log(skew_normal_pdf(data, xi, omega, alpha)))  


def fit_distribution(data, dist_name):  
    if dist_name == "Normal":  
        mu = np.mean(data)  
        sigma = np.std(data)  
        params = [mu, sigma]  
    elif dist_name == "GenT":  
        initial_guess = [np.mean(data), np.std(data), 5.0]  
        result = minimize(gen_t_log_likelihood, initial_guess, args=(data,),  
                          method='L-BFGS-B', bounds=[(None, None), (1e-5, None), (1e-5, None)])  
        params = result.x  
    elif dist_name == "NIG":  
        params = fit_nig(data)  
    elif dist_name == "SkewNormal":  
        initial_guess = [np.mean(data), np.std(data), 0.0]  
        result = minimize(skew_normal_log_likelihood, initial_guess, args=(data,),  
                          method='L-BFGS-B', bounds=[(None, None), (1e-5, None), (None, None)])  
        params = result.x  
    return params  

def calculate_aic(data, params, dist_name):  
    n_params = len(params)  
    if dist_name == "Normal":  
        log_likelihood = -normal_log_likelihood(params, data)  
    elif dist_name == "GenT":  
        log_likelihood = -gen_t_log_likelihood(params, data)  
    elif dist_name == "NIG":  
        log_likelihood = -nig_log_likelihood(params, data)  
    elif dist_name == "SkewNormal":  
        log_likelihood = -skew_normal_log_likelihood(params, data)  
    return 2 * n_params - 2 * log_likelihood  

best_models = {}  
model_params = {}  

for symbol in train_returns.columns:  
    data = train_returns[symbol].dropna().values  
    aic_scores = {}  
    for dist in ["Normal", "GenT", "NIG", "SkewNormal"]:  
        params = fit_distribution(data, dist)  
        aic = calculate_aic(data, params, dist)  
        aic_scores[dist] = aic  
        model_params[(symbol, dist)] = params  
    best_model = min(aic_scores, key=aic_scores.get)  
    best_models[symbol] = best_model  
    
best_models = pd.DataFrame.from_dict(best_models, orient='index', columns=['Best_Model'])
print(best_models)
model_params = pd.DataFrame
print(model_params)


import numpy as np
import pandas as pd
from scipy.stats import norm, t, skewnorm
from scipy.special import gammaln, k1e
from scipy.optimize import minimize, differential_evolution
from copulae import GaussianCopula

#normal distribution
def normal_log_likelihood(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(norm.logpdf(data, loc=mu, scale=sigma))

#Generalized t distribution
def gen_t_log_likelihood(params, data):
    mu, sigma, nu = params
    if sigma <= 0 or nu <= 0:
        return np.inf
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * sigma**2)
    log_pdf = const - (nu + 1) / 2 * np.log(1 + (data - mu)**2 / (nu * sigma**2))
    return -np.sum(log_pdf)

# NIG distribution
def nig_pdf(x, alpha, beta, mu, delta):
    gamma = np.sqrt(alpha**2 - beta**2)
    z = (x - mu) / delta
    bessel_arg = alpha * delta * np.sqrt(1 + z**2)
    pdf = (alpha * delta / np.pi) * np.exp(delta * gamma + beta * (x - mu)) * k1e(bessel_arg) / np.sqrt(1 + z**2)
    return np.clip(pdf, 1e-10, None)

def nig_log_pdf(x, alpha, beta, mu, delta):
    gamma = np.sqrt(alpha**2 - beta**2)
    z = (x - mu) / delta
    bessel_arg = alpha * delta * np.sqrt(1 + z**2)
    log_pdf = np.log(alpha * delta / np.pi) + delta * gamma + beta * (x - mu) + np.log(k1e(bessel_arg)) - 0.5 * np.log(1 + z**2)
    return log_pdf

def nig_log_likelihood(params, data):
    alpha, beta, mu, delta = params
    if alpha <= 0 or delta <= 0 or np.abs(beta) >= alpha:
        return np.inf
    return -np.sum(nig_log_pdf(data, alpha, beta, mu, delta))

# Skew Normal distribution
def skew_normal_pdf(x, xi, omega, alpha):
    z = (x - xi) / omega
    return (2 / omega) * norm.pdf(z) * norm.cdf(alpha * z)

def skew_normal_log_likelihood(params, data):
    xi, omega, alpha = params
    if omega <= 0:
        return np.inf
    return -np.sum(np.log(skew_normal_pdf(data, xi, omega, alpha)))

def nig_initial_params(data):
    mean = np.mean(data)
    var = np.var(data)
    skew = pd.Series(data).skew()
    alpha_init = 1.0 / np.sqrt(var)
    beta_init_raw = skew / (3 * np.sqrt(var))
    beta_init = np.clip(beta_init_raw, -0.9 * alpha_init, 0.9 * alpha_init)
    delta_init = np.sqrt(var)
    mu_init = mean - delta_init * beta_init / np.sqrt(alpha_init**2 - beta_init**2)
    return [alpha_init, beta_init, mu_init, delta_init]

def fit_nig(data):
    initial_params = nig_initial_params(data)
    alpha_init = initial_params[0]
    bounds = [
        (1e-5, 100),                        # α: (0, 100)
        (-0.9 * alpha_init, 0.9 * alpha_init),  # β: |β| < 0.9α
        (None, None),                        # μ 无约束
        (1e-5, 10)                           # δ: (0, 10)
    ]
    result = minimize(nig_log_likelihood, initial_params, args=(data,),
                      method='L-BFGS-B', bounds=bounds)
    alpha, beta, mu, delta = result.x
    if alpha <= 0 or delta <= 0 or np.abs(beta) >= alpha:
        return [np.nan] * 4 
    return result.x

def fit_nig_robust(data):
    result = fit_nig(data)
    if not np.isnan(result[0]):
        return result
    bounds = [
        (1e-5, 100),                   # α
        (-50, 50),                     # β（根据数据范围适当调整）
        (np.min(data), np.max(data)),  # μ
        (1e-5, 10)                     # δ
    ]
    result_global = differential_evolution(
        lambda params: nig_log_likelihood(params, data),
        bounds=bounds,
        maxiter=100
    )
    return result_global.x

def fit_distribution(data, dist_name):
    if dist_name == "Normal":
        mu = np.mean(data)
        sigma = np.std(data)
        params = [mu, sigma]
    elif dist_name == "GenT":
        initial_guess = [np.mean(data), np.std(data), 5.0]
        result = minimize(gen_t_log_likelihood, initial_guess, args=(data,),
                          method='L-BFGS-B', bounds=[(None, None), (1e-5, None), (1e-5, None)])
        params = result.x
    elif dist_name == "NIG":
        params = fit_nig(data)
    elif dist_name == "SkewNormal":
        initial_guess = [np.mean(data), np.std(data), 0.0]
        result = minimize(skew_normal_log_likelihood, initial_guess, args=(data,),
                          method='L-BFGS-B', bounds=[(None, None), (1e-5, None), (None, None)])
        params = result.x
    else:
        params = None
    return params

def calculate_aic(data, params, dist_name):
    n_params = len(params)
    if dist_name == "Normal":
        log_likelihood = -normal_log_likelihood(params, data)
    elif dist_name == "GenT":
        log_likelihood = -gen_t_log_likelihood(params, data)
    elif dist_name == "NIG":
        log_likelihood = -nig_log_likelihood(params, data)
    elif dist_name == "SkewNormal":
        log_likelihood = -skew_normal_log_likelihood(params, data)
    else:
        log_likelihood = -np.inf
    return 2 * n_params - 2 * log_likelihood


best_params = {}  # 存放最佳模型拟合参数，键为 symbol

for symbol in train_returns.columns:
    data = train_returns[symbol].dropna().values
    aic_scores = {}
    params_dict = {}
    for dist in ["Normal", "GenT", "NIG", "SkewNormal"]:
        params = fit_distribution(data, dist)
        aic = calculate_aic(data, params, dist)
        aic_scores[dist] = aic
        params_dict[dist] = params
    best_model = min(aic_scores, key=aic_scores.get)
    best_params[symbol] = {
        "Best_Model": best_model,
        "Parameters": params_dict[best_model]
    }

# 将最佳模型的拟合参数转换为 DataFrame 输出
best_params_df = pd.DataFrame([
    {"Symbol": symbol, "Best_Model": info["Best_Model"], "Parameters": info["Parameters"]}
    for symbol, info in best_params.items()
])

print("各标的最佳模型及其拟合参数：")
print(best_params_df)