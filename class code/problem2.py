import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

# 读取数据
df_portfolio = pd.read_csv('Projects/Final project/initial_portfolio.csv')
df_rf = pd.read_csv('Projects/Final project/rf.csv')
df_prices = pd.read_csv('Projects/Final project/DailyPrices.csv')

# 处理日期格式
df_prices['Date'] = pd.to_datetime(df_prices['Date'])
df_rf['Date'] = pd.to_datetime(df_rf['Date'])
df_prices.set_index('Date', inplace=True)
df_rf.set_index('Date', inplace=True)

# 计算收益率
df_returns = df_prices.pct_change().dropna()

# 拆分训练和测试数据
train_end = pd.Timestamp('2023-12-31')
train_returns = df_returns[df_returns.index <= train_end]
hold_returns = df_returns[df_returns.index > train_end]

train_rf = df_rf[df_rf.index <= train_end]
hold_rf = df_rf[df_rf.index > train_end]

# 提取股票列表
stocks = df_portfolio['Symbol'].unique().tolist()
n_stocks = len(stocks)

# CAPM 模型拟合
capm_params = {}
betas = []

for ticker in stocks:
    if ticker == "SPY":
        capm_params[ticker] = {'alpha': 0, 'beta': 1, 'residuals': 0}
        betas.append(1)
        continue
    
    try:
        # 准备数据
        stock_train = train_returns[ticker].dropna()
        df_temp = pd.DataFrame({
            'Stock': stock_train,
            'market': train_returns['SPY'],
            'rf': train_rf['rf']
        }).dropna()
        
        df_temp['excess_stock'] = df_temp['Stock'] - df_temp['rf']
        df_temp['excess_market'] = df_temp['market'] - df_temp['rf']
        
        # 拟合模型
        X = sm.add_constant(df_temp['excess_market'])
        y = df_temp['excess_stock']
        model = sm.OLS(y, X).fit()
        
        beta_value = model.params['excess_market']
        capm_params[ticker] = {
            'alpha': model.params['const'],
            'beta': beta_value,
            'residuals': model.mse_resid
        }
        betas.append(beta_value)
    except Exception as e:
        print(f"处理 {ticker} 时出错: {e}")
        capm_params[ticker] = {'alpha': 0, 'beta': 1, 'residuals': 0}
        betas.append(1)

# 将 betas 转换为 numpy 数组
betas = np.array(betas)

# 创建 CAPM Betas DataFrame
capm_betas = pd.DataFrame({'Symbol': stocks, 'Beta': betas})
print("CAPM Betas:")
print(capm_betas)

# 实现 expost_factor 函数
def expost_factor(w, upReturns, upFfData, Betas):
    """
    实现基于 CAPM 的事后归因分析，与 Julia 版本保持一致
    
    参数:
    w - 权重向量 (pandas Series)
    upReturns - 资产收益率 DataFrame
    upFfData - 因子数据 DataFrame (SPY)
    Betas - 资产的贝塔系数向量 (numpy array)
    
    返回:
    包含归因分析结果的字典
    """
    # 获取股票和因子名称
    stocks = upReturns.columns.tolist()
    factors = upFfData.columns.tolist()
    
    # 获取时间周期长度
    n = len(upReturns)
    
    # 初始化结果数组
    pReturn = np.zeros(n)
    residReturn = np.zeros(n)
    weights = np.zeros((n, len(w)))
    factorWeights = np.zeros((n, len(factors)))
    
    # 复制初始权重
    lastW = w.values.copy() if isinstance(w, pd.Series) else w.copy()
    
    # 准备收益率矩阵
    matReturns = upReturns.values
    ffReturns = upFfData.values
    
    # 计算残差收益率 (特殊性成分)
    Betas_matrix = np.array(Betas).reshape(-1, 1)
    residIndivual = matReturns - np.dot(ffReturns, Betas_matrix.T)
    
    # 逐期计算收益和权重
    for i in range(n):
        # 保存当前权重
        weights[i, :] = lastW
        
        # 计算因子权重 (portfolio beta)
        factorWeights[i, :] = np.sum(Betas * lastW)
        
        # 根据收益更新权重
        lastW = lastW * (1.0 + matReturns[i, :])
        
        # 计算组合收益率
        pR = np.sum(lastW)
        
        # 归一化权重
        lastW = lastW / pR
        
        # 保存收益率
        pReturn[i] = pR - 1
        
        # 计算残差收益
        residReturn[i] = (pR - 1) - np.dot(factorWeights[i, :], ffReturns[i, :])
    
    # 将收益率添加到因子DataFrame中
    upFfData_copy = upFfData.copy()
    upFfData_copy['Alpha'] = residReturn
    upFfData_copy['Portfolio'] = pReturn
    
    # 计算总收益率
    totalRet = np.exp(np.sum(np.log(pReturn + 1))) - 1
    
    # 计算Carino K
    k = np.log(totalRet + 1) / totalRet if totalRet != 0 else 1
    
    # 计算Carino k_t
    carinoK = np.zeros_like(pReturn)
    non_zero_mask = pReturn != 0
    carinoK[non_zero_mask] = np.log(1.0 + pReturn[non_zero_mask]) / pReturn[non_zero_mask] / k
    carinoK[~non_zero_mask] = 1  # 避免除以零
    
    # 计算归因
    attrib = pd.DataFrame(
        ffReturns * factorWeights * carinoK.reshape(-1, 1), 
        columns=factors
    )
    attrib['Alpha'] = residReturn * carinoK
    
    # 处理残差
    for i in range(n):
        residIndivual[i, :] *= weights[i, :]
    
    # 创建最终归因结果，确保包含所有需要的数据
    result = {
        'totalRet': totalRet,
        'attrib': attrib,
        'upFfData': upFfData_copy,
        'weights': weights,
        'factorWeights': factorWeights,
        'residIndivual': residIndivual,
        'residReturn': residReturn,
        'carinoK': carinoK,
        'pReturn': pReturn
    }
    
    return result

def run_attribution(realized_returns, realized_spy, last_date, start_prices, portfolio, betas):
    """
    执行投资组合归因分析，完全按照Julia代码逻辑实现
    """
    # 计算总价值
    portfolio_holdings = portfolio.set_index('Symbol')['Holding']
    t_value = (start_prices * portfolio_holdings).sum()
    
    # 计算初始权重
    w = pd.Series(0, index=stocks)
    for ticker in stocks:
        if ticker in portfolio_holdings.index:
            w[ticker] = start_prices[ticker] * portfolio_holdings[ticker] / t_value
        else:
            w[ticker] = 0
    
    # 执行总体归因
    result = expost_factor(w, realized_returns, realized_spy, betas)
    
    # 提取结果
    totalRet = result['totalRet']
    attrib = result['attrib']
    upFfData = result['upFfData']
    pReturn = result['pReturn']
    
    # 创建输出DataFrame，与Julia代码一致
    attribution = pd.DataFrame({'Value': ['TotalReturn', 'Return Attribution']})
    
    # 添加因子和Portfolio列
    factors = realized_spy.columns.tolist()
    newFactors = factors + ['Alpha']
    
    for s in newFactors + ['Portfolio']:
        if s in upFfData.columns:
            # 计算因子的总收益
            tr = np.exp(np.sum(np.log(1 + upFfData[s].values))) - 1
            
            # 计算归因贡献
            if s != 'Portfolio':
                atr = attrib[s].sum()
            else:
                atr = tr
                
            # 设置值
            attribution[s] = [tr, atr]
    
    # 波动率归因计算
    # Y是按时间加权的股票收益
    Y = np.column_stack([
        realized_spy.values * result['factorWeights'], 
        result['residReturn']
    ])
    
    # 设置X包含组合收益
    X = np.column_stack([np.ones(len(pReturn)), pReturn])
    
    # 计算Beta并丢弃截距
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    betas_vol = B[1, :]
    
    # 组分标准差是Beta乘以组合的标准差
    cSD = betas_vol * np.std(pReturn)
    
    # 添加波动率归因到输出
    vol_attribution = pd.DataFrame({'Value': ['Vol Attribution']})
    
    # 为每个因子和Portfolio添加波动率贡献
    for i, factor in enumerate(newFactors):
        vol_attribution[factor] = [cSD[i]]
    
    vol_attribution['Portfolio'] = [np.std(pReturn)]
    
    # 合并结果
    attribution = pd.concat([attribution, vol_attribution], ignore_index=True)
    
    print("\n总体组合归因:")
    print(attribution)
    
    # 按子组合归因
    portfolios = portfolio['Portfolio'].unique()
    portfolio_results = []
    
    for p in portfolios:
        print(f"\n{p} 组合归因:")
        p_stocks = portfolio[portfolio['Portfolio'] == p]['Symbol'].tolist()
        p_holdings = portfolio[portfolio['Portfolio'] == p].set_index('Symbol')['Holding']
        
        p_t_value = (start_prices[p_stocks] * p_holdings).sum()
        p_w = pd.Series(0, index=stocks)
        
        for ticker in p_stocks:
            p_w[ticker] = start_prices[ticker] * p_holdings[ticker] / p_t_value
        
        # 归因分析
        p_result = expost_factor(p_w, realized_returns, realized_spy, betas)
        
        # 提取结果
        p_totalRet = p_result['totalRet']
        p_attrib = p_result['attrib']
        p_upFfData = p_result['upFfData']
        p_pReturn = p_result['pReturn']
        
        # 创建输出DataFrame
        p_attribution = pd.DataFrame({'Value': ['TotalReturn', 'Return Attribution']})
        
        # 添加因子和Portfolio列
        for s in newFactors + ['Portfolio']:
            if s in p_upFfData.columns:
                # 计算因子的总收益
                tr = np.exp(np.sum(np.log(1 + p_upFfData[s].values))) - 1
                
                # 计算归因贡献
                if s != 'Portfolio':
                    atr = p_attrib[s].sum()
                else:
                    atr = tr
                    
                # 设置值
                p_attribution[s] = [tr, atr]
        
        # 波动率归因计算
        p_Y = np.column_stack([
            realized_spy.values * p_result['factorWeights'], 
            p_result['residReturn']
        ])
        
        p_X = np.column_stack([np.ones(len(p_pReturn)), p_pReturn])
        p_B = np.linalg.inv(p_X.T @ p_X) @ p_X.T @ p_Y
        p_betas_vol = p_B[1, :]
        p_cSD = p_betas_vol * np.std(p_pReturn)
        
        # 添加波动率归因到输出
        p_vol_attribution = pd.DataFrame({'Value': ['Vol Attribution']})
        
        for i, factor in enumerate(newFactors):
            p_vol_attribution[factor] = [p_cSD[i]]
        
        p_vol_attribution['Portfolio'] = [np.std(p_pReturn)]
        
        # 合并结果
        p_attribution = pd.concat([p_attribution, p_vol_attribution], ignore_index=True)
        
        print(p_attribution)
        
        # 存储结果
        portfolio_results.append({
            'Portfolio': p,
            'Attribution': p_attribution
        })
    
    return {'Total': attribution, 'Portfolios': portfolio_results}

# 准备归因分析所需的数据
realized_returns = hold_returns[stocks]
realized_spy = pd.DataFrame({'SPY': hold_returns['SPY']})
last_date = train_returns.index[-1]
start_prices = df_prices.loc[last_date, stocks]

# 执行Part 1归因分析
print("\n===== Part 1: 原始投资组合归因分析 =====")
part1_results = run_attribution(realized_returns, realized_spy, last_date, start_prices, df_portfolio, betas)

# ===== Part 2: 最大夏普比率投资组合优化 =====
print("\n===== Part 2: 最大夏普比率投资组合优化 =====")

# 计算持有期前的 SPY 平均收益率
avg_spy_return = train_returns['SPY'].mean()

# 计算持有期前的平均无风险利率
avg_rf_rate = train_rf['rf'].mean()

print(f"训练期SPY平均收益率: {avg_spy_return:.6f}")
print(f"训练期平均无风险利率: {avg_rf_rate:.6f}")

# 预期超额收益
expected_excess_returns = {}
for ticker in stocks:
    if ticker == "SPY":
        expected_excess_returns[ticker] = avg_spy_return - avg_rf_rate
    else:
        # 使用拟合的 beta，假设 alpha = 0
        beta = capm_params[ticker]['beta']
        expected_excess_returns[ticker] = beta * (avg_spy_return - avg_rf_rate)

# 创建预期收益 DataFrame
expected_returns = pd.Series(expected_excess_returns) + avg_rf_rate

# 使用训练期的数据计算协方差矩阵
cov_matrix = train_returns[stocks].cov()

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    计算负的夏普比率（用于最小化）
    
    参数:
    weights - 权重向量
    expected_returns - 预期收益向量
    cov_matrix - 协方差矩阵
    risk_free_rate - 无风险利率
    
    返回:
    负夏普比率
    """
    port_return = np.sum(expected_returns * weights)
    port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_stddev
    return -sharpe

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate, tickers):
    """
    优化投资组合以获得最大夏普比率
    
    参数:
    expected_returns - 预期收益 Series
    cov_matrix - 协方差矩阵 DataFrame
    risk_free_rate - 无风险利率
    tickers - 股票代码列表
    
    返回:
    优化的权重向量
    """
    # 提取特定股票的期望收益和协方差
    sub_expected_returns = expected_returns[tickers]
    sub_cov_matrix = cov_matrix.loc[tickers, tickers]
    
    # 初始权重均匀分配
    initial_weights = np.ones(len(tickers)) / len(tickers)
    
    # 权重约束：和为1，且每个权重大于等于0
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # 优化
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(sub_expected_returns, sub_cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # 返回优化的权重
    return pd.Series(result['x'], index=tickers)

# 创建优化的子投资组合
optimized_portfolios = {}
portfolios = df_portfolio['Portfolio'].unique()

for p in portfolios:
    p_stocks = df_portfolio[df_portfolio['Portfolio'] == p]['Symbol'].tolist()
    
    # 优化子投资组合
    optimized_weights = optimize_portfolio(expected_returns, cov_matrix, avg_rf_rate, p_stocks)
    
    # 存储优化结果
    optimized_portfolios[p] = optimized_weights
    
    # 计算优化后的夏普比率
    port_return = np.sum(expected_returns[p_stocks] * optimized_weights)
    port_stddev = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix.loc[p_stocks, p_stocks], optimized_weights)))
    sharpe = (port_return - avg_rf_rate) / port_stddev
    
    print(f"\n优化后的 {p} 组合:")
    print(f"预期收益率: {port_return:.6f}")
    print(f"预期波动率: {port_stddev:.6f}")
    print(f"夏普比率: {sharpe:.6f}")
    
    # 显示权重变化
    original_weights = {}
    p_holdings = df_portfolio[df_portfolio['Portfolio'] == p].set_index('Symbol')['Holding']
    p_t_value = (start_prices[p_stocks] * p_holdings).sum()
    
    for ticker in p_stocks:
        original_weights[ticker] = start_prices[ticker] * p_holdings[ticker] / p_t_value
    
    weight_comparison = pd.DataFrame({
        'Original Weight': pd.Series(original_weights),
        'Optimized Weight': optimized_weights
    })
    
    print("\n权重比较:")
    print(weight_comparison)

# 创建一个新的投资组合DataFrame，包含优化后的持股量
optimized_df_portfolio = []

for p, weights in optimized_portfolios.items():
    p_stocks = weights.index.tolist()
    
    # 为每个股票创建新的持股量记录
    for ticker in p_stocks:
        # 假设我们保持每个子投资组合的总价值不变
        p_holdings = df_portfolio[df_portfolio['Portfolio'] == p].set_index('Symbol')['Holding']
        p_t_value = (start_prices[p_stocks] * p_holdings).sum()
        
        # 新的持股量 = 优化权重 * 总价值 / 股票价格
        new_holding = weights[ticker] * p_t_value / start_prices[ticker]
        
        optimized_df_portfolio.append({
            'Portfolio': p,
            'Symbol': ticker,
            'Holding': new_holding
        })

# 转换为DataFrame
optimized_df_portfolio = pd.DataFrame(optimized_df_portfolio)

# 使用优化后的投资组合重新运行归因分析
print("\n===== Part 2: 优化投资组合归因分析 =====")
part2_results = run_attribution(realized_returns, realized_spy, last_date, start_prices, optimized_df_portfolio, betas)

# 比较原始和优化后的结果
def calculate_metrics(attribution_df):
    """
    从归因DataFrame中提取关键指标
    """
    return {
        'Total Return': attribution_df.loc[0, 'Portfolio'],
        'Systematic Return': attribution_df.loc[1, 'SPY'],
        'Specific Return': attribution_df.loc[1, 'Alpha'],
        'Portfolio Volatility': attribution_df.loc[2, 'Portfolio'],
        'Systematic Volatility': attribution_df.loc[2, 'SPY'],
        'Specific Volatility': attribution_df.loc[2, 'Alpha']
    }

# 计算总体投资组合指标
original_metrics = calculate_metrics(part1_results['Total'])
optimized_metrics = calculate_metrics(part2_results['Total'])

# 创建指标比较DataFrame
metrics_comparison = pd.DataFrame({
    'Original Portfolio': original_metrics,
    'Optimized Portfolio': optimized_metrics,
    'Difference': {k: optimized_metrics[k] - original_metrics[k] for k in original_metrics}
})

print("\n===== 总体投资组合指标比较 =====")
print(metrics_comparison)

# 计算子投资组合指标比较
for p in portfolios:
    # 找到原始和优化后的对应子投资组合结果
    original_p_result = next(res for res in part1_results['Portfolios'] if res['Portfolio'] == p)['Attribution']
    optimized_p_result = next(res for res in part2_results['Portfolios'] if res['Portfolio'] == p)['Attribution']
    
    # 计算指标
    original_p_metrics = calculate_metrics(original_p_result)
    optimized_p_metrics = calculate_metrics(optimized_p_result)
    
    # 创建比较DataFrame
    p_metrics_comparison = pd.DataFrame({
        'Original Portfolio': original_p_metrics,
        'Optimized Portfolio': optimized_p_metrics,
        'Difference': {k: optimized_p_metrics[k] - original_p_metrics[k] for k in original_p_metrics}
    })
    
    print(f"\n===== {p} 组合指标比较 =====")
    print(p_metrics_comparison)

# 计算每只股票的预期特质性风险与实现值之间的比较
print("\n===== 特质性风险预期与实现值比较 =====")

# 计算原始和优化投资组合的股票权重
original_weights = {}
for ticker in stocks:
    holdings = df_portfolio[df_portfolio['Symbol'] == ticker]['Holding'].sum()
    original_weights[ticker] = holdings * start_prices[ticker] / (start_prices * df_portfolio.set_index('Symbol')['Holding']).sum()

optimized_weights = {}
for ticker in stocks:
    holdings_data = optimized_df_portfolio[optimized_df_portfolio['Symbol'] == ticker]
    if not holdings_data.empty:
        holdings = holdings_data['Holding'].sum()
        optimized_weights[ticker] = holdings * start_prices[ticker] / (start_prices * optimized_df_portfolio.set_index('Symbol')['Holding']).sum()
    else:
        optimized_weights[ticker] = 0

# 计算预期特质性风险
expected_idiosyncratic_risk = {}
for ticker in stocks:
    if ticker == "SPY":
        expected_idiosyncratic_risk[ticker] = 0
    else:
        # 特质性风险是残差的标准差
        expected_idiosyncratic_risk[ticker] = np.sqrt(capm_params[ticker]['residuals'])

# 计算实现的特质性风险 (使用持有期数据)
realized_idiosyncratic_risk = {}
for ticker in stocks:
    if ticker == "SPY":
        realized_idiosyncratic_risk[ticker] = 0
    else:
        # 计算残差收益
        excess_returns = hold_returns[ticker] - hold_rf['rf']
        market_excess_returns = hold_returns['SPY'] - hold_rf['rf']
        beta = capm_params[ticker]['beta']
        residuals = excess_returns - beta * market_excess_returns
        realized_idiosyncratic_risk[ticker] = residuals.std()

# 创建特质性风险比较DataFrame
idiosyncratic_risk_comparison = pd.DataFrame({
    'Original Weight': pd.Series(original_weights),
    'Optimized Weight': pd.Series(optimized_weights),
    'Expected Idiosyncratic Risk': pd.Series(expected_idiosyncratic_risk),
    'Realized Idiosyncratic Risk': pd.Series(realized_idiosyncratic_risk),
    'Difference': {k: realized_idiosyncratic_risk[k] - expected_idiosyncratic_risk[k] for k in expected_idiosyncratic_risk}
})

print(idiosyncratic_risk_comparison)

# 计算原始投资组合的贝塔值
def calculate_portfolio_beta(portfolio_df, stocks_list, betas_array, start_prices_series):
    """
    计算投资组合贝塔值
    
    参数:
    portfolio_df - 投资组合DataFrame
    stocks_list - 所有股票列表
    betas_array - 所有股票贝塔值数组
    start_prices_series - 起始价格Series
    
    返回:
    投资组合贝塔值
    """
    portfolio_beta = 0
    
    # 计算投资组合总市值
    total_value = 0
    for ticker in portfolio_df['Symbol'].unique():
        holdings = portfolio_df[portfolio_df['Symbol'] == ticker]['Holding'].sum()
        if ticker in start_prices_series.index:
            total_value += holdings * start_prices_series[ticker]
    
    # 计算加权贝塔
    for ticker in portfolio_df['Symbol'].unique():
        holdings = portfolio_df[portfolio_df['Symbol'] == ticker]['Holding'].sum()
        if ticker in start_prices_series.index:
            weight = holdings * start_prices_series[ticker] / total_value
            idx = stocks_list.index(ticker) if ticker in stocks_list else -1
            if idx >= 0:
                portfolio_beta += weight * betas_array[idx]
    
    return portfolio_beta

# 计算原始总投资组合贝塔
original_portfolio_beta = calculate_portfolio_beta(original_weights, betas, stocks)

# 计算优化后总投资组合贝塔
optimized_portfolio_beta = calculate_portfolio_beta(optimized_weights, betas, stocks)

# 添加贝塔值到总体指标比较
metrics_comparison['Original Portfolio']['Portfolio Beta'] = original_portfolio_beta
metrics_comparison['Optimized Portfolio']['Portfolio Beta'] = optimized_portfolio_beta
metrics_comparison['Difference']['Portfolio Beta'] = optimized_portfolio_beta - original_portfolio_beta

print("\n===== 总体投资组合贝塔比较 =====")
print(f"原始投资组合贝塔值: {original_portfolio_beta:.4f}")
print(f"优化投资组合贝塔值: {optimized_portfolio_beta:.4f}")
print(f"贝塔值变化: {optimized_portfolio_beta - original_portfolio_beta:.4f}")

# 计算各子投资组合的贝塔值比较
print("\n===== 子投资组合贝塔比较 =====")
for p in portfolios:
    # 计算原始子投资组合权重
    original_p_weights = {}
    p_stocks = df_portfolio[df_portfolio['Portfolio'] == p]['Symbol'].tolist()
    p_holdings = df_portfolio[df_portfolio['Portfolio'] == p].set_index('Symbol')['Holding']
    p_t_value = (start_prices[p_stocks] * p_holdings).sum()
    
    for ticker in p_stocks:
        original_p_weights[ticker] = start_prices[ticker] * p_holdings[ticker] / p_t_value
    
    # 计算优化后子投资组合权重
    optimized_p_weights = optimized_portfolios[p]
    
    # 计算贝塔值
    original_p_beta = calculate_portfolio_beta(original_p_weights, betas, stocks)
    optimized_p_beta = calculate_portfolio_beta(optimized_p_weights, betas, stocks)
    
    # 添加贝塔值到子投资组合指标比较
    p_metrics_comparison = pd.DataFrame({
        'Portfolio': [p],
        'Original Beta': [original_p_beta],
        'Optimized Beta': [optimized_p_beta],
        'Beta Change': [optimized_p_beta - original_p_beta]
    })
    
    print(p_metrics_comparison)
    
original_portfolio_betas = {}
for p in portfolios:
    p_df = df_portfolio[df_portfolio['Portfolio'] == p]
    original_portfolio_betas[p] = calculate_portfolio_beta(p_df, stocks, betas, start_prices)

original_portfolio_beta = calculate_portfolio_beta(df_portfolio, stocks, betas, start_prices)

# 创建综合贝塔比较表
beta_comparison = pd.DataFrame({
    'Portfolio': ['Total'] + list(portfolios),
    'Original Beta': [original_portfolio_beta] + [original_portfolio_betas[p] for p in portfolios],
    'Optimized Beta': [optimized_portfolio_beta] + [calculate_portfolio_beta(
        pd.DataFrame({'Symbol': optimized_portfolios[p].index, 
                     'Holding': optimized_portfolios[p].values}), 
        stocks, betas, start_prices) for p in portfolios]
})

beta_comparison['Beta Change'] = beta_comparison['Optimized Beta'] - beta_comparison['Original Beta']
beta_comparison['Change %'] = (beta_comparison['Beta Change'] / beta_comparison['Original Beta'] * 100).round(2).astype(str) + '%'

print("\n===== 投资组合贝塔综合比较 =====")
print(beta_comparison)

# 讨论结果
print("\n===== 结果讨论 =====")
print("1. 最大夏普比率组合优化：")
print("   通过最大化夏普比率，我们重新分配了各子投资组合中的资产权重，以期望获得更好的风险调整后收益。")

print("\n2. 系统性和特质性收益变化：")
if optimized_metrics['Systematic Return'] > original_metrics['Systematic Return']:
    print("   优化后的投资组合具有更高的系统性收益贡献，表明对市场因素的暴露增加。")
else:
    print("   优化后的投资组合具有较低的系统性收益贡献，表明对市场因素的暴露减少。")

if optimized_metrics['Specific Return'] > original_metrics['Specific Return']:
    print("   优化后的特质性收益贡献增加，表明特定股票选择的收益提高。")
else:
    print("   优化后的特质性收益贡献减少，表明特定股票选择的收益下降。")

print("\n3. 风险变化：")
if optimized_metrics['Portfolio Volatility'] < original_metrics['Portfolio Volatility']:
    print("   优化后的投资组合总体波动率降低，表明风险降低。")
else:
    print("   优化后的投资组合总体波动率增加，表明风险增加。")

print("\n4. 特质性风险预期与实现：")
mean_diff = np.mean(list(idiosyncratic_risk_comparison['Difference'].values()))
if mean_diff > 0:
    print(f"   实现的特质性风险平均高于预期 ({mean_diff:.6f})，表明CAPM模型可能低估了特质性风险。")
else:
    print(f"   实现的特质性风险平均低于预期 ({mean_diff:.6f})，表明CAPM模型可能高估了特质性风险。")

print("\n5. 投资组合优化效果：")
if optimized_metrics['Total Return'] > original_metrics['Total Return']:
    print("   优化后的投资组合取得了更高的总收益，优化策略在后验检验中表现良好。")
else:
    print("   优化后的投资组合总收益降低，说明预期与实际结果存在差异，或市场在持有期内发生了变化。")

sharpe_original = (original_metrics['Total Return'] - avg_rf_rate * len(hold_rf)) / original_metrics['Portfolio Volatility']
sharpe_optimized = (optimized_metrics['Total Return'] - avg_rf_rate * len(hold_rf)) / optimized_metrics['Portfolio Volatility']

print(f"\n   原始投资组合实现的夏普比率: {sharpe_original:.6f}")
print(f"   优化投资组合实现的夏普比率: {sharpe_optimized:.6f}")

if sharpe_optimized > sharpe_original:
    print("   优化策略成功提高了风险调整后收益。")
else:
    print("   优化策略未能提高风险调整后收益，可能是因为模型假设与实际市场条件不符。")