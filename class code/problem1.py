import pandas as pd
import numpy as np
import statsmodels.api as sm

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

# 准备归因分析所需的数据
realized_returns = hold_returns[stocks]
realized_spy = pd.DataFrame({'SPY': hold_returns['SPY']})
last_date = train_returns.index[-1]
start_prices = df_prices.loc[last_date, stocks]

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

# 执行归因分析
attribution_results = run_attribution(realized_returns, realized_spy, last_date, start_prices, df_portfolio, betas)

# 打印总体归因结果
print("\n归因分析结果摘要:")
print(attribution_results['Total'])


# 计算投资组合贝塔
portfolio_holdings = df_portfolio.set_index('Symbol')['Holding']
total_value = (start_prices * portfolio_holdings).sum()
weights = pd.Series(0, index=stocks)

for ticker in stocks:
    if ticker in portfolio_holdings.index:
        weights[ticker] = start_prices[ticker] * portfolio_holdings[ticker] / total_value

portfolio_beta = np.sum(weights * betas)
print(f"\n投资组合总体贝塔: {portfolio_beta:.4f}")

# 按子组合计算贝塔
portfolios = df_portfolio['Portfolio'].unique()
for p in portfolios:
    p_stocks = df_portfolio[df_portfolio['Portfolio'] == p]['Symbol'].tolist()
    p_holdings = df_portfolio[df_portfolio['Portfolio'] == p].set_index('Symbol')['Holding']
    
    p_t_value = (start_prices[p_stocks] * p_holdings).sum()
    p_weights = pd.Series(0, index=stocks)
    
    for ticker in p_stocks:
        p_weights[ticker] = start_prices[ticker] * p_holdings[ticker] / p_t_value
    
    p_beta = np.sum(p_weights * betas)
    print(f"{p} 组合贝塔: {p_beta:.4f}")

# 计算风险指标
portfolio_std = realized_returns.dot(weights).std() * np.sqrt(252)  # 年化波动率
market_std = realized_spy['SPY'].std() * np.sqrt(252)  # 市场年化波动率

print(f"\n风险指标:")
print(f"投资组合年化波动率: {portfolio_std:.4f}")
print(f"市场年化波动率: {market_std:.4f}")
print(f"系统性风险比例: {(portfolio_beta * market_std / portfolio_std)**2:.4f}")
print(f"非系统性风险比例: {1 - (portfolio_beta * market_std / portfolio_std)**2:.4f}")

# 讨论结果
print("\n结果讨论:")
print("1. 系统性风险来源: 通过 CAPM 模型，我们看到每个投资组合的收益可以被分解为系统性和特殊性成分。")
print("2. 贝塔系数解读: 贝塔值大于1的投资组合对市场变动更敏感，而贝塔值小于1的投资组合则相对稳定。")
print("3. 特殊性收益: 特殊性收益反映了投资组合中各股票的独特表现，与整体市场走势无关。")
print("4. 投资组合分析: 通过比较不同投资组合的归因结果，可以评估它们的风险和收益特征。")
print("5. 模型局限性: CAPM 模型假设市场是有效的，且收益与单一因子(市场)线性相关，这可能无法捕捉所有现实市场的复杂性。")