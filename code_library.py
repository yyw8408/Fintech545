import pandas as pd
import numpy as np

#covariance of missing values
def cov_missing(x, skipMiss = True, fun = np.cov):
    x = pd.DataFrame(x)
    if skipMiss:
       return fun(x.dropna().values, rowvar=False) #rowvar=False表示按列向量进行运算
    m = x.shape[1] #存储的列数，变量数
    out = np.full((m, m), np.nan) #创建一个m*m的空矩阵用于填充协方差数值
    for i in range(m):
        for j in range(i+1): #双层for循环遍历矩阵下三角部分
            valid_data = x[[i,j]].dropna()
            if valid_data.shape[0] > 1:
                cov_ij = fun(valid_data.values, rowvar=False)[0,1] #计算返回的是一个2*2协方差矩阵，[0,1]索引表示协方差数值
                out[i,j] = out[j,i] = cov_ij
    return out
                
# data_1_1 = pd.read_csv("testfiles/data/test1.csv")
# cov_matrix = cov_missing(data_1_1)
# output_file = "testfiles/data/test1_1_result.csv"
# pd.DataFrame(cov_matrix).to_csv(output_file, index=False, header=False)

#correlation of missing values
def corr_missing(x, skipMiss = True):
    m = x.shape[1]
    x = pd.DataFrame(x)
    if skipMiss:
        x_dropna = x.dropna()
        cov_matrix = cov_missing(x, skipMiss)
        #cov_matrix = np.cov(x_dropna, rowvar=False)
        std_dev = np.std(x_dropna, axis=0, ddof=1) #nanstd对非缺失值求标准差，axis=0表示对列求标准差，ddof=1表示计算样本标准差
        corr_matrix = np.full((m, m),np.nan)    
    
        for i in range(m):
            for j in range(m):
                if std_dev[i] > 0 and std_dev[j] > 0:
                    corr_matrix[i, j] = cov_matrix[i, j] / (std_dev[i]*std_dev[j])
        return corr_matrix
    cov_matrix = np.full((m, m), np.nan)
    std_devs = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(i + 1):
            pair_data = x.iloc[:, [i, j]].dropna()  # 仅保留两列的非缺失值数据
            n_samples = pair_data.shape[0]

            if n_samples < 2:
                cov_ij, std_i, std_j = np.nan, np.nan, np.nan
            else:
                cov_ij = np.cov(pair_data.values, rowvar=False)[0, 1]
                std_i = np.std(pair_data.iloc[:, 0], ddof=1)  # 样本标准差
                std_j = np.std(pair_data.iloc[:, 1], ddof=1)

            cov_matrix[i, j] = cov_matrix[j, i] = cov_ij
            std_devs[i, j] = std_devs[j, i] = std_i * std_j
    corr_matrix = np.divide(cov_matrix, std_devs, where=(std_devs != 0) & ~np.isnan(std_devs))
    corr_matrix[std_devs == 0] = np.nan  # 避免标准差为 0 时除以 0
    np.fill_diagonal(corr_matrix, 1)  # 变量自身的相关性设为 1

    return corr_matrix

# data_1_2 = pd.read_csv("testfiles/data/test1.csv")
# corr_matrix = corr_missing(data_1_2)
# output_file = "testfiles/data/test1_2_result.csv"
# pd.DataFrame(corr_matrix).to_csv(output_file, index=False, header=False)

def cov_pairwise(x):
    x = pd.DataFrame(x)
    m = x. shape[1]
    cov_matrix = np.full((m,m), np.nan)
    for i in range(m):
        for j in range(i,m):
            pair_data = x.iloc[:, [i,j]].dropna() # : 表示选取所有行，[i,j]表示第i和j列
            if pair_data.shape[0] >= 2: #至少需要两个样本
                cov_ij = np.cov(pair_data.values, rowvar=False)[0,1]
                cov_matrix[i,j] = cov_matrix[j,i] = cov_ij
    return cov_matrix

# data_1_3 = pd.read_csv("testfiles/data/test1.csv")
# cov_matrix = cov_pairwise(data_1_3)
# output_file = "testfiles/data/test1_3_result.csv"
# pd.DataFrame(cov_matrix).to_csv(output_file, index=False, header=False)

def corr_pairwise(x):
    x = pd.DataFrame(x)
    m = x.shape[1]
    cov_matrix = np.full((m, m), np.nan)
    std_devs = np.full((m, m), np.nan)

    for i in range(m):
        for j in range(i, m):  # 从 i 开始，避免重复计算
            pair_data = x.iloc[:, [i, j]].dropna()
            n_samples = pair_data.shape[0]

            if n_samples >= 2:
                cov_ij = np.cov(pair_data.values, rowvar=False)[0, 1]
                std_i = np.std(pair_data.iloc[:, 0], ddof=1)
                std_j = np.std(pair_data.iloc[:, 1], ddof=1)

                cov_matrix[i, j] = cov_matrix[j, i] = cov_ij
                std_devs[i, j] = std_devs[j, i] = std_i * std_j

    corr_matrix = np.divide(cov_matrix, std_devs, where=(std_devs != 0) & ~np.isnan(std_devs))

    return corr_matrix


# data_1_4 = pd.read_csv("testfiles/data/test1.csv")
# corr_matrix = corr_pairwise(data_1_4)
# output_file = "testfiles/data/test1_4_result.csv"
# pd.DataFrame(corr_matrix).to_csv(output_file, index=False, header=False)


def compute_ew_cov(df, λ):
    n, m = df.shape  

    # 生成逆序权重（确保最近数据权重更大）
    w = np.array([(1 - λ) * (λ ** i) for i in range(n)])
    w = w[::-1]  # 反转权重顺序
    w /= w.sum()  # 归一化

    ew_cov_matrix = np.full((m, m), np.nan)

    for i in range(m):
        for j in range(i, m):  
            x = df.iloc[:, i].values  
            y = df.iloc[:, j].values  

            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            valid_data_count = np.sum(valid_mask)
            if valid_data_count < 2:
                continue  # 至少需要2个样本

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            w_valid = w[valid_mask]

            sum_weights = w_valid.sum()
            if sum_weights <= 1e-9:  # 避免除以0
                continue

            # 计算加权均值
            μ_x = np.sum(w_valid * x_valid) / sum_weights
            μ_y = np.sum(w_valid * y_valid) / sum_weights

            # 计算协方差（除以有效权重和）
            cov_ij = np.sum(w_valid * (x_valid - μ_x) * (y_valid - μ_y)) / sum_weights

            ew_cov_matrix[i, j] = ew_cov_matrix[j, i] = cov_ij

    return pd.DataFrame(ew_cov_matrix, index=df.columns, columns=df.columns)

# data_2_1 = pd.read_csv("testfiles/data/test2.csv")
# ewm_cov_matrix = compute_ew_cov(data_2_1,0.97)
# output_file = "testfiles/data/test2_1_result.csv"
# pd.DataFrame(ewm_cov_matrix).to_csv(output_file, index=False, header=False)

def compute_ew_corr(df, λ):
    """
    计算指数加权相关系数矩阵
    参数:
        df : pd.DataFrame 输入数据（按时间升序排列，最近数据在最后一行）
        λ  : float        衰减因子（0 < λ < 1）
    返回:
        pd.DataFrame 相关系数矩阵
    """
    n, m = df.shape

    # 生成逆序权重（确保最近数据权重更大）
    w = np.array([(1 - λ) * (λ ** i) for i in range(n)])[::-1]  # 逆序
    w /= w.sum()  # 全局归一化

    corr_matrix = np.full((m, m), np.nan)

    for i in range(m):
        for j in range(i, m):  # 遍历上三角（含对角线）
            x = df.iloc[:, i].values
            y = df.iloc[:, j].values

            # 获取共同有效数据点
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            valid_data_count = np.sum(valid_mask)
            if valid_data_count < 2:
                continue  # 至少需要2个样本

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            w_valid = w[valid_mask]

            sum_weights = w_valid.sum()
            if sum_weights <= 1e-9:
                continue  # 避免除以0

            # 归一化有效权重
            w_norm = w_valid / sum_weights

            # 计算加权均值
            mu_x = np.sum(w_norm * x_valid)
            mu_y = np.sum(w_norm * y_valid)

            # 计算协方差和方差
            cov_ij = np.sum(w_norm * (x_valid - mu_x) * (y_valid - mu_y))
            var_x = np.sum(w_norm * (x_valid - mu_x) ** 2)
            var_y = np.sum(w_norm * (y_valid - mu_y) ** 2)

            # 计算标准差
            std_x = np.sqrt(var_x) if var_x > 0 else 0.0
            std_y = np.sqrt(var_y) if var_y > 0 else 0.0

            # 计算相关系数
            if std_x * std_y == 0:
                corr = np.nan
            else:
                corr = cov_ij / (std_x * std_y)
                corr = np.clip(corr, -1, 1)  # 确保数值稳定性

            corr_matrix[i, j] = corr_matrix[j, i] = corr

    # 填充对角线为1（当方差非零时）
    np.fill_diagonal(corr_matrix, 1.0)
    # 修正方差为零时的对角线为NaN
    for i in range(m):
        x = df.iloc[:, i].values
        valid_mask = ~np.isnan(x)
        if valid_mask.sum() < 2:
            corr_matrix[i, i] = np.nan
        else:
            w_valid = w[valid_mask]
            sum_weights = w_valid.sum()
            if sum_weights <= 1e-9:
                corr_matrix[i, i] = np.nan
            else:
                x_valid = x[valid_mask]
                w_norm = w_valid / sum_weights
                mu_x = np.sum(w_norm * x_valid)
                var_x = np.sum(w_norm * (x_valid - mu_x) ** 2)
                if var_x <= 1e-9:
                    corr_matrix[i, i] = np.nan

    return pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)

data_2_2 = pd.read_csv("testfiles/data/test2.csv")
ewm_corr_matrix = compute_ew_cov(data_2_2,0.94)
output_file = "testfiles/data/test2_2_result.csv"
pd.DataFrame(ewm_corr_matrix).to_csv(output_file, index=False, header=False)
