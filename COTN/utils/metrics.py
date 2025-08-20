import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R2(pred, true):
    """Coefficient of determination (R²)"""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - (ss_res / ss_tot)

def SMAPE(pred, true):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))

def WAPE(pred, true):
    """Weighted Absolute Percentage Error"""
    return 100 * np.sum(np.abs(pred - true)) / np.sum(np.abs(true))

def NRMSE(pred, true):
    """Normalized Root Mean Square Error"""
    return RMSE(pred, true) / (true.max() - true.min())

def MBE(pred, true):
    """Mean Bias Error"""
    return np.mean(pred - true)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe

def comprehensive_metrics(pred, true):
    """Calculate comprehensive evaluation metrics"""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)
    smape = SMAPE(pred, true)
    wape = WAPE(pred, true)
    nrmse = NRMSE(pred, true)
    mbe = MBE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    
    return {
        'MAE': mae,
        'MSE': mse, 
        'RMSE': rmse,
        'MAPE': mape,
        'MSPE': mspe,
        'R2': r2,
        'SMAPE': smape,
        'WAPE': wape,
        'NRMSE': nrmse,
        'MBE': mbe,
        'RSE': rse,
        'CORR': corr
    }