import numpy as np

def historical_cvar(returns, confidence_level):
    """
    Calculate Conditional Value at Risk (CVaR) using historical simulation method.
    
    Parameters:
        returns (array-like): Historical returns of the portfolio or asset.
        confidence_level (float): Confidence level for CVaR estimation (e.g., 0.95 for 95% confidence).
    
    Returns:
        float: Conditional Value at Risk (CVaR) at the specified confidence level.
    """
    sorted_returns = np.sort(returns)
    index = ((1 - confidence_level) * len(sorted_returns))
    var = -sorted_returns[index]
    cvar = np.mean(sorted_returns[:index])
    return cvar

# Example data (historical returns)
historical_returns = [-0.02, -0.01, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]

# Calculate CVaR at 95% confidence level
confidence_level = 0.95
cvar_95 = historical_cvar(historical_returns, confidence_level)
print("CVaR at 95% confidence level:", cvar_95)
