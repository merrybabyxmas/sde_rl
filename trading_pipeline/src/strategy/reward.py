def calculate_reward(old_wealth, new_wealth, slippage, volatility, lambda1=0.1, lambda2=0.05):
    """
    Calculates the reward based on profit, slippage penalty, and volatility penalty.

    Args:
        old_wealth (float): Portfolio value before action.
        new_wealth (float): Portfolio value after action.
        slippage (float): Slippage incurred during trade (0 for Hold).
        volatility (float): Recent volatility measure.
        lambda1 (float): Penalty weight for slippage.
        lambda2 (float): Penalty weight for volatility.

    Returns:
        float: Calculated reward.
    """
    if old_wealth == 0:
        return 0.0

    profit = (new_wealth - old_wealth) / old_wealth
    penalty = (lambda1 * slippage) + (lambda2 * volatility)
    return profit - penalty
