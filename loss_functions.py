import jax.numpy as jnp

def exp_unit_deviance(y_est, y, eps=1e-6):
    """
    Exponential unit deviance loss function.
    
    Args:
        beta (jnp.ndarray): Predicted values (model output).
        y (jnp.ndarray): True values (observed data).
    
    Returns:
        jnp.ndarray: The exponential unit deviance loss.
    """
    return 10 * (jnp.log((y_est + eps) / (y + eps)) + ((y + eps) / (y_est + eps)) - 1)

def huber_loss(y_est, y, delta=1.0):
    """
    Huber loss function.
    
    Args:
        y_est (jnp.ndarray): Predicted values.
        y (jnp.ndarray): True values.
        delta (float): The threshold at which to switch between L1 and L2 loss.
    
    Returns:
        jnp.ndarray: The Huber loss.
    """
    error = y_est - y
    is_small_error = jnp.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (jnp.abs(error) - 0.5 * delta)
    return jnp.where(is_small_error, squared_loss, linear_loss)

def log_mse_loss(y_est, y, eps=1e-3):
    """
    Log Mean Squared Error loss function.
    
    Args:
        y_est (jnp.ndarray): Predicted values.
        y (jnp.ndarray): True values.
        eps (float): Small value to avoid log(0).
    
    Returns:
        jnp.ndarray: The log MSE loss.
    """
    return 10 * (jnp.log(y_est + eps) - jnp.log(y + eps)) ** 2

def quadratic_loss(y_est, y):
    """
    quadratic loss function between predicted and true values.
    
    Args:
        y_est (jnp.ndarray): Predicted values.
        y (jnp.ndarray): True values.
    
    Returns:
        jnp.ndarray: The quadratic loss.
    """
    return 10 * (y_est - y) ** 2 