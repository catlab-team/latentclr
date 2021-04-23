from scipy.stats import truncnorm


def truncated_normal(size, threshold=1):
    """Samples values from truncated normal distribution centered at 0

    Args:
        size: shape or amount of samples
        threshold: cut-off value for distribution

    Returns:
        numpy array of given size

    """
    return truncnorm.rvs(-threshold, threshold, size=size)
