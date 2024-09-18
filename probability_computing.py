# +
import numpy as np
from scipy.stats import norm

def to_sec(hour):
    '''Converts a string formatted as 'hh:mm:ss' to seconds.'''
    h, m, s = map(int, hour.split(':'))
    return h * 3600 + m * 60 + s
    
def gauss_cdf(mean, std, time):
    """
    Probabiltiy that an Gaussian(mean, std^2) random variable is smaller than time when it has the given mean and std
    
    :param time: the time to compute the probability
    :param mean: the estimated mean from the data
    :param std: the estimated std from the data
    :return: the probability to arrive before the given time
    """
    return norm.cdf(time, loc=mean, scale=std)
    
def exponential_cdf(x, mean):
    """
    Compute the cumulative distribution function for the exponential distribution.

    Parameters:
    x (float or array-like): The value(s) up to which the CDF is calculated.
    mean (float): The mean of the exponential distribution.

    Returns:
    float or array-like: The probability that a random variable is less than x.
    """
    if mean == 0: 
        return 1
    if x == 0:
        return 1
    lambda_value = 1 / mean
    cdf = 1 - np.exp(-lambda_value * x)
    print(f'free time: {x}')
    print(f'proba : {cdf}')
    return cdf

def get_avg_delay_for_stop(stop_means, stop_id, hour):
    '''Fetches the average delay for a given stop at a specific hour or returns the overall average if not specific data is available.'''
    condition = (stop_means['stop_id'] == stop_id) & (stop_means['hour'] == hour)
    if stop_means[condition].any().any():
        return stop_means.loc[condition, 'avg_delay'].values[0], stop_means.loc[condition, 'std_delay'].values[0]
    return stop_means['avg_delay'].mean(), stop_means['std_delay'].mean()

def update_trip_info(edge, prev_stop_time):
    '''Updates trip-related information based on the current edge in the path.'''
    if edge['trip_id'] == 'None':
        walking_time = to_sec(edge['end_time'])-to_sec(edge['start_time'])-10
        #walking_time = int(edge['expected_travel_time'])
        return None, walking_time, prev_stop_time

    if prev_stop_time != 0:
        free_time = to_sec(edge['start_time']) - prev_stop_time
    else:
        free_time = 0

    return edge['trip_id'], free_time, to_sec(edge['end_time'])

def calculate_connection_probability(path, avg_delay):
    '''Calculates the probability of successfully making all connections in a path.'''
    current_trip = ''
    free_time = 0
    prev_stop_time = 0
    probability_exp = 1.0
    probability_norm = 1.0

    for edge in path:
        if edge['trip_id'] != 'None' and edge['trip_id'] != current_trip:
            if current_trip:
                free_time += to_sec(edge['start_time']) - prev_stop_time
                probability_exp *= exponential_cdf(free_time, mean_delay)
                probability_norm *= gauss_cdf(mean_delay, std, free_time)
            current_trip, _, prev_stop_time = update_trip_info(edge, prev_stop_time)
            free_time = 0
        elif edge['trip_id'] == 'None':
            current_trip, walking_time, _ = update_trip_info(edge, prev_stop_time)    
            free_time -= walking_time

        if current_trip != 'None':
            hour = prev_stop_time // 3600
            mean_delay, std = get_avg_delay_for_stop(avg_delay, edge['end_stop_id'], hour)

    return probability_exp, probability_norm
