# +
import numpy as np
from pyspark.sql.functions import col
from probability_computing import to_sec

def compute_valid_delays(stop_delay, free_time):
    """
    Compute the number of valid delays and total delays.
    """
    num_valid = stop_delay.where(col('delay') <= free_time).count()
    num_all = stop_delay.count()
    return num_valid, num_all

def frequency(stop_delay, all_delays, free_time, total):
    """
    Compute the probability of having a delay smaller or equal to f_time following an exponential distribution.
    """
    num_valid, num_all = compute_valid_delays(stop_delay, free_time)

    if num_all == 0:
        num_valid = all_delays.where(col('delay') <= free_time).count()
        num_all = total

    return num_valid / num_all

def update_trip_and_time(edge, current_trip, prev_stop_time):
    """
    Update the current trip ID and compute the free time based on the edge information.
    """
    if edge['trip_id'] == 'None':
        walking_time = to_sec(edge['end_time'])-to_sec(edge['start_time'])-10
        return 'None', walking_time, prev_stop_time

    if current_trip != edge['trip_id'] and prev_stop_time != 0:
        free_time = to_sec(edge['start_time']) - prev_stop_time
    else:
        free_time = 0

    return edge['trip_id'], free_time, to_sec(edge['end_time'])

def historic_frequency(path, all_delays, total):
    """
    Compute the probability of catching every connection of a given path.
    """
    current_trip = ''
    free_time = 0
    prev_stop_time = 0
    freq = 1.0
    
    for edge in path:
        current_trip, extra_time, prev_stop_time = update_trip_and_time(edge, current_trip, prev_stop_time)
        
        if edge['trip_id'] == 'None':
            if free_time < extra_time :
                free_time = 0
            else :
                free_time -= extra_time
            continue

        if current_trip != 'None':
            free_time += extra_time
            stop_delay = all_delays.where((col('stop_id') == edge['end_stop_id']) & (col('hour') == prev_stop_time // 3600))
            freq *= frequency(stop_delay, all_delays, free_time, total)
            free_time = 0

    return freq


