from datetime import datetime

def get_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def get_sec(time_str):
    time_str = str(time_str)
    t = time_str.split(':')
    return int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])

def calculate_total_time(data_list):
    """Calculate the total time between the start and end of a path

    Args:
        data_list: A list of dictionaries representing the path.

    Returns:
        total_time: The total time in seconds between the start and end
                    of the path.
    """
    time_format = '%H:%M:%S'

    # Convert the start time and end time to datetime objects
    start_time = datetime.strptime(data_list[0]['start_time'], time_format)
    end_time = datetime.strptime(data_list[-1]['end_time'], time_format)

    # Calculate the total time in seconds
    total_time = (end_time - start_time).total_seconds()

    return total_time
