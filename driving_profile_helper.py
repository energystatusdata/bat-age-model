# helper functions to generate the scenario's driving day types used in use_case_model_EV_modular scripts

import pandas as pd
import numpy as np
import math
from datetime import date, timedelta
from enum import IntEnum
from random import randrange
import pytz
import typing


# day type class
class day_type(IntEnum):  # day type - exact usage depends on the use case. Generally used to determine which & when
    # driving profiles are applied on this day. For an example, see get_car_usage_days_v01()
    NO_CAR_USE_DAY = 0  # car is unused during this day
    WORK_DAY = 1  # work day - e.g., car drives to work and back home
    FREE_DAY = 2  # free day - e.g., car drives to leisure activity or shopping and home
    TRIP_DAY = 3  # trip day - e.g., car drives to a destination (one trip)


TIMEZONE_DEFAULT = 'Europe/Berlin'


# return the randomly determined day types for the complete simulation period
def get_car_usage_days_v01(date_start: date, date_last: date, timezone: typing.Union[str, None] = TIMEZONE_DEFAULT
                           ) -> pd.Series:
    # Each year is filled in this order:
    # TRIP_DAY: for holidays - randomly applied to two saturdays or sundays with a distance of 1 week (3x) or 2 weeks
    #                          (1x), which equivalents to 5x5 = 25 holiday days spent. This results in 8 trip days.
    # WORK_DAY: for workdays - randomly applied to 220 of the remaining Mondays to Fridays
    # FREE_DAY: for other days with car usage (leisure, shopping, ...) - randomly applied to 52 of the remaining days
    # NO_CAR_USE_DAY: car is not used the remaining days (85 for regular year, 86 for leap years)

    N_TRIPS_X_WEEKS = {2: 1, 1: 3}  # results in N_TRIP_DAYS = 2 * (N_TRIPS_ONE_WEEK + N_TRIPS_TWO_WEEKS) trip days
    N_WORK_DAYS = 220
    N_FREE_DAYS = 52
    # N_NO_CAR_USE_DAYS = (days in year) - N_TRIP_DAYS - N_WORK_DAYS - N_FREE_DAYS = 85 or 86

    # date_last = date(date_start.year + num_years, date_start.month, date_start.day) + timedelta(days=-1)

    car_usage_days = pd.Series(dtype=int)
    year_first = date_start.year
    year_last = date_last.year
    num_years_distribute = year_last - year_first + 1  # can be (num_years) or (num_years + 1)
    for i_year in range(num_years_distribute):
        # this could certainly be done smarter/faster, but I have no time for that now and runtime is not critical here
        this_year = year_first + i_year
        ixs = pd.Index(np.arange(date(this_year, 1, 1), date(this_year + 1, 1, 1),
                                 timedelta(days=1), dtype='datetime64[D]'))
        n_days_in_year = len(ixs)
        year_car_usage_days = pd.Series(day_type.NO_CAR_USE_DAY, index=ixs)
        year_holidays = year_car_usage_days.copy()
        for holiday_duration_week, num_holidays in N_TRIPS_X_WEEKS.items():
            for i in range(num_holidays):
                while True:
                    n_days = 7 * holiday_duration_week
                    i_day = randrange(n_days_in_year - n_days)
                    weekday = year_car_usage_days.index[i_day].weekday()
                    if (((weekday == 5) or (weekday == 6))
                            and all(year_holidays.iloc[i_day:(i_day + n_days + 1)] == day_type.NO_CAR_USE_DAY)):
                        year_holidays.iloc[i_day:(i_day + n_days + 1)] = day_type.TRIP_DAY
                        year_car_usage_days.iloc[i_day] = day_type.TRIP_DAY
                        year_car_usage_days.iloc[i_day + n_days] = day_type.TRIP_DAY
                        break

        for i in range(N_WORK_DAYS):
            while True:
                i_day = randrange(n_days_in_year)
                if ((year_car_usage_days.iloc[i_day] == day_type.NO_CAR_USE_DAY)
                        and (year_holidays.iloc[i_day] == day_type.NO_CAR_USE_DAY)):
                    weekday = year_car_usage_days.index[i_day].weekday()
                    if weekday < 5:  # Monday (0) to Friday (4)
                        year_car_usage_days.iloc[i_day] = day_type.WORK_DAY
                        break

        for i in range(N_FREE_DAYS):
            while True:
                i_day = randrange(n_days_in_year)
                if year_car_usage_days.iloc[i_day] == day_type.NO_CAR_USE_DAY:
                    # weekday = year_car_usage_days.index[i_day].weekday()
                    year_car_usage_days.iloc[i_day] = day_type.FREE_DAY
                    break

        year_car_usage_days = year_car_usage_days[(year_car_usage_days.index >= pd.Timestamp(date_start))
                                                  & (year_car_usage_days.index <= pd.Timestamp(date_last))]
        car_usage_days = pd.concat([car_usage_days, year_car_usage_days]).copy()

    if timezone is not None:
        car_usage_days.index = car_usage_days.index.tz_localize(timezone)
    return car_usage_days


# get the duration from a pandas timestamp to a time of the day (e.g., earliest departure time)
# aware of timezone and daylight saving time
def get_duration_from_pd_ts_to_time_of_day(pd_timestamp_start: pd.Timestamp, hour: int, minute: int, second: int):
    pd_timestamp_end = pd.Timestamp(year=pd_timestamp_start.year, month=pd_timestamp_start.month,
                                    day=pd_timestamp_start.day, hour=hour, minute=minute, second=second,
                                    tz=pd_timestamp_start.tz)
    duration = pd_timestamp_end - pd_timestamp_start
    if duration.total_seconds() <= 0:
        return 0
    return duration.total_seconds()


# get the duration from a UNIX timestamp to a time of the day (e.g., earliest departure time)
# aware of timezone and daylight saving time
def get_duration_from_unix_ts_to_time_of_day(unix_timestamp_start: int, hour: int, minute: int, second: int,
                                             timezone: typing.Union[str, None] = TIMEZONE_DEFAULT):
    pd_timestamp_start = pd.Timestamp(ts_input=unix_timestamp_start, tz=timezone, unit="s")
    return get_duration_from_pd_ts_to_time_of_day(pd_timestamp_start, hour, minute, second)


# get the duration from a UNIX timestamp to midnight (end of the day)
# aware of timezone and daylight saving time
def get_duration_from_unix_ts_to_midnight(this_date: pd.Timestamp, unix_timestamp_start: int,
                                          timezone: typing.Union[str, None] = None):
    pd_timestamp_start = pd.Timestamp(ts_input=unix_timestamp_start, tz=timezone, unit="s")
    next_date_no_tz = pd.Timestamp(year=this_date.year, month=this_date.month, day=this_date.day) + timedelta(days=1)
    this_tz = pytz.timezone(timezone)
    pd_timestamp_end = this_tz.localize(next_date_no_tz)
    if pd_timestamp_start >= pd_timestamp_end:  # (this_date + pd.Timedelta(days=1)):
        return 0
    # pd_timestamp_end = pd.Timestamp(year=pd_timestamp_start.year, month=pd_timestamp_start.month,
    #                                 day=pd_timestamp_start.day, hour=0, minute=0, second=0,
    #                                 tz=pd_timestamp_start.tz) + pd.Timedelta(days=1)
    duration = pd_timestamp_end - pd_timestamp_start
    if duration.total_seconds() <= 0:
        return 0
    return duration.total_seconds()


# return a random departure time in the range departure_range_h
def get_random_departure(departure_range_h):
    # if type(departure_range_h) is not list:  # if type(departure_range_h) is int:
    if np.issubdtype(type(departure_range_h), np.number):
        return get_hour_and_minute_from_fractional_hour(departure_range_h)
    if departure_range_h[0] == departure_range_h[1]:
        return get_hour_and_minute_from_fractional_hour(departure_range_h[0])
    hour_float = randrange(departure_range_h[0] * 60.0, departure_range_h[1] * 60.0) / 60.0
    hour, minute = get_hour_and_minute_from_fractional_hour(hour_float)
    return hour, minute, hour_float


# return a random duration in the range duration_range_h
def get_random_duration_s(duration_range_h):
    # if type(duration_range_h) is int:
    if np.issubdtype(type(duration_range_h), np.number):
        return duration_range_h * 3600.0
    if duration_range_h[0] == duration_range_h[1]:
        return duration_range_h[0] * 3600.0
    duration_h_float = randrange(duration_range_h[0] * 60.0, duration_range_h[1] * 60.0) / 60.0
    return duration_h_float * 3600.0


# return the earliest departure time (in seconds) from the current time t_now and the minimum value in duration_range_h
# (in hours), e.g., for scheduled charging)
def get_earliest_departure_from_hour_duration_s(t_now, duration_range_h):
    # if type(duration_range_h) is int:
    if np.issubdtype(type(duration_range_h), np.number):
        return t_now + duration_range_h * 3600.0
    return t_now + duration_range_h[0] * 3600.0


# return the earliest departure time (in seconds) from the current time t_now and the minimum value in duration_range_s
# (in seconds)
def get_earliest_departure_from_second_duration_s(t_now, duration_range_s):
    # if type(duration_range_h) is int:
    if np.issubdtype(type(duration_range_s), np.number):
        return t_now + duration_range_s
    return t_now + duration_range_s[0]


# return the earliest departure time (as a pandas timestamp) from a departure date_departure, and a departure range (in
# hours) - aware of timezone and daylight saving time
def get_earliest_departure_pd_ts(date_departure, departure_range_h, timezone):
    # if type(departure_range_h) is not list:  # if type(departure_range_h) is int:
    if np.issubdtype(type(departure_range_h), np.number):
        dep_h, dep_m = get_hour_and_minute_from_fractional_hour(departure_range_h)
    else:
        dep_h, dep_m = get_hour_and_minute_from_fractional_hour(departure_range_h[0])
    earliest_departure_pd_ts_no_tz = pd.Timestamp(
        year=date_departure.year, month=date_departure.month, day=date_departure.day, hour=dep_h, minute=dep_m)
    this_tz = pytz.timezone(timezone)
    earliest_departure_pd_ts = this_tz.localize(earliest_departure_pd_ts_no_tz)
    return earliest_departure_pd_ts


# return the earliest departure time (as a UNIX timestamp) from a departure date_departure, and a departure range (in
# hours) - aware of timezone and daylight saving time
def get_earliest_departure_unix_ts(date_departure, departure_range_h, timezone):
    earliest_departure_pd_ts = get_earliest_departure_pd_ts(date_departure, departure_range_h, timezone)
    # earliest_departure_unix_ts = ((earliest_departure_pd_ts.tz_convert('UTC') - pd.Timestamp("1970-01-01", tz='UTC'))
    #                               // pd.Timedelta("1s"))
    earliest_departure_unix_ts = ((earliest_departure_pd_ts - pd.Timestamp("1970-01-01", tz='UTC'))
                                  // pd.Timedelta("1s"))
    return earliest_departure_unix_ts


# return hour and minute as integers from a fractional hour float value
def get_hour_and_minute_from_fractional_hour(hour_float):
    hour = math.floor(hour_float)
    minute = math.floor((hour_float - hour) * 60)
    return hour, minute


# return a fractional hour float value from hour and minute as integers
def get_fractional_hour_from_hour_and_minute(hour, minute):
    hour_float = hour + minute / 60.0
    return hour_float


# if __name__ == "__main__":  # for debugging only
#     car_usage_days = get_car_usage_days_v01(date(2022, 10, 12), 5)
#     print("debug here")
