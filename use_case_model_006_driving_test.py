# Example: simple simulation of an EV driving
# (one scenario only, this was just for testing - the full simulation is in use_case_model_EV_modular_v01.py)

import numpy as np
# import plotly.io as pio
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import pandas as pd
import datetime
import pytz

import bat_model_v01 as bat
import driving_profile_helper as drv
import input_data_helper
import result_plot

USE_CASE_NAME = "Use case model 6: driving test"
PLOT_VI = False  # plot voltage and current (larger plot size!)
if PLOT_VI:
    RESULT_SUBPLOT_TITLES = ["Cell power", "Cell current", "Cell voltage", "Cell SoC", "Cell temperature",
                             "Remaining usable capacity"]
    RESULT_SUBPLOT_YAXIS_TITLES = ["Power [W]", "Current [A]", "Voltage [V]", "SoC [0..1]", "Temperature [°C]",
                                   "Capacity [Ah]"]
else:
    RESULT_SUBPLOT_TITLES = ["Cell power", "Cell SoC", "Cell temperature", "Remaining usable capacity"]
    RESULT_SUBPLOT_YAXIS_TITLES = ["Power [W]", "SoC [0..1]", "Temperature [°C]", "Capacity [Ah]"]
EXPORT_PATH = "H:\\Luh\\bat\\analysis\\use_case_models\\images\\"
OPEN_IN_BROWSER = True
EXPORT_HTML = True
EXPORT_IMAGE = None  # "png" - Check failed: message->data_num_bytes() <= Channel::kMaximumMessageSize - IPC message ...
EXPORT_FILENAME = "use_case_model_006_driving_test"


TIMEZONE = 'Europe/Berlin'
TIMEZONE_INFO = pytz.timezone(TIMEZONE)
T_AMBIENT = 20
T_TRIP_MIN = 15  # if the battery is colder than this during long trips, heat battery to this temperature
T_FAST_CHARGING = 30  # hold battery at 30°C at long trips
DATE_START = datetime.date(2013, 1, 1)
SIMULATION_YEARS = 10  # 20

T_RESOLUTION_ACTIVE = 15  # in seconds, temporal resolution for modeling an active cell (charging, discharging)
T_RESOLUTION_PROFILE = 1  # in seconds, temporal resolution for modeling a profile aging cell (discharging) -> need 1s!
T_RESOLUTION_REST = 300  # in seconds, temporal resolution for modeling a resting cell (idle)

DRIVING_PROFILE_WORK = bat.wltp_profiles.full
DRIVING_PROFILE_FREE = bat.wltp_profiles.low + bat.wltp_profiles.medium
DRIVING_PROFILE_TRIP = bat.wltp_profiles.extra_high
DRIVING_PROFILE_TRIP_REPEAT = 48  # repeat driving_profile_trip 48x
TRIP_V_MIN = bat.get_ocv_from_soc(0.1)  # recharge if ocv < voltage at 10 % SoC

HOME_TO_WORK_DEPARTURE_H_RANGE = [7.0, 8.0]
WORK_DURATION_H_RANGE = [3.5, 5.0]
HOME_TO_FREE_DEPARTURE_H_RANGE = [10, 18]
FREE_DURATION_H_RANGE = [0.5, 7.0]
TRIP_DEPARTURE_H_RANGE = [6, 11]


V_CHARGE_LIMIT_HOME = bat.get_ocv_from_soc(0.9)  # charge terminal voltage of 90% SoC
P_CHARGE_HOME_KW = 11.0
P_CHARGE_HOME_CELL = P_CHARGE_HOME_KW * bat.wltp_profiles.P_EV_KW_TO_P_CELL_W
# V_CHARGE_LIMIT_WORK = bat.get_ocv_from_soc(0.9)  # charge to 90% SoC
# P_CHARGE_WORK_KW = 11.0
# P_CHARGE_WORK_CELL = P_CHARGE_HOME_KW * bat.wltp_profiles.P_EV_KW_TO_P_CELL_W
I_CHARGE_AC_CUTOFF = bat.CAP_NOMINAL / 20.0  # cutoff at C/20

V_CHARGE_LIMIT_FAST = bat.get_ocv_from_soc(0.95)  # charge terminal voltage of 95% SoC but limit cutoff!
P_CHARGE_FAST_KW = 100.0
P_CHARGE_FAST_CELL = P_CHARGE_FAST_KW * bat.wltp_profiles.P_EV_KW_TO_P_CELL_W
I_CHARGE_FAST_CUTOFF = bat.CAP_NOMINAL / 2.0  # cutoff at C/2 -> ca. 6.2 W cell power -> 36 kW charging power

COL_Q_LOSS_SEI = "q_loss_sei_total"
COL_Q_LOSS_CYC = "q_loss_cyclic_total"
COL_Q_LOSS_LOW = "q_loss_cyclic_low_total"
COL_Q_LOSS_PLA = "q_loss_plating_total"
I_COL_Q_LOSS_SEI = 0
I_COL_Q_LOSS_CYC = 1
I_COL_Q_LOSS_LOW = 2
I_COL_Q_LOSS_PLA = 3
COL_ARR_AGING_STATES = [COL_Q_LOSS_SEI, COL_Q_LOSS_CYC, COL_Q_LOSS_LOW, COL_Q_LOSS_PLA]


def run():
    cap_aged, aging_states, temp_cell, soc = bat.init()  # init battery
    car_usage_days = drv.get_car_usage_days_v01(DATE_START, SIMULATION_YEARS, TIMEZONE)  # init simulation period
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = bat.init_empty_df()
    temp_ambient_df = input_data_helper.load_temperature_data(TIMEZONE, True)
    cap_aged_df = pd.Series(np.nan, index=car_usage_days.index)
    aging_states_df = pd.DataFrame(np.nan, columns=COL_ARR_AGING_STATES, index=car_usage_days.index)
    t_start = car_usage_days.index[0].timestamp()
    for date, car_usage_day_type in car_usage_days.items():
        cap_aged_df.loc[t_start] = cap_aged
        aging_states_df.loc[t_start, COL_Q_LOSS_SEI] = aging_states[I_COL_Q_LOSS_SEI]
        aging_states_df.loc[t_start, COL_Q_LOSS_CYC] = aging_states[I_COL_Q_LOSS_CYC]
        aging_states_df.loc[t_start, COL_Q_LOSS_LOW] = aging_states[I_COL_Q_LOSS_LOW]
        aging_states_df.loc[t_start, COL_Q_LOSS_PLA] = aging_states[I_COL_Q_LOSS_PLA]
        # noinspection PyTypeChecker
        cap_aged, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, t_start = \
            simulate_day(date, t_start, car_usage_day_type, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                         v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df
                         )

    # date = car_usage_days.index[-1] + datetime.timedelta(days=1)
    cap_aged_df.loc[t_start] = cap_aged
    aging_states_df.loc[t_start, COL_Q_LOSS_SEI] = aging_states[I_COL_Q_LOSS_SEI]
    aging_states_df.loc[t_start, COL_Q_LOSS_CYC] = aging_states[I_COL_Q_LOSS_CYC]
    aging_states_df.loc[t_start, COL_Q_LOSS_LOW] = aging_states[I_COL_Q_LOSS_LOW]
    aging_states_df.loc[t_start, COL_Q_LOSS_PLA] = aging_states[I_COL_Q_LOSS_PLA]

    # save data
    run_timestring = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_filename_csv = EXPORT_FILENAME + "_" + run_timestring + ".csv"
    data_df = pd.concat([p_cell_df, i_cell_df, v_cell_df, soc_df, temp_cell_df, cap_aged_df], axis=1,
                        keys=["P_cell [W]", "I_cell [A]", "V_cell [V]", "SoC_cell [0..1]", "T_cell [degC]",
                               "Remaining capacity [Ah]"])
    data_df = pd.concat([data_df, aging_states_df], axis=1)
    # data_df.rename(columns={"p_cell_df": "P_cell [W]", "i_cell_df": "I_cell [A]", "v_cell_df": "V_cell [V]",
    #                         "soc_df": "SoC_cell [0..1]", "temp_cell_df": "T_cell [degC]",
    #                         "cap_aged_df": "Remaining capacity [Ah]"})
    data_df.to_csv(EXPORT_PATH + export_filename_csv, index=True, index_label="timestamp",
                   sep=";", float_format="%.4f")  # , na_rep="nan")

    # generate plot
    result_fig = result_plot.generate_base_figure(len(RESULT_SUBPLOT_TITLES), 1, '%s - result' % USE_CASE_NAME,
                                                  RESULT_SUBPLOT_TITLES, RESULT_SUBPLOT_YAXIS_TITLES)

    x_data = pd.to_datetime(p_cell_df.index, unit="s", origin='unix', utc=True).tz_convert(TIMEZONE)
    # removed this, it's too slow:
    # text_data = x_data.strftime('%Y-%m-%d %H:%M:%S')
    # text_data = x_data
    text_data = None

    # x_data_cap = pd.to_datetime(cap_aged_df.index, unit="s", origin='unix', utc=True).tz_convert(TIMEZONE)
    x_data_cap = cap_aged_df.index
    # text_data_cap = x_data_cap.strftime('%Y-%m-%d %H:%M:%S')
    text_data_cap = x_data_cap
    # text_data_cap = None

    result_plot.add_result_trace(result_fig, 0, 0, x_data, p_cell_df.values, result_plot.COLOR_P_CELL,
                                 False, True, text_data, TIMEZONE, False)
    if PLOT_VI:
        result_plot.add_result_trace(result_fig, 1, 0, x_data, i_cell_df.values, result_plot.COLOR_I_CELL,
                                     False, True, text_data, TIMEZONE, False)
        result_plot.add_result_trace(result_fig, 2, 0, x_data, v_cell_df.values, result_plot.COLOR_V_CELL,
                                     False, True, text_data, TIMEZONE, False)
        result_plot.add_result_trace(result_fig, 3, 0, x_data, soc_df.values, result_plot.COLOR_SOC_CELL,
                                     False, True, text_data, TIMEZONE, False)
        result_plot.add_result_trace(result_fig, 4, 0, x_data, temp_cell_df.values,
                                     result_plot.COLOR_T_CELL, False, True, text_data, TIMEZONE, False)
        result_plot.add_result_trace(result_fig, 5, 0, x_data_cap, cap_aged_df.values,
                                     result_plot.COLOR_SOH_CAP, True, True, text_data_cap, TIMEZONE, True)
    else:
        result_plot.add_result_trace(result_fig, 1, 0, x_data, soc_df.values, result_plot.COLOR_SOC_CELL,
                                     False, True, text_data, TIMEZONE, False)
        result_plot.add_result_trace(result_fig, 2, 0, x_data, temp_cell_df.values,
                                     result_plot.COLOR_T_CELL, False, True, text_data, TIMEZONE, False)
        result_plot.add_result_trace(result_fig, 3, 0, x_data_cap, cap_aged_df.values,
                                     result_plot.COLOR_SOH_CAP, True, True, text_data_cap, TIMEZONE, True)

    # result_fig.show(validate=False)
    result_plot.export_figure(result_fig, EXPORT_HTML, EXPORT_IMAGE, EXPORT_PATH, EXPORT_FILENAME, OPEN_IN_BROWSER)
    print("debug")


def simulate_day(date, t_start, car_usage_day_type, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                 v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df
                 ):
    if date.day == 1:
        print("Simulating %4u-%02u..." % (date.year, date.month))
    t_start_beginning_of_day = date.timestamp()
    if (v_cell_df.shape[0] == 0) or (v_cell_df.index[-1] < t_start_beginning_of_day):
        # t_start = t_start_beginning_of_day
        if t_start != t_start_beginning_of_day:
            print("warning: t_start != t_start_beginning_of_day -> %u != %u" % (t_start, t_start_beginning_of_day))
        if v_cell_df.shape[0] > 0:
            if v_cell_df.index[-1] < (t_start_beginning_of_day - 3600.0):  # + datetime.timedelta(hours=-1)
                print("warning: last timestep came too long ago")
    else:
        print("debug: last activity passed midnight on %04u-%02u-%02u..." % (date.year, date.month, date.day))

    if car_usage_day_type == drv.day_type.WORK_DAY:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
            simulate_work_day(date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                              v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df))
    elif car_usage_day_type == drv.day_type.FREE_DAY:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
            simulate_free_day(date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                              v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df))
    elif car_usage_day_type == drv.day_type.TRIP_DAY:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
            simulate_trip_day(date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                              v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df))
    elif car_usage_day_type == drv.day_type.NO_CAR_USE_DAY:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
            simulate_no_car_use_day(date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                                    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df))
    else:
        print("unsupported day_type: " + str(car_usage_day_type))

    return cap_aged, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, t_start


def simulate_work_day(date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                      v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df):
    # wait until HOME_TO_WORK_DEPARTURE_H_RANGE
    h, m, _ = drv.get_random_departure(HOME_TO_WORK_DEPARTURE_H_RANGE)
    rest_duration = drv.get_duration_from_unix_ts_to_time_of_day(t_start, h, m, 0, TIMEZONE)
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))

    # insert 1x driving_profile_work [Home -> Work]
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_power_profile(t_start, T_RESOLUTION_PROFILE, DRIVING_PROFILE_WORK, temp_ambient_df, v_cell_df,
                                i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))

    # wait until end of work (WORK_DURATION_H_RANGE)
    # rest_duration = drv.get_duration_from_unix_ts_to_time_of_day(t_start, 12, 0, 0, TIMEZONE)
    rest_duration = drv.get_random_duration_s(WORK_DURATION_H_RANGE)
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))

    # insert 1x driving_profile_work [Work -> Home]
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_power_profile(t_start, T_RESOLUTION_PROFILE, DRIVING_PROFILE_WORK, temp_ambient_df, v_cell_df,
                                i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))

    # charge to V_CHARGE_LIMIT_HOME at home
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_cp_cv(t_start, T_RESOLUTION_ACTIVE, V_CHARGE_LIMIT_HOME, P_CHARGE_HOME_CELL, I_CHARGE_AC_CUTOFF,
                        temp_ambient_df, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
                        cap_aged, aging_states, temp_cell, soc))

    # wait until midnight
    rest_duration = drv.get_duration_from_unix_ts_to_midnight(date, t_start, TIMEZONE)
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start


def simulate_free_day(date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                      v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df):
    # wait until HOME_TO_FREE_DEPARTURE_H_RANGE
    h, m, _ = drv.get_random_departure(HOME_TO_FREE_DEPARTURE_H_RANGE)
    rest_duration = drv.get_duration_from_unix_ts_to_time_of_day(t_start, h, m, 0, TIMEZONE)
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))

    # insert 1x DRIVING_PROFILE_FREE [Home -> Activity]
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_power_profile(t_start, T_RESOLUTION_PROFILE, DRIVING_PROFILE_FREE, temp_ambient_df, v_cell_df,
                                i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))

    # wait for FREE_DURATION_H_RANGE
    # rest_duration = drv.get_duration_from_unix_ts_to_time_of_day(t_start, 12, 0, 0, TIMEZONE)
    rest_duration = drv.get_random_duration_s(FREE_DURATION_H_RANGE)
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))

    # insert 1x DRIVING_PROFILE_FREE [Activity -> Home]
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_power_profile(t_start, T_RESOLUTION_PROFILE, DRIVING_PROFILE_FREE, temp_ambient_df, v_cell_df,
                                i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))

    # AC charge to V_CHARGE_LIMIT_HOME
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_cp_cv(t_start, T_RESOLUTION_ACTIVE, V_CHARGE_LIMIT_HOME, P_CHARGE_HOME_CELL, I_CHARGE_AC_CUTOFF,
                        temp_ambient_df, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
                        cap_aged, aging_states, temp_cell, soc))

    # wait until midnight
    rest_duration = drv.get_duration_from_unix_ts_to_midnight(date, t_start, TIMEZONE)
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start


def simulate_trip_day(date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                      v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df):

    # wait until TRIP_DEPARTURE_H_RANGE
    h, m, _ = drv.get_random_departure(TRIP_DEPARTURE_H_RANGE)
    rest_duration = drv.get_duration_from_unix_ts_to_time_of_day(t_start, h, m, 0, TIMEZONE)
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))

    # for long distance trips, heat vehicle battery to 20°C if it is colder than that
    t_conditioning_trip = temp_ambient_df.loc[t_start:t_start + (36 * 60 * 60)].copy()
    t_conditioning_trip[t_conditioning_trip < T_TRIP_MIN] = T_TRIP_MIN

    # for charging, keep it at 32°C
    t_conditioning_fast_charging = T_FAST_CHARGING

    n_remaining = DRIVING_PROFILE_TRIP_REPEAT
    # insert DRIVING_PROFILE_TRIP_REPEAT x DRIVING_PROFILE_TRIP, charge in between
    while True:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start, n_rep =\
            bat.apply_power_profile_repeat(t_start, T_RESOLUTION_PROFILE, DRIVING_PROFILE_TRIP, n_remaining,
                                           t_conditioning_trip, None, TRIP_V_MIN, v_cell_df, i_cell_df, p_cell_df,
                                           temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)
        n_remaining = n_remaining - n_rep
        if n_remaining <= 0:
            break
        # else: fast charge to V_CHARGE_LIMIT_FAST
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
            bat.apply_cp_cv(t_start, T_RESOLUTION_ACTIVE, V_CHARGE_LIMIT_FAST, P_CHARGE_FAST_CELL, I_CHARGE_FAST_CUTOFF,
                            t_conditioning_fast_charging, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
                            cap_aged, aging_states, temp_cell, soc))

    # at home (or in holiday: assume there is destination charging) --> charge to V_CHARGE_LIMIT_HOME
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_cp_cv(t_start, T_RESOLUTION_ACTIVE, V_CHARGE_LIMIT_HOME, P_CHARGE_HOME_CELL, I_CHARGE_AC_CUTOFF,
                        temp_ambient_df, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
                        cap_aged, aging_states, temp_cell, soc))

    # wait until midnight
    rest_duration = drv.get_duration_from_unix_ts_to_midnight(date, t_start, TIMEZONE)
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start


def simulate_no_car_use_day(date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                            v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df):
    # wait until midnight
    rest_duration = drv.get_duration_from_unix_ts_to_midnight(date, t_start, TIMEZONE)
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start


if __name__ == "__main__":
    run()
