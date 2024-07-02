# more extended documentation in use_case_model_EV_modular_v01.py
# Note: The "..._fast.py" models run faster, but do not return a voltage/current/power/temperature profile of the cell
#       and the power/price/emissions/residual load/... of the grid.
#       That makes you a bit "blind", because you only get the capacity fade of the cell, but don't see what it did.
#       I suggest starting with the "regular" models until you are absolutely sure about what you do and that the
#       results are legitimate. Then, you can compare the results with the fast model and continue there.

import time
import math
import numpy as np
import pandas as pd
import datetime
import multiprocessing
import os
import traceback

import bat_model_v01_fast as bat
import driving_profile_helper as drv
import scenario_helper as sc
import input_data_helper
# import result_plot
import logger


# --- general definitions ----------------------------------------------------------------------------------------------
USE_CASE_NAME = "Use case 7: modular driving (scenario %u)"
# scenario definition further below

# timezone
TIMEZONE = 'Europe/Berlin'

# export
# EXPORT_PATH = "H:\\Luh\\bat\\analysis\\use_case_models\\images\\"
EXPORT_PATH = "D:\\bat\\analysis\\use_case_models\\images\\"
OPEN_IN_BROWSER = False  # True  # when simulating multiple years, the browser mights struggle to show the result
EXPORT_HTML = False  # True  # when simulating multiple years, this might take long and cause memory errors
EXPORT_IMAGE = None  # "png" - Check failed: message->data_num_bytes() <= Channel::kMaximumMessageSize - IPC message ...
EXPORT_FILENAME_BASE = "use_case_model_007_modular_driving_sc%03u"

# multiprocessing settings
# NUMBER_OF_PROCESSORS_TO_USE = max(multiprocessing.cpu_count() - 1, 1)  # leave one free -> for high performant systems
# NUMBER_OF_PROCESSORS_TO_USE = math.ceil(multiprocessing.cpu_count() / 2)  # use half of the processors -> ...medium...
NUMBER_OF_PROCESSORS_TO_USE = 2  # use two processor
# NUMBER_OF_PROCESSORS_TO_USE = 1  # only use one processor --> use this if you have a low-performant system
modeling_task_queue = multiprocessing.Queue()


# logging_filename = "H:\\Luh\\bat\\analysis\\use_case_models\\log\\use_case_model_007.txt"
logging_filename = "D:\\bat\\analysis\\use_case_models\\log\\use_case_model_007_fast3.txt"
logging = logger.bat_logger(logging_filename)


# --- scenario constants -----------------------------------------------------------------------------------------------
# --- general definitions ---

# # test 2023:
# SIM_DATE_START_DEFAULT = datetime.date(2023, 1, 1)
# SIM_DATE_STOP_DEFAULT = datetime.date(2023, 12, 31)
# SIM_DATE_STOP_FREQ_CTRL = datetime.date(2023, 1, 7)

# # test 2035:
# SIM_DATE_START_DEFAULT = datetime.date(2035, 1, 1)
# SIM_DATE_STOP_DEFAULT = datetime.date(2035, 12, 31)
# SIM_DATE_STOP_FREQ_CTRL = datetime.date(2035, 1, 7)

# # last 5 years:
# SIM_DATE_START_DEFAULT = datetime.date(2019, 3, 13)  # datetime.date(2019, 3, 13)
# SIM_DATE_STOP_DEFAULT = datetime.date(2024, 3, 12)  # datetime.date(2023, 3, 12)
# # SIM_DATE_STOP_DEFAULT = datetime.date(2019, 3, 19)  # datetime.date(2023, 3, 12)
# SIM_DATE_STOP_FREQ_CTRL = SIM_DATE_STOP_DEFAULT

# 20 years:
# SIM_DATE_START_DEFAULT = datetime.date(2019, 3, 13)  # datetime.date(2019, 3, 13)
# SIM_DATE_STOP_DEFAULT = datetime.date(2039, 3, 12)  # datetime.date(2023, 3, 12)

# SIM_DATE_START_DEFAULT = datetime.date(2025, 1, 1)  # this is used in the dissertation
# SIM_DATE_STOP_DEFAULT = datetime.date(2044, 12, 31)

# for tests
SIM_DATE_START_DEFAULT = datetime.date(2025, 5, 1)
SIM_DATE_STOP_DEFAULT = datetime.date(2025, 5, 15)

# SIM_DATE_START_DEFAULT = datetime.date(2025, 8, 1)
# SIM_DATE_STOP_DEFAULT = datetime.date(2025, 8, 31)

# SIM_DATE_START_DEFAULT = datetime.date(2025, 1, 1)
# SIM_DATE_STOP_DEFAULT = datetime.date(2025, 12, 31)

# the frequency control strategy is quite slow (1 s resolution), you can select a different end date if you want:
SIM_DATE_STOP_FREQ_CTRL = SIM_DATE_STOP_DEFAULT

# workaround for huge plots when using frequency control that will throw a memory error
MINIMAL_FREQUENCY_CONTROL_PLOT = True

USE_COMMON_DRIVING_DAYS = True  # if True, use same driving days in all scenarios (using SIM_DATE_START/STOP_DEFAULT)

T_RESOLUTION_ACTIVE = 30  # 15  # in seconds, temporal resolution for modeling an active cell (charging, discharging)
T_RESOLUTION_PROFILE = 1  # in seconds, temporal resolution for modeling a profile aging cell (discharging) -> need 1s
#                           since the profiles have this resolution. Change resolution of profiles when changing this.
T_RESOLUTION_REST = 300  # in seconds, temporal resolution for modeling a resting cell (idle)

# --- driving profiles: workday, free day (leisure / shopping / other activity...), trip (holiday / long 1-way trip) ---
DRIVING_PROFILE_WORK = bat.wltp_profiles.full
DRIVING_PROFILE_WORK_DISTANCE = bat.wltp_profiles.full_distance
DRIVING_PROFILE_FREE = bat.wltp_profiles.low + bat.wltp_profiles.medium
DRIVING_PROFILE_FREE_DISTANCE = bat.wltp_profiles.low_distance + bat.wltp_profiles.medium_distance
DRIVING_PROFILE_TRIP = bat.wltp_profiles.extra_high
DRIVING_PROFILE_TRIP_DISTANCE = bat.wltp_profiles.extra_high_distance

# driving profiles - long-distance trip settings
DRIVING_PROFILE_TRIP_REPEAT = 48  # repeat DRIVING_PROFILE_TRIP 48x
TRIP_V_MIN = bat.get_ocv_from_soc(0.1)  # in V, recharge if ocv < voltage at 10 % SoC
# TEMP_TRIP_MIN / TEMP_FAST_CHARGING -> determine the "ambient" temperature. It is assumed the thermal management system
# of the battery can do this. The effective R_TH between the cell and the ambient temperature is bat.R_TH_CELL and can
# also be overwritten here, e.g.: bat.R_TH_CELL = 10
TEMP_TRIP_MIN = 15  # in °C, if battery colder than this during long trips, let therm. management heat ambient to this T

driving_distances = {
    drv.day_type.NO_CAR_USE_DAY: 0.0,  # no profile
    drv.day_type.WORK_DAY: 2 * DRIVING_PROFILE_WORK_DISTANCE,
    drv.day_type.FREE_DAY: 2 * DRIVING_PROFILE_FREE_DISTANCE,
    drv.day_type.TRIP_DAY: DRIVING_PROFILE_TRIP_REPEAT * DRIVING_PROFILE_TRIP_DISTANCE,
}

# --- departure and duration settings: ---
# work:
HOME_TO_WORK_DEPARTURE_H_RANGE = [7.0, 8.0]  # vary departure time (in h) between these values
WORK_DURATION_H_RANGE_HALF = [3.5, 5.0]  # duration at work for a part-time employment (4 h + 15 min break +/-45 min)
WORK_DURATION_H_RANGE_FULL = [7.25, 10.25]  # duration at work for a full-time employment (8 h + 45 min break +/-90 min)
# free day (leisure / shopping / other activity with car usage):
HOME_TO_FREE_DEPARTURE_H_RANGE = [10, 18]  # vary departure time (in h) between these values
FREE_DURATION_H_RANGE = [0.5, 7.0]  # duration at activity
# trip day (long-distance one-way trip, e.g., for holiday) -> departure: 1 or 2 weeks later (see driving_profile_helper)
TRIP_DEPARTURE_H_RANGE = [6, 11]  # vary departure time (in h) between these values

# --- charging settings ---
# charging strategy: constant power (limited by max. cell current) -> constant voltage. If the charge voltage limit is
# below 100 % (= bat.get_ocv_from_soc(1.0)), the CV will only be charged to this voltage - this limits charging power
# earlier but might prevent lithium plating at colder temperature, high C-rates, and for aged cells. Alternatively, the
# charging voltage can be set to 100 %, but the cutoff current can be limited earlier. Make sure the minimum current in
# the constant power phase is always greater than the cutoff current limit - otherwise, charging will not start.

# energy efficiency
CHG_EFFICIENCY = 0.95  # efficiency of charging or discharging (only charger, battery losses excluded!), 0.95 = 95%

# charging power (battery power, i.e., after on-board or stationary charger or cable losses):
# P_CHG_EV_AC_STD_KW = 7.0   # in kW, standard AC charge:  7 kW charging (7.4 kW -> ca. 7.0 kW after charging losses)
P_CHG_EV_AC_STD_KW = 10.5    # in kW, standard AC charge: 11 kW charging (11.1 kW -> ca. 10.5 kW after charging losses)
P_CHG_EV_AC_SLOW_KW = 5.25   # in kW, slow AC charging:  5.5 kW charging (50% of P_CHG_EV_AC_STD_KW)
P_CHG_EV_DC_MED_KW = 50.0    # in kW, medium DC charging: 50 kW charging
P_CHG_EV_DC_FAST_KW = 100.0  # in kW, fast* DC charging: 100 kW charging
# *not really "fast" as of 2024's battery cells, but the cell chemistry tested is comparatively old -> max 1.67 ... 2 C
# -> p_chg_cell_W = P_CHG_EV_..._KW * bat.wltp_profiles.P_EV_KW_TO_P_CELL_W

# terminal voltages / SoCs
V_CHG_LIMIT_100 = bat.get_ocv_from_soc(1.0)  # charge terminal voltage of 100 % SoC
V_CHG_LIMIT_95 = bat.get_ocv_from_soc(0.95)  # charge terminal voltage of 95 % SoC
V_CHG_LIMIT_90 = bat.get_ocv_from_soc(0.9)  # charge terminal voltage of 90 % SoC
V_CHG_LIMIT_85 = bat.get_ocv_from_soc(0.85)  # charge terminal voltage of 85 % SoC
V_CHG_LIMIT_80 = bat.get_ocv_from_soc(0.8)  # charge terminal voltage of 80 % SoC
V_CHG_LIMIT_75 = bat.get_ocv_from_soc(0.75)  # charge terminal voltage of 80 % SoC
V_CHG_LIMIT_60 = bat.get_ocv_from_soc(0.6)  # charge terminal voltage of 60 % SoC
V_CHG_LIMIT_BEFORE_TRIP = V_CHG_LIMIT_100  # charge terminal voltage before starting a long-distance trip

# cut-off currents
I_CHG_CUTOFF_C_40 = bat.CAP_NOMINAL / 40.0  # cutoff at C/40 (ca. 64 kWh / 40 h = 1.6 kW or ca. 0.3 W cell power)
I_CHG_CUTOFF_C_20 = bat.CAP_NOMINAL / 20.0  # cutoff at C/20 (ca. 64 kWh / 20 h = 3.2 kW or ca. 0.6 W cell power)
I_CHG_CUTOFF_C_10 = bat.CAP_NOMINAL / 10.0  # cutoff at C/2 (ca. 64 kWh / 10 h = 6.4 kW or ca. 1.2 W cell power)
I_CHG_CUTOFF_C_5 = bat.CAP_NOMINAL / 5.0  # cutoff at C/2 (ca. 64 kWh / 5 h = 12.8 kW or ca. 2.5 W cell power)
I_CHG_CUTOFF_C_2 = bat.CAP_NOMINAL / 2.0  # cutoff at C/2 (ca. 64 kWh / 2 h = 32 kW or ca. 6 W cell power)
I_CHG_CUTOFF_BEFORE_TRIP = I_CHG_CUTOFF_C_40  # cutoff current before starting a long-distance trip

# charging temperature
TEMP_FAST_CHARGING = 30  # in °C, hold ambient temperature at this temperature during the fast-charging process
TEMP_CHARGING_MIN = 0  # in °C, if battery colder than this when charging, let thermal management heat ambient to this T

# scheduled charging buffer added to the estimated charging time -> t_chg = cap_remaining / (p_chg / v_lim) + t_buf
T_SCHEDULED_CHARGING_BUFFER_S = 20 * 60  # in seconds, 20 * 60 = 20 minutes

# V1G/V2G charging/discharging preference - see smart_charging() function
# higher = more preference to discharge / use V2G. 0.1...0.5 is probably reasonable
# PREFERENCE_TO_DISCHARGE is multiplied by PREFERENCE_TO_DISCHARGE_BASE_FACTOR - the product should be in range (0, 1)
PREFERENCE_TO_DISCHARGE_BASE_FACTOR = 0.35  # (0..1], higher = make V2G usage more likely, lower = less likely
PREFERENCE_TO_DISCHARGE = {
    sc.CHG_STRAT.V2G_OPT_COST: 1.09,  # 0.38/0.35 = 1.0857
    sc.CHG_STRAT.V2G_OPT_EMISSION: 0.77,  # 0.27/0.35 = 0.7714
    sc.CHG_STRAT.V2G_OPT_REN: 0.91,  # 0.32/0.35 = 0.9142
}
PREFERENCE_TO_CHARGE = 0.99  # higher = more preference to charge. 0.85...1.0 is probably reasonable

# lower SoC threshold for charging
SOC_THS_40 = 0.4
SOC_THS_25 = 0.25

# charging optimization
CHG_OPTIMIZE_INTERVAL_S = 5 * 60  # in seconds, align to intervals of this duration (5 * 60 = 5 minutes)

# home / work PV system peak power (only one power value is supported for all scenarios and locations, sorry)
PV_POWER_PEAK_KW = 10  # in kW
PV_CHARGE_MIN_KW = 0.3  # in kW (>0), minimum charging power supported by the EV when using solar charging
PV_DISCHARGE_MIN_KW = -0.3  # in kW (<0), minimum discharging power supported by the EV when using solar charging

# --- scenario definitions ---------------------------------------------------------------------------------------------
SCENARIO_LIST = [
    # {sc.ID: 0, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.EARLY,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_100, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            # sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_100, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            # sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_100, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {  # sc.CHG_STRATEGY: sc.CHG_STRAT.EARLY, --> for destination charging, same as at home is used.
    #            sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 1, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.EARLY,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 2, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.EARLY,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_80, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 3, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.EARLY_IF_LOW, sc.CHG_SOC_LOW: SOC_THS_40,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_100, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 4, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.EARLY_IF_LOW, sc.CHG_SOC_LOW: SOC_THS_40,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 5, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.EARLY_IF_LOW, sc.CHG_SOC_LOW: SOC_THS_40,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_80, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 6, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.LATE,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_100, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 7, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.LATE,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 8, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.LATE,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_80, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 9, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.LATE_IF_LOW, sc.CHG_SOC_LOW: SOC_THS_40,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_100, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 10, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.LATE_IF_LOW, sc.CHG_SOC_LOW: SOC_THS_40,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 11, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.LATE_IF_LOW, sc.CHG_SOC_LOW: SOC_THS_40,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_80, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 12, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V1G_OPT_EMISSION, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V1G_OPT_EMISSION, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 13, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V1G_OPT_COST, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V1G_OPT_COST, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 14, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V1G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V1G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 15, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_EMISSION, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_EMISSION, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 16, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_COST, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_COST, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 17, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 18, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_FREQ_CTRL,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_FREQ, sc.CHG_SOC_LOW: SOC_THS_40,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_80, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_FREQ, sc.CHG_SOC_LOW: SOC_THS_40,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_80, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 19, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,  # battery optimized
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.LATE_IF_LOW, sc.CHG_SOC_LOW: SOC_THS_40,
    #            sc.CHG_P: P_CHG_EV_AC_SLOW_KW, sc.CHG_V_LIM: V_CHG_LIMIT_60, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_85, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,
    #            },
    #  },
    # {sc.ID: 20, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_75, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_75, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {
    #            sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,
    #            },
    #  },
    # {sc.ID: 21, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_SLOW_KW, sc.CHG_V_LIM: V_CHG_LIMIT_75, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_SLOW_KW, sc.CHG_V_LIM: V_CHG_LIMIT_75, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {
    #            sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,
    #            },
    #  },
    # {sc.ID: 22, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_SLOW_KW, sc.CHG_V_LIM: V_CHG_LIMIT_60, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_SLOW_KW, sc.CHG_V_LIM: V_CHG_LIMIT_60, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {
    #            sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,
    #            },
    #  },
    # {sc.ID: 23, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V1G_OPT_EMISSION, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 24, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V1G_OPT_COST, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 25, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V1G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 26, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_EMISSION, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 27, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_COST, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 28, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_90, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,  # sc.DURATION: N/A --> one-way trip
    #            },
    #  },
    # {sc.ID: 29, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_75, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,
    #            },
    #  },
    # {sc.ID: 30, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_SLOW_KW, sc.CHG_V_LIM: V_CHG_LIMIT_75, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #            sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,
    #            },
    #  },
    # {sc.ID: 31, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,
    #  sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_REN, sc.CHG_SOC_LOW: SOC_THS_25,
    #            sc.CHG_P: P_CHG_EV_AC_SLOW_KW, sc.CHG_V_LIM: V_CHG_LIMIT_60, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
    #            },
    #  sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
    #            },
    #  sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
    #            sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
    #            },
    #  sc.TRIP: {
    #      sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
    #      sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,
    #     },
    #  },
    {sc.ID: 32, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,  # full-time job, home+work PV
     sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_PV, sc.CHG_SOC_LOW: SOC_THS_40,
               sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_80, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
               },
     sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_PV, sc.CHG_SOC_LOW: SOC_THS_40,
               sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_80, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
               sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_FULL,
               },
     sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
               sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
               },
     sc.TRIP: {
         sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
         sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,
        },
     },
    {sc.ID: 33, sc.SIM_START: SIM_DATE_START_DEFAULT, sc.SIM_STOP: SIM_DATE_STOP_DEFAULT,  # part-time job, home PV
     sc.HOME: {sc.CHG_STRATEGY: sc.CHG_STRAT.V2G_OPT_PV, sc.CHG_SOC_LOW: SOC_THS_40,
               sc.CHG_P: P_CHG_EV_AC_STD_KW, sc.CHG_V_LIM: V_CHG_LIMIT_80, sc.CHG_I_CO: I_CHG_CUTOFF_C_20,
               },
     sc.WORK: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
               sc.DEPARTURE: HOME_TO_WORK_DEPARTURE_H_RANGE, sc.DURATION: WORK_DURATION_H_RANGE_HALF,  # part-time job!
               },
     sc.FREE: {sc.CHG_STRATEGY: sc.CHG_STRAT.NONE,
               sc.DEPARTURE: HOME_TO_FREE_DEPARTURE_H_RANGE, sc.DURATION: FREE_DURATION_H_RANGE,
               },
     sc.TRIP: {
         sc.CHG_P: P_CHG_EV_DC_FAST_KW, sc.CHG_V_LIM: V_CHG_LIMIT_95, sc.CHG_I_CO: I_CHG_CUTOFF_C_2,
         sc.DEPARTURE: TRIP_DEPARTURE_H_RANGE,
        },
     },

]


# --- plot settings ----------------------------------------------------------------------------------------------------
# the more settings are enabled, the larger the plot (not only the height, but more importantly, the file size! The
# html can easily have several GB, which may be hard to render in a browser. Downscaling would be possible, but prevents
# looking at driving patterns into detail. Dynamic resampling is only possible with a running Python server, not with
# a standalone html file.
PLOT_VI = False  # plot voltage and current
# 0: don't plot, 1: plot if used in charging strategy, 2: always plot
PLOT_REN = 2  # plot renewable energy generation, load, and residual load (historic values or projected data)
PLOT_EMISSIONS = 2  # plot specific emission of energy mix (historic data or roughly estimated values)
PLOT_PRICE = 2  # plot electricity price (historic data or roughly estimated values)
PLOT_FREQUENCY = 1  # plot grid frequency (historic data or roughly estimated values)
PLOT_PV_LOAD = 1  # plot load profile data with PV
if PLOT_VI:
    # RESULT_SUBPLOT_TITLES = ["Grid power", "Cell power", "Cell current", "Cell voltage", "Cell SoC",
    #                          "Cell temperature", "Remaining usable capacity"]
    RESULT_SUBPLOT_TITLES = [None, None, None, None, None, None, None]
    RESULT_SUBPLOT_YAXIS_TITLES = ["Grid power [kW]", "Cell power [W]", "Cell current [A]", "Cell voltage [V]",
                                   "Cell SoC [0..1]", "Bat. temp. [°C]", "Rem. capacity [Ah]"]
    RESULT_SUBPLOT_YAXIS_LIM = [None, None, None, None, None, None, [2.1, 3.1]]
else:
    # RESULT_SUBPLOT_TITLES = ["Grid power", "Cell power", "Cell SoC", "Cell temperature", "Remaining usable capacity"]
    RESULT_SUBPLOT_TITLES = [None, None, None, None, None]
    RESULT_SUBPLOT_YAXIS_TITLES = ["Grid power [kW]", "Cell power [W]", "Cell SoC [0..1]",
                                   "Bat. temp. [°C]", "Rem. capacity [Ah]"]
    RESULT_SUBPLOT_YAXIS_LIM = [None, None, None, None, [2.1, 3.1]]

RESULT_SUBPLOT_TITLES_MINIMAL = [None, None, None]
RESULT_SUBPLOT_YAXIS_TITLES_MINIMAL = ["Cell power [W]", "Cell SoC [0..1]", "Rem. capacity [Ah]"]
RESULT_SUBPLOT_YAXIS_LIM_MINIMAL = [None, None, [2.1, 3.1]]


TITLE_RE = ("%s - result (%s) - cell: Qc/d: %.2f/%.2f kAh (%.1f EFC), Ec/d: %.2f/%.2f kWh<br>"
            "grid: E<sub>c/d</sub>: %.0f/%.0f kWh, %.2f € (%.1f/%.1f ct/kWh), CO<sub>2</sub>: %.1f kg (%.0f/%.0f g/kWh)"
            ", avg. res. load: (%.1f/%.1f GW), %.0f km tot.")


# --- other constants --------------------------------------------------------------------------------------------------
COL_Q_LOSS_SEI = "q_loss_sei_total"
COL_Q_LOSS_CYC = "q_loss_cyclic_total"
COL_Q_LOSS_LOW = "q_loss_cyclic_low_total"
COL_Q_LOSS_PLA = "q_loss_plating_total"
COL_Q_CHG_TOTAL = "Q_chg_total"
COL_Q_DISCHG_TOTAL = "Q_dischg_total"
COL_E_CHG_TOTAL = "E_chg_total"
COL_E_DISCHG_TOTAL = "E_dischg_total"
I_COL_Q_LOSS_SEI = 0
I_COL_Q_LOSS_CYC = 1
I_COL_Q_LOSS_LOW = 2
I_COL_Q_LOSS_PLA = 3
I_COL_Q_CHG_TOTAL = 4
I_COL_Q_DISCHG_TOTAL = 5
I_COL_E_CHG_TOTAL = 6
I_COL_E_DISCHG_TOTAL = 7
COL_ARR_AGING_STATES = [COL_Q_LOSS_SEI, COL_Q_LOSS_CYC, COL_Q_LOSS_LOW, COL_Q_LOSS_PLA,
                        COL_Q_CHG_TOTAL, COL_Q_DISCHG_TOTAL, COL_E_CHG_TOTAL, COL_E_DISCHG_TOTAL]

COL_SCENARIO = "scenario"
COL_INPUT_DATA = "input_data"
COL_INPUT_DATA_T = "temp_ambient"
COL_INPUT_DATA_PRICE = "electricity_price"
COL_INPUT_DATA_EMISSIONS = "electricity_emissions"
COL_INPUT_DATA_FREQUENCY = "grid_frequency"
COL_INPUT_DATA_EL_GEN_DEM = "electricity_generation_and_demand"
COL_INPUT_DATA_LOAD_PROFILE = "load_profile"
# COL_INPUT_DATA_PV = "pv"
COL_DRIVING_DAYS = "driving days"
COL_DATE_START = "start date"
COL_DATE_STOP = "stop date"

FREQUENCY_CONTROL_RESOLUTION_S = 1  # in seconds, temporal resolution of freq. ctrl. (min: 1s = resolution of data set)
FREQUENCY_CONTROL_FREQ_NOMINAL = 50.0  # in Hz, nominal grid frequency
FREQUENCY_CONTROL_DEAD_BAND = 0.01  # in Hz, 0.01 = if abs(frequency deviation) < 10 mHz, set P_ctrl = 0 (don't act)
FREQUENCY_CONTROL_MAX_DELTA = 0.2  # in Hz, at this frequency deviation, the maximum power is injected/drawn
FREQUENCY_CONTROL_P_FAC = 1.0  # 0.5 = only use 50% of the standard charging power for frequency control
FREQUENCY_CONTROL_SOC_MIN = 0.25  # do not apply power in frequency control, when SoC is below this limit

BASE_SETTINGS_TEXT = \
    ("T_RESOLUTION_ACTIVE: %u, T_RESOLUTION_PROFILE: %u, T_RESOLUTION_REST: %u, CHG_OPTIMIZE_INTERVAL_S: %u\n"
     "CHG_EFFICIENCY: %.3f, PREFERENCE_TO_CHARGE: %.3f, PREFERENCE_TO_DISCHARGE_BASE_FACTOR: %.3f,\n"
     "PREFERENCE_TO_DISCHARGE: %s"
     % (T_RESOLUTION_ACTIVE, T_RESOLUTION_PROFILE, T_RESOLUTION_REST, CHG_OPTIMIZE_INTERVAL_S,
        CHG_EFFICIENCY, PREFERENCE_TO_CHARGE, PREFERENCE_TO_DISCHARGE_BASE_FACTOR, str(PREFERENCE_TO_DISCHARGE)))


def run():
    start_timestamp = datetime.datetime.now()
    logging.log.info(os.path.basename(__file__))
    report_manager = multiprocessing.Manager()
    report_queue = report_manager.Queue()

    logging.log.debug(BASE_SETTINGS_TEXT)

    run_models(report_queue)

    logging.log.info("\n\n========== All tasks ended - summary ==========\n")
    while True:
        if (report_queue is None) or report_queue.empty():
            break  # no more reports

        try:
            report_item = report_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        if report_item is None:
            break  # no more reports

        report_msg = report_item["msg"]
        report_level = report_item["level"]
        logging.log.log(level=report_level, msg=report_msg)

    stop_timestamp = datetime.datetime.now()
    logging.log.info("\nScript runtime: %s h:mm:ss.ms" % str(stop_timestamp - start_timestamp))


def run_models(report_queue):
    # check validity of scenario definition
    if not sc.validate_scenario_list(SCENARIO_LIST):
        return

    # load input data
    logging.log.info("Loading input data...")

    temp_ambient = input_data_helper.load_temperature_data(output_timezone=TIMEZONE)
    electricity_price = input_data_helper.load_electricity_price_data(output_timezone=TIMEZONE)
    electricity_emissions = input_data_helper.load_emission_data()
    load_profile = input_data_helper.load_load_profile_data()

    grid_frequency_used = False
    for scenario in SCENARIO_LIST:
        for loc in sc.LOCATION_ARRAY:
            if loc in scenario:
                sc_loc = scenario.get(loc)
                if sc.CHG_STRATEGY in sc_loc:
                    chg_strat = sc_loc.get(sc.CHG_STRATEGY)
                    if chg_strat == sc.CHG_STRAT.V2G_OPT_FREQ:
                        grid_frequency_used = True
                        break
        if grid_frequency_used:
            break
    if grid_frequency_used:
        grid_frequency = input_data_helper.load_freq_data(output_timezone=TIMEZONE)
    else:
        grid_frequency = None

    el_gen_dem_df = input_data_helper.load_el_gen_dem_data()
    input_data = {COL_INPUT_DATA_T: temp_ambient,
                  COL_INPUT_DATA_PRICE: electricity_price,
                  COL_INPUT_DATA_EMISSIONS: electricity_emissions,
                  COL_INPUT_DATA_FREQUENCY: grid_frequency,
                  COL_INPUT_DATA_LOAD_PROFILE: load_profile,
                  COL_INPUT_DATA_EL_GEN_DEM: el_gen_dem_df}

    if USE_COMMON_DRIVING_DAYS:
        date_start = SIM_DATE_START_DEFAULT
        date_stop = SIM_DATE_STOP_DEFAULT
        car_usage_days = drv.get_car_usage_days_v01(date_start, date_stop, TIMEZONE)  # init simulation period
        for scenario in SCENARIO_LIST:
            if sc.SIM_START in scenario:
                this_date_start = scenario.get(sc.SIM_START)
            else:
                this_date_start = SIM_DATE_START_DEFAULT
            if sc.SIM_STOP in scenario:
                this_date_stop = scenario.get(sc.SIM_STOP)
            else:
                this_date_stop = SIM_DATE_STOP_DEFAULT
            modeling_task_queue.put({COL_SCENARIO: scenario, COL_INPUT_DATA: input_data,
                                     COL_DRIVING_DAYS: car_usage_days,
                                     COL_DATE_START: this_date_start, COL_DATE_STOP: this_date_stop})
    else:
        for scenario in SCENARIO_LIST:
            modeling_task_queue.put({COL_SCENARIO: scenario, COL_INPUT_DATA: input_data})
    total_queue_size = modeling_task_queue.qsize()

    # Create processes
    processes = []
    logging.log.info("Starting processes to extend LOG data...")
    num_processors = min(NUMBER_OF_PROCESSORS_TO_USE, total_queue_size)
    for processor_number in range(0, num_processors):
        logging.log.debug("  Starting process %u" % processor_number)
        processes.append(multiprocessing.Process(
            target=modeling_thread, args=(processor_number, modeling_task_queue, report_queue, total_queue_size)))
    for processor_number in range(0, num_processors):
        processes[processor_number].start()
    for processor_number in range(0, num_processors):
        processes[processor_number].join()
        logging.log.debug("Joined process %u" % processor_number)


def modeling_thread(processor_number, job_queue, thread_report_queue, total_queue_size):
    time.sleep(1)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        try:
            remaining_size = job_queue.qsize()
            job = job_queue.get(block=False)
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if job is None:
            break  # no more files

        retry_counter = 0

        # --- initialize and load variables ----------------------------------------------------------------------------
        num_infos = 0
        num_warnings = 0
        num_errors = 0
        scenario = job[COL_SCENARIO]
        input_data = job[COL_INPUT_DATA]

        sc_id = scenario[sc.ID]
        # simulation_years = scenario[sc.SIM_YEARS]

        progress = 0.0
        if total_queue_size > 0:
            progress = (1.0 - remaining_size / total_queue_size) * 100.0
        logging.log.info("Thread %u scenario %u starting... (progress: %.1f %%)" % (processor_number, sc_id, progress))

        # --- scenario modeling ----------------------------------------------------------------------------------------
        cap_aged, aging_states, temp_cell, soc = bat.init()  # init battery
        if USE_COMMON_DRIVING_DAYS:
            car_usage_days = job[COL_DRIVING_DAYS]
            date_start = job[COL_DATE_START]
            date_stop = job[COL_DATE_STOP]
        else:
            # init simulation period
            date_start = scenario[sc.SIM_START]
            date_stop = scenario[sc.SIM_STOP]
            car_usage_days = drv.get_car_usage_days_v01(date_start, date_stop, TIMEZONE)
        # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = bat.init_empty_df()
        # p_grid_df = p_cell_df.copy()
        driving_distance = 0.0

        # get temperature data in region of interest (might use data of another year if year not available in input)
        temp_ambient_df = input_data[COL_INPUT_DATA_T]
        t_u0 = pd.Timestamp("1970-01-01", tz='UTC')
        datetime_min = car_usage_days.index[0]
        datetime_max = car_usage_days.index[-1] + datetime.timedelta(days=1)
        ts_min_roi = (datetime_min - t_u0) // pd.Timedelta("1s")
        ts_max_roi = (datetime_max - t_u0) // pd.Timedelta("1s")
        ts_resolution = temp_ambient_df.index[1] - temp_ambient_df.index[0]
        ts = np.arange(ts_min_roi, ts_max_roi, ts_resolution)
        temp_ambient_df = input_data_helper.get_temperature_data(temp_ambient_df, ts, True)

        # load_profile_df = input_data[COL_INPUT_DATA_LOAD_PROFILE]
        # ts_resolution = load_profile_df.index[1] - load_profile_df.index[0]
        # ts = np.arange(ts_min_roi, ts_max_roi, ts_resolution)
        # load_profile_df = input_data_helper.get_load_profile_data(temp_ambient_df, ts, True)

        grid_input_data = {COL_INPUT_DATA_T: input_data[COL_INPUT_DATA_T],
                           COL_INPUT_DATA_PRICE: input_data[COL_INPUT_DATA_PRICE],
                           COL_INPUT_DATA_EMISSIONS: input_data[COL_INPUT_DATA_EMISSIONS],
                           COL_INPUT_DATA_FREQUENCY: input_data[COL_INPUT_DATA_FREQUENCY],
                           COL_INPUT_DATA_LOAD_PROFILE: input_data[COL_INPUT_DATA_LOAD_PROFILE],
                           COL_INPUT_DATA_EL_GEN_DEM: input_data[COL_INPUT_DATA_EL_GEN_DEM]}
        # cap_aged_df = pd.Series(np.nan, index=car_usage_days.index)
        cap_aged_df = pd.Series(dtype=np.float64)
        # aging_states_df = pd.DataFrame(np.nan, columns=COL_ARR_AGING_STATES, index=car_usage_days.index)
        aging_states_df = pd.DataFrame(dtype=np.float64, columns=COL_ARR_AGING_STATES)
        # E_grid, el_cost, emissions,
        #   E_grid_chg, el_cost_chg, emissions_chg,
        #   E_grid_dischg, el_cost_dischg, emissions_dischg
        # el_cost(_chg/dischg) in ct, emissions(_chg/dischg) in g, E_grid(_chg/dischg) in kWh, residual/excess in GW
        grid_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        t_start = car_usage_days.index[0].timestamp()
        for date, car_usage_day_type in car_usage_days.items():
            this_date = date.date()
            if this_date < date_start:
                continue
            elif this_date > date_stop:
                break
            cap_aged_df.loc[t_start] = cap_aged
            aging_states_df.loc[t_start, COL_Q_LOSS_SEI] = aging_states[I_COL_Q_LOSS_SEI]
            aging_states_df.loc[t_start, COL_Q_LOSS_CYC] = aging_states[I_COL_Q_LOSS_CYC]
            aging_states_df.loc[t_start, COL_Q_LOSS_LOW] = aging_states[I_COL_Q_LOSS_LOW]
            aging_states_df.loc[t_start, COL_Q_LOSS_PLA] = aging_states[I_COL_Q_LOSS_PLA]

            aging_states_df.loc[t_start, COL_Q_CHG_TOTAL] = aging_states[I_COL_Q_CHG_TOTAL]
            aging_states_df.loc[t_start, COL_Q_DISCHG_TOTAL] = aging_states[I_COL_Q_DISCHG_TOTAL]
            aging_states_df.loc[t_start, COL_E_CHG_TOTAL] = aging_states[I_COL_E_CHG_TOTAL]
            aging_states_df.loc[t_start, COL_E_DISCHG_TOTAL] = aging_states[I_COL_E_DISCHG_TOTAL]

            # noinspection PyTypeChecker
            # (cap_aged, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, t_start,
            #  p_grid_df, grid_params, driving_distance, num_infos, num_warnings, num_errors) = \
            #     simulate_day(scenario, date, t_start, car_usage_day_type, temp_ambient_df, cap_aged, aging_states,
            #                  temp_cell, soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, p_grid_df,
            #                  grid_input_data, grid_params, driving_distance, num_infos, num_warnings, num_errors)
            (cap_aged, aging_states, temp_cell, soc, t_start, grid_params, driving_distance,
             num_infos, num_warnings, num_errors) = simulate_day(
                scenario, date, t_start, car_usage_day_type, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                grid_input_data, grid_params, driving_distance, num_infos, num_warnings, num_errors)

        # date = car_usage_days.index[-1] + datetime.timedelta(days=1)
        # date = pd.Timestamp(ts_input=p_cell_df.index[-1], tz=TIMEZONE, unit="s") + datetime.timedelta(days=1)
        # date = (pd.to_datetime(p_cell_df.index[-1], unit="s", origin='unix', utc=True).tz_convert(TIMEZONE)
        #         + datetime.timedelta(days=1))
        # date = pd.Timestamp(ts_input=p_cell_df.index[-1], tz=TIMEZONE, unit="s")  # use last timestamp
        cap_aged_df.loc[t_start] = cap_aged
        aging_states_df.loc[t_start, COL_Q_LOSS_SEI] = aging_states[I_COL_Q_LOSS_SEI]
        aging_states_df.loc[t_start, COL_Q_LOSS_CYC] = aging_states[I_COL_Q_LOSS_CYC]
        aging_states_df.loc[t_start, COL_Q_LOSS_LOW] = aging_states[I_COL_Q_LOSS_LOW]
        aging_states_df.loc[t_start, COL_Q_LOSS_PLA] = aging_states[I_COL_Q_LOSS_PLA]
        Qc_tot = aging_states[I_COL_Q_CHG_TOTAL]
        Qd_tot = aging_states[I_COL_Q_DISCHG_TOTAL]
        Ec_tot = aging_states[I_COL_E_CHG_TOTAL]
        Ed_tot = aging_states[I_COL_E_DISCHG_TOTAL]
        aging_states_df.loc[t_start, COL_Q_CHG_TOTAL] = Qc_tot
        aging_states_df.loc[t_start, COL_Q_DISCHG_TOTAL] = Qd_tot
        aging_states_df.loc[t_start, COL_E_CHG_TOTAL] = Ec_tot
        aging_states_df.loc[t_start, COL_E_DISCHG_TOTAL] = Ed_tot
        EFC_tot = (Qc_tot + Qd_tot) / 2.0 / bat.CAP_NOMINAL

        # find indexes of p_cell_df that are not available in p_grid_df -> fill them with 0
        # p_cell_ixs = p_cell_df.index
        # p_grid_ixs = p_grid_df.index
        # new_ixs = p_cell_ixs[~p_cell_ixs.isin(p_grid_ixs)]
        # all_ixs = p_grid_ixs.union(new_ixs)
        # p_grid_df = p_grid_df.reindex(all_ixs)
        # p_grid_df[new_ixs] = 0.0
        # p_grid_df.sort_index(inplace=True)

        # el_cost, emissions, E_grid_chg, el_cost_chg, emissions_chg, E_grid_dischg, el_cost_dischg, emissions_dischg
        # el_cost(_chg/dischg) in ct, emissions(_chg/dischg) in g, E_grid_chg/dischg in kWh
        (E_grid, el_cost, emissions, E_grid_chg, el_cost_chg, emissions_chg, t_residual_chg_s, residual_chg,
         E_grid_dischg, el_cost_dischg, emissions_dischg, t_residual_dischg_s, residual_dischg) = grid_params
        if E_grid_chg != 0.0:
            el_cost_chg_avg = el_cost_chg / E_grid_chg  # in ct/kWh
            emissions_chg_avg = emissions_chg / E_grid_chg  # in gCO2eq/kWh
        else:
            el_cost_chg_avg = 0.0
            emissions_chg_avg = 0.0
        if E_grid_dischg != 0.0:
            el_cost_dischg_avg = el_cost_dischg / E_grid_dischg  # in ct/kWh
            emissions_dischg_avg = emissions_dischg / E_grid_dischg  # in gCO2eq/kWh
        else:
            el_cost_dischg_avg = 0.0
            emissions_dischg_avg = 0.0
        el_cost_EUR = el_cost / 100.0  # ct in €
        emissions_kg = emissions / 1000.0  # g in kg

        avg_residual_chg_GW = 0.0  # average residual load (>0) or excess energy (<0) when charging or discharging
        if t_residual_chg_s != 0.0:
            avg_residual_chg_GW = residual_chg / t_residual_chg_s
        avg_residual_dischg_GW = 0.0
        if t_residual_dischg_s != 0.0:
            avg_residual_dischg_GW = residual_dischg / t_residual_dischg_s

        Qc_tot_kAh = Qc_tot / 1000.0
        Qd_tot_kAh = Qd_tot / 1000.0
        Ec_tot_kWh = Ec_tot / 1000.0
        Ed_tot_kWh = Ed_tot / 1000.0

        # summary
        use_case_name = USE_CASE_NAME % sc_id
        run_timestring = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        # %s - result (%s) - cell: Qc/d: %.2f/%.2f kAh (%.1f EFC), Ec/d: %.2f/%.2f kWh<br>
        # grid: E<sub>c/d</sub>: %.0f/%.0f kWh, %.2f € (%.1f/%.1f ct/kWh), CO<sub>2</sub>: %.1f kg (%.0f/%.0f g/kWh)
        # , avg. res. load: (%.1f/%.1f GW), %.0f km tot.
        plot_title = TITLE_RE % (use_case_name, run_timestring, Qc_tot_kAh, Qd_tot_kAh, EFC_tot, Ec_tot_kWh, Ed_tot_kWh,
                                 E_grid_chg, E_grid_dischg, el_cost_EUR, el_cost_chg_avg, el_cost_dischg_avg,
                                 emissions_kg, emissions_chg_avg, emissions_dischg_avg,
                                 avg_residual_chg_GW, avg_residual_dischg_GW, driving_distance)

        # --- print result to console/log in case saving doesn't work (e.g., because plot too large) -------------------
        plot_title_details = sc.get_scenario_subtitle(scenario)
        log_stat = ("Remaining capacity: %.4f Ah (simulation from %s to %s)"
                    % (cap_aged, str(date_start), str(date_stop)))
        result_string = ("\n   %s\n   %s   %s\n" % (plot_title.replace("<br>", "\n   "),
                                                    plot_title_details.replace("<br>", "\n   "), log_stat))
        logging.log.debug(("Scenario %u:" % sc_id) + result_string)

        # --- save result data to csv ----------------------------------------------------------------------------------
        filename_base = EXPORT_FILENAME_BASE % sc_id + "_" + run_timestring
        export_filename_csv = filename_base + ".csv"
        # csv_dataframes = [p_grid_df, p_cell_df, i_cell_df, v_cell_df, soc_df, temp_cell_df, cap_aged_df]
        # csv_keys = ["P_grid [kW]", "P_cell [W]", "I_cell [A]", "V_cell [V]", "SoC_cell [0..1]",
        #             "T_cell [degC]", "Remaining capacity [Ah]"]
        csv_dataframes = [cap_aged_df]
        csv_keys = ["Remaining capacity [Ah]"]
        # csv_ixs = p_grid_df.index
        # csv_ixs = cap_aged_df.index

        # ---------- optional .csv exports - comment out if not needed ----------
        # # 0. generate residual load
        # scale_shift_years = 0
        # if sc.SHIFT_BY_YEARS in scenario:
        #     scale_shift_years = scenario.get(sc.SHIFT_BY_YEARS)
        # gen_dem_in_df = grid_input_data.get(COL_INPUT_DATA_EL_GEN_DEM)
        # gen_dem_df = input_data_helper.get_el_gen_dem_data(gen_dem_in_df, csv_ixs, scale_shift_years=scale_shift_years)
        # residual_df = gen_dem_df[input_data_helper.RESIDUAL_LOAD]

        # # 1. save electricity price/cost
        # price_df = grid_input_data.get(COL_INPUT_DATA_PRICE)
        # price_df = input_data_helper.get_price_data(price_df, csv_ixs, residual_load=residual_df)
        # csv_dataframes.append(price_df)
        # csv_keys.append("Electricity price [ct/kWh]")

        # # 2. save emissions
        # emission_df = grid_input_data.get(COL_INPUT_DATA_EMISSIONS)
        # emission_df = input_data_helper.get_emission_data(emission_df, csv_ixs, residual_load=residual_df)
        # csv_dataframes.append(emission_df)
        # csv_keys.append("Emissions [gCO2eq/kWh]")

        # # 3. save renewables and residual load
        # biomass_df = gen_dem_df[input_data_helper.GEN_BIOMASS]
        # hydro_df = gen_dem_df[input_data_helper.GEN_HYDRO]
        # wind_onshore_df = gen_dem_df[input_data_helper.GEN_WIND_OFFSHORE]
        # wind_offshore_df = gen_dem_df[input_data_helper.GEN_WIND_ONSHORE]
        # pv_df = gen_dem_df[input_data_helper.GEN_PV]
        # load_df = gen_dem_df[input_data_helper.DEMAND]
        # csv_dataframes.extend([biomass_df, hydro_df, wind_onshore_df, wind_offshore_df, pv_df, load_df, residual_df])
        # csv_keys.extend(["Biomass [GW]", "Hydropower [GW]", "Wind onshore [GW]", "Wind offshore [GW]", "PV [GW]",
        #                  "Demand [GW]", "Residual load [GW]"])

        # # 4. save frequency (doesn't need 0. / residual load)
        # frequency_df = grid_input_data.get(COL_INPUT_DATA_FREQUENCY)
        # frequency_df = input_data_helper.get_freq_data(frequency_df, csv_ixs)
        # csv_dataframes.append(frequency_df)
        # csv_keys.append("Grid frequency [Hz]")

        # # 5. save local PV power and demand (load profile)
        # pv_df = input_data_helper.get_el_gen_pv_data(gen_dem_in_df, csv_ixs) * PV_POWER_PEAK_KW
        # load_profile_df = grid_input_data.get(COL_INPUT_DATA_LOAD_PROFILE)
        # load_df = input_data_helper.get_load_profile_data(load_profile_df, csv_ixs)
        # csv_dataframes.extend([pv_df, load_df])
        # csv_keys.extend(["PV system [kW]", "Load profile [kW]"])

        # ---------- end of optional .csv exports ----------

        data_df = pd.concat(csv_dataframes, axis=1,
                            keys=csv_keys)
        data_df = pd.concat([data_df, aging_states_df], axis=1)
        data_df.to_csv(EXPORT_PATH + export_filename_csv, index=True, index_label="timestamp",
                       sep=";", float_format="%.4f")  # , na_rep="nan")

        # # --- evaluate what to plot ------------------------------------------------------------------------------------
        # chg_strat_arr = []
        # for loc in sc.LOCATION_ARRAY:
        #     if loc in scenario:
        #         sc_loc = scenario.get(loc)
        #         if sc.CHG_STRATEGY in sc_loc:
        #             chg_strat_arr.append(sc_loc.get(sc.CHG_STRATEGY))
        #
        # plot_ren = False
        # if PLOT_REN == 2:
        #     plot_ren = True
        # elif PLOT_REN == 1:
        #     for chg_strat in chg_strat_arr:
        #         if (chg_strat == sc.CHG_STRAT.V1G_OPT_REN) or (chg_strat == sc.CHG_STRAT.V2G_OPT_REN):
        #             plot_ren = True
        #             break
        #
        # plot_emissions = False
        # if PLOT_EMISSIONS == 2:
        #     plot_emissions = True
        # elif PLOT_EMISSIONS == 1:
        #     for chg_strat in chg_strat_arr:
        #         if (chg_strat == sc.CHG_STRAT.V1G_OPT_EMISSION) or (chg_strat == sc.CHG_STRAT.V2G_OPT_EMISSION):
        #             plot_emissions = True
        #             break
        #
        # plot_price = False
        # if PLOT_PRICE == 2:
        #     plot_price = True
        # elif PLOT_PRICE == 1:
        #     for chg_strat in chg_strat_arr:
        #         if (chg_strat == sc.CHG_STRAT.V1G_OPT_COST) or (chg_strat == sc.CHG_STRAT.V2G_OPT_COST):
        #             plot_price = True
        #             break
        #
        # plot_frequency = False
        # if PLOT_FREQUENCY == 2:
        #     plot_frequency = True
        # elif PLOT_FREQUENCY == 1:
        #     for chg_strat in chg_strat_arr:
        #         if chg_strat == sc.CHG_STRAT.V2G_OPT_FREQ:
        #             plot_frequency = True
        #             break
        #
        # plot_pv_load = False
        # if PLOT_PV_LOAD == 2:
        #     plot_pv_load = True
        # elif PLOT_PV_LOAD == 1:
        #     for chg_strat in chg_strat_arr:
        #         if chg_strat == sc.CHG_STRAT.V2G_OPT_PV:
        #             plot_pv_load = True
        #             break
        #
        # minimal_plot = False
        # if MINIMAL_FREQUENCY_CONTROL_PLOT:
        #     for chg_strat in chg_strat_arr:
        #         if chg_strat == sc.CHG_STRAT.V2G_OPT_FREQ:
        #             minimal_plot = True
        #             break
        #
        # if minimal_plot:
        #     subplot_titles = RESULT_SUBPLOT_TITLES_MINIMAL.copy()
        #     subplot_yaxis_titles = RESULT_SUBPLOT_YAXIS_TITLES_MINIMAL.copy()
        #     subplot_yaxis_lim = RESULT_SUBPLOT_YAXIS_LIM_MINIMAL.copy()
        # else:
        #     subplot_titles = RESULT_SUBPLOT_TITLES.copy()
        #     subplot_yaxis_titles = RESULT_SUBPLOT_YAXIS_TITLES.copy()
        #     subplot_yaxis_lim = RESULT_SUBPLOT_YAXIS_LIM.copy()
        #     if plot_ren:
        #         subplot_titles.append(None)
        #         subplot_yaxis_titles.append("Power [GW]")
        #         subplot_yaxis_lim.append(None)
        #     if plot_emissions:
        #         subplot_titles.append(None)
        #         subplot_yaxis_titles.append("Emiss. [gCO<sub>2,eq</sub>/kWh]")
        #         subplot_yaxis_lim.append(None)
        #     if plot_price:
        #         subplot_titles.append(None)
        #         subplot_yaxis_titles.append("Price [ct/kWh]")
        #         subplot_yaxis_lim.append(None)
        #     if plot_frequency:
        #         subplot_titles.append(None)
        #         subplot_yaxis_titles.append("Grid frequency [Hz]")
        #         subplot_yaxis_lim.append(None)
        #     if plot_pv_load:
        #         subplot_titles.append(None)
        #         subplot_yaxis_titles.append("PV/demand power [kW]")
        #         subplot_yaxis_lim.append(None)
        #
        # # --- generate plot --------------------------------------------------------------------------------------------
        # if OPEN_IN_BROWSER or EXPORT_HTML or ((EXPORT_IMAGE is not None) and (EXPORT_IMAGE != "")):
        #     # noinspection PyBroadException
        #     try:
        #         result_fig = result_plot.generate_base_figure(
        #             len(subplot_titles), 1, plot_title, subplot_titles, subplot_yaxis_titles,
        #             plot_title_details=plot_title_details, y_lim_arr=subplot_yaxis_lim)
        #
        #         x_data = pd.to_datetime(p_cell_df.index, unit="s", origin='unix', utc=True).tz_convert(TIMEZONE)
        #         x_data_cap = pd.to_datetime(cap_aged_df.index, unit="s", origin='unix', utc=True).tz_convert(TIMEZONE)
        #         # text_data_cap = x_data_cap.strftime('%Y-%m-%d %H:%M:%S')
        #         # text_data_cap = x_data_cap
        #         # text_data_cap = None
        #         i_row = 0
        #         if not minimal_plot:
        #             result_plot.add_result_trace(result_fig, i_row, 0, x_data, p_grid_df.values,
        #                                          result_plot.COLOR_P_GRID, False, True, None, TIMEZONE, False)
        #             i_row = i_row + 1
        #
        #         result_plot.add_result_trace(result_fig, i_row, 0, x_data, p_cell_df.values,
        #                                      result_plot.COLOR_P_CELL, False, True, None, TIMEZONE, False)
        #         i_row = i_row + 1
        #
        #         if minimal_plot:
        #             result_plot.add_result_trace(result_fig, i_row, 0, x_data, soc_df.values,
        #                                          result_plot.COLOR_SOC_CELL, False, True, None, TIMEZONE, False)
        #             i_row = i_row + 1
        #         elif PLOT_VI:
        #             result_plot.add_result_trace(result_fig, i_row, 0, x_data, i_cell_df.values,
        #                                          result_plot.COLOR_I_CELL, False, True, None, TIMEZONE, False)
        #             i_row = i_row + 1
        #             result_plot.add_result_trace(result_fig, i_row, 0, x_data, v_cell_df.values,
        #                                          result_plot.COLOR_V_CELL, False, True, None, TIMEZONE, False)
        #             i_row = i_row + 1
        #             result_plot.add_result_trace(result_fig, i_row, 0, x_data, soc_df.values,
        #                                          result_plot.COLOR_SOC_CELL, False, True, None, TIMEZONE, False)
        #             i_row = i_row + 1
        #             result_plot.add_result_trace(result_fig, i_row, 0, x_data, temp_cell_df.values,
        #                                          result_plot.COLOR_T_CELL, False, True, None, TIMEZONE, False)
        #             i_row = i_row + 1
        #         else:
        #             result_plot.add_result_trace(result_fig, i_row, 0, x_data, soc_df.values,
        #                                          result_plot.COLOR_SOC_CELL, False, True, None, TIMEZONE, False)
        #             i_row = i_row + 1
        #             result_plot.add_result_trace(result_fig, i_row, 0, x_data, temp_cell_df.values,
        #                                          result_plot.COLOR_T_CELL, False, True, None, TIMEZONE, False)
        #             i_row = i_row + 1
        #
        #         result_plot.add_result_trace(result_fig, i_row, 0, x_data_cap, cap_aged_df.values,
        #                                      result_plot.COLOR_SOH_CAP, True, True, None, TIMEZONE, False)
        #         i_row = i_row + 1
        #
        #         t_1 = p_cell_df.index[0]
        #         t_2 = p_cell_df.index[-1]
        #         if (plot_ren or plot_emissions or plot_price or plot_pv_load) and not minimal_plot:
        #             # el_gen_dem_roi = pd.Series(dtype=np.float64)
        #             emission_df = pd.Series(dtype=np.float64)
        #             price_df = pd.Series(dtype=np.float64)
        #             # pv_df = pd.Series(dtype=np.float64)
        #             # load_profile_df = pd.Series(dtype=np.float64)
        #
        #             scale_shift_years = 0
        #             if sc.SHIFT_BY_YEARS in scenario:
        #                 scale_shift_years = scenario.get(sc.SHIFT_BY_YEARS)
        #             gen_dem_df = grid_input_data.get(COL_INPUT_DATA_EL_GEN_DEM)
        #
        #             ren_dt = gen_dem_df.index[1] - gen_dem_df.index[0]
        #             ren_ixs = pd.Index(np.arange(t_1, t_2 + ren_dt, ren_dt))
        #             combined_ixs = ren_ixs.copy()
        #             emission_ixs = pd.Index([])
        #             price_ixs = pd.Index([])
        #             # load_profile_ixs = pd.Index([])
        #             # pv_ixs = pd.Index([])
        #             if plot_emissions:
        #                 emission_df = grid_input_data.get(COL_INPUT_DATA_EMISSIONS)
        #                 emission_dt = emission_df.index[1] - emission_df.index[0]
        #                 emission_ixs = pd.Index(np.arange(t_1, t_2 + emission_dt, emission_dt))
        #                 combined_ixs = combined_ixs.append(emission_ixs).drop_duplicates().sort_values()
        #             if plot_price:
        #                 price_df = grid_input_data.get(COL_INPUT_DATA_PRICE)
        #                 price_dt = price_df.index[1] - price_df.index[0]
        #                 price_ixs = pd.Index(np.arange(t_1, t_2 + price_dt, price_dt))
        #                 combined_ixs = combined_ixs.append(price_ixs).drop_duplicates().sort_values()
        #
        #             el_gen_dem_combined = input_data_helper.get_el_gen_dem_data(gen_dem_df, combined_ixs,
        #                                                                         scale_shift_years=scale_shift_years)
        #             residual_combined = el_gen_dem_combined[input_data_helper.RESIDUAL_LOAD]
        #             el_gen_dem_plot_ren = el_gen_dem_combined.loc[ren_ixs, :]
        #             if plot_ren:
        #                 result_plot.add_generation_and_demand_trace(result_fig, i_row, 0, el_gen_dem_plot_ren,
        #                                                             False, True, None, TIMEZONE, False)
        #                 i_row = i_row + 1
        #             if plot_emissions:
        #                 residual_emissions = residual_combined[emission_ixs]
        #                 emissions_plot = input_data_helper.get_emission_data(emission_df, emission_ixs,
        #                                                                      residual_load=residual_emissions)
        #                 em_x = pd.to_datetime(emissions_plot.index,
        #                                       unit="s", origin='unix', utc=True).tz_convert(TIMEZONE)
        #                 result_plot.add_result_trace(result_fig, i_row, 0, em_x, emissions_plot.values,
        #                                              result_plot.COLOR_EMISSIONS, False, True, None, TIMEZONE, False)
        #                 i_row = i_row + 1
        #             if plot_price:
        #                 residual_price = residual_combined[price_ixs]
        #                 price_plot = input_data_helper.get_price_data(price_df, price_ixs, residual_load=residual_price)
        #                 pr_x = pd.to_datetime(price_plot.index, unit="s", origin='unix', utc=True).tz_convert(TIMEZONE)
        #                 result_plot.add_result_trace(result_fig, i_row, 0, pr_x, price_plot.values,
        #                                              result_plot.COLOR_PRICE, False, True, None, TIMEZONE, False)
        #                 i_row = i_row + 1
        #             if plot_pv_load:
        #                 # load the load profile data
        #                 load_profile_df = grid_input_data.get(COL_INPUT_DATA_LOAD_PROFILE)
        #                 load_profile_dt = load_profile_df.index[1] - load_profile_df.index[0]
        #                 load_profile_ixs = pd.Index(np.arange(t_1, t_2 + load_profile_dt, load_profile_dt))
        #                 combined_ixs = combined_ixs.append(load_profile_ixs).drop_duplicates().sort_values()
        #                 combined_ixs = combined_ixs.append(p_grid_df.index).drop_duplicates().sort_values()
        #
        #                 # load PV base data, scale to PV_POWER_PEAK_KW kWp, generate data at the indexes that are needed
        #                 pv_plot = input_data_helper.get_el_gen_pv_data(gen_dem_df, combined_ixs) * PV_POWER_PEAK_KW
        #                 load_profile_plot = input_data_helper.get_load_profile_data(load_profile_df, combined_ixs)
        #
        #                 result_plot.add_pv_and_load_profile_trace(result_fig, i_row, 0, pv_plot, load_profile_plot,
        #                                                           p_grid_df, False, True, None, TIMEZONE, False)
        #
        #                 i_row = i_row + 1
        #         if plot_frequency and not minimal_plot:
        #             freq_df = grid_input_data[COL_INPUT_DATA_FREQUENCY]
        #             freq_dt = freq_df.index[1] - freq_df.index[0]
        #             freq_ixs = pd.Index(np.arange(t_1, t_2 + freq_dt, freq_dt))
        #             freq_plot = input_data_helper.get_freq_data(freq_df, freq_ixs)
        #             fr_x = pd.to_datetime(freq_plot.index, unit="s", origin='unix', utc=True).tz_convert(TIMEZONE)
        #             result_plot.add_result_trace(result_fig, i_row, 0, fr_x, freq_plot.values,
        #                                          result_plot.COLOR_FREQUENCY, False, True, None, TIMEZONE, False)
        #             # i_row = i_row + 1
        #
        #         result_plot.export_figure(result_fig, EXPORT_HTML, EXPORT_IMAGE, EXPORT_PATH, filename_base,
        #                                   OPEN_IN_BROWSER, append_date=False)
        #     except Exception:  # prevent program termination -> we want to continue with the other scenarios regardless
        #         num_errors = num_errors + 1
        #         logging.log.error("Scenario %u - Python Error during plot generation / export:\n%s"
        #                           % (sc_id, traceback.format_exc()))

        # --- reporting to main thread ---------------------------------------------------------------------------------
        report_msg = (f"%s - Scenario %u done - %u infos, %u warnings, %u errors%s"
                      % (filename_base, sc_id, num_infos, num_warnings, num_errors, result_string))
        report_level = logger.INFO
        if num_errors > 0:
            report_level = logger.ERROR
        elif num_warnings > 0:
            report_level = logger.WARNING
        logging.log.log(level=report_level, msg=report_msg)

        job_report = {"msg": report_msg, "level": report_level}
        thread_report_queue.put(job_report)

    modeling_task_queue.close()
    logging.log.info("Thread %u - no more jobs - exiting" % processor_number)


# def simulate_day(scenario, date, t_start, car_usage_day_type, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
#                  v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
#                  p_grid_df, grid_input_data, grid_params, driving_distance, num_infos, num_warnings, num_errors):
def simulate_day(scenario, date, t_start, car_usage_day_type, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                 grid_input_data, grid_params, driving_distance, num_infos, num_warnings, num_errors):
    sc_id = scenario[sc.ID]

    # if date.day == 1:
    logging.log.debug("Scenario %u - simulating %4u-%02u-%02u... (cap_aged: %.6f)"
                      % (sc_id, date.year, date.month, date.day, cap_aged))

    # t_start_beginning_of_day = date.timestamp()
    # if (v_cell_df.shape[0] == 0) or (v_cell_df.index[-1] < t_start_beginning_of_day):
    #     # t_start = t_start_beginning_of_day
    #     if t_start != t_start_beginning_of_day:
    #         logging.log.warning("Scenario %u - warning: t_start != t_start_beginning_of_day -> %u != %u"
    #                             % (sc_id, t_start, t_start_beginning_of_day))
    #         num_warnings = num_warnings + 1
    #     if v_cell_df.shape[0] > 0:
    #         if v_cell_df.index[-1] < (t_start_beginning_of_day - 3600.0):  # + datetime.timedelta(hours=-1)
    #             logging.log.warning("Scenario %u - warning: last timestep came too long ago" % sc_id)
    #             num_warnings = num_warnings + 1
    # else:
    #     logging.log.debug("Scenario %u - debug: last activity passed midnight on %04u-%02u-%02u..."
    #                       % (sc_id, date.year, date.month, date.day))

    if (car_usage_day_type == drv.day_type.WORK_DAY) and (sc.WORK in scenario):
        # (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
        #  p_grid_df, grid_params, num_infos, num_warnings, num_errors) = \
        #     simulate_two_trip_day(scenario, sc.WORK, date, t_start, temp_ambient_df, cap_aged, aging_states,temp_cell,
        #                           soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
        #                           p_grid_df, grid_input_data, grid_params, num_infos, num_warnings, num_errors)
        (cap_aged, aging_states, temp_cell, soc, t_start, grid_params, num_infos, num_warnings, num_errors) = \
            simulate_two_trip_day(scenario, sc.WORK, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell,
                                  soc, grid_input_data, grid_params, num_infos, num_warnings, num_errors)
        driving_distance = driving_distance + driving_distances.get(drv.day_type.WORK_DAY)
    elif (car_usage_day_type == drv.day_type.FREE_DAY) and (sc.FREE in scenario):
        # (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
        #  p_grid_df, grid_params, num_infos, num_warnings, num_errors) = \
        #     simulate_two_trip_day(scenario, sc.FREE, date, t_start, temp_ambient_df, cap_aged, aging_states,temp_cell,
        #                           soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
        #                           p_grid_df, grid_input_data, grid_params, num_infos, num_warnings, num_errors)
        (cap_aged, aging_states, temp_cell, soc, t_start, grid_params, num_infos, num_warnings, num_errors) = \
            simulate_two_trip_day(scenario, sc.FREE, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell,
                                  soc, grid_input_data, grid_params, num_infos, num_warnings, num_errors)
        driving_distance = driving_distance + driving_distances.get(drv.day_type.FREE_DAY)
    elif (car_usage_day_type == drv.day_type.TRIP_DAY) and (sc.TRIP in scenario):
        # (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
        #  p_grid_df, grid_params, num_infos, num_warnings, num_errors) = \
        #     simulate_trip_day(scenario, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
        #                       v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
        #                       p_grid_df, grid_input_data, grid_params, num_infos, num_warnings, num_errors)
        (cap_aged, aging_states, temp_cell, soc, t_start, grid_params, num_infos, num_warnings, num_errors) = \
            simulate_trip_day(scenario, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                              grid_input_data, grid_params, num_infos, num_warnings, num_errors)
        driving_distance = driving_distance + driving_distances.get(drv.day_type.TRIP_DAY)
    elif (car_usage_day_type == drv.day_type.NO_CAR_USE_DAY) and (sc.HOME in scenario):
        # do nothing - we will determine what to do in the current day in the iteration of the next day
        driving_distance = driving_distance + driving_distances.get(drv.day_type.NO_CAR_USE_DAY)
        # (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
        #  num_infos, num_warnings, num_errors) = \
        #     simulate_no_car_use_day(scenario, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
        #                             v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
        #                             num_infos, num_warnings, num_errors)
    else:
        logging.log.warning("Scenario %u - warning: unsupported/unimplemented day_type: %s"
                            % (sc_id, str(car_usage_day_type)))
        num_warnings = num_warnings + 1

    # return (cap_aged, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, t_start,
    #         p_grid_df, grid_params, driving_distance, num_infos, num_warnings, num_errors)
    return (cap_aged, aging_states, temp_cell, soc, t_start, grid_params, driving_distance,
            num_infos, num_warnings, num_errors)


# def simulate_two_trip_day(scenario, dst_loc, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
#                           v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
#                           p_grid_df, grid_input_data, grid_params, num_infos, num_warnings, num_errors):
def simulate_two_trip_day(scenario, dst_loc, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                          grid_input_data, grid_params, num_infos, num_warnings, num_errors):
    # we start the day at [Home], drive [Home -> Activity], stay at [Activity], drive [Activity -> Home], stay at [Home]
    sc_id = scenario[sc.ID]
    sc_dst = scenario[dst_loc]  # dst = destination, sc.WORK or sc.FREE
    dep_range_h = sc_dst[sc.DEPARTURE]
    dep_h, dep_m, dep_hour_float = drv.get_random_departure(dep_range_h)
    dst_rest_duration = drv.get_random_duration_s(sc_dst[sc.DURATION])
    if dst_loc == sc.WORK:
        driving_profile = DRIVING_PROFILE_WORK
    elif dst_loc == sc.FREE:
        driving_profile = DRIVING_PROFILE_FREE
    else:
        num_warnings = num_warnings + 1
        logging.log.warning("Scenario %s: unexpected destination %s for two-trip day -> using full WLTP profile"
                            % (sc_id, dst_loc))
        driving_profile = bat.wltp_profiles.full  # fallback

    # while the EV rests, do things according to HOME charging strategy until the earliest configured departure
    t_earliest_departure = drv.get_earliest_departure_unix_ts(date, dep_range_h, TIMEZONE)
    t_actual_departure = drv.get_earliest_departure_unix_ts(date, dep_hour_float, TIMEZONE)
    # (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
    #  p_grid_df, grid_params, num_infos, num_warnings, num_errors) = simulate_rest(
    #     scenario, sc.HOME, t_start, t_earliest_departure, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
    #     v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
    #     p_grid_df, grid_input_data, grid_params, num_infos, num_warnings, num_errors)
    (cap_aged, aging_states, temp_cell, soc, t_start, grid_params, num_infos, num_warnings, num_errors) = simulate_rest(
        scenario, sc.HOME, t_start, t_earliest_departure, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
        grid_input_data, grid_params, num_infos, num_warnings, num_errors)

    # wait until actual departure
    rest_duration = t_actual_departure - t_start
    # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
    #     bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df,
    #                     p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
        t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, cap_aged, aging_states, temp_cell, soc)

    # drive [Home -> Activity] = insert 1x driving_profile
    # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
    #     bat.apply_power_profile(t_start, T_RESOLUTION_PROFILE, driving_profile, temp_ambient_df, v_cell_df,
    #                             i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    cap_aged, aging_states, temp_cell, soc, t_start, _ = (
        bat.apply_power_profile(t_start, T_RESOLUTION_PROFILE, driving_profile, temp_ambient_df,
                                cap_aged, aging_states, temp_cell, soc))

    # while the EV rests, do things according to DESTINATION charging strategy until the earliest configured departure
    t_earliest_departure = drv.get_earliest_departure_from_hour_duration_s(t_start, sc_dst[sc.DURATION])
    t_actual_departure = drv.get_earliest_departure_from_second_duration_s(t_start, dst_rest_duration)
    # (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
    #  p_grid_df, grid_params, num_infos, num_warnings, num_errors) = simulate_rest(
    #     scenario, dst_loc, t_start, t_earliest_departure, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
    #     v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
    #     p_grid_df, grid_input_data, grid_params, num_infos, num_warnings, num_errors)
    (cap_aged, aging_states, temp_cell, soc, t_start, grid_params, num_infos, num_warnings, num_errors) = simulate_rest(
        scenario, dst_loc, t_start, t_earliest_departure, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
        grid_input_data, grid_params, num_infos, num_warnings, num_errors)

    # wait until actual departure
    rest_duration = t_actual_departure - t_start
    # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
    #     bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df,
    #                     p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
        t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, cap_aged, aging_states, temp_cell, soc)

    # drive [Activity -> Home] = insert 1x driving_profile
    # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
    #     bat.apply_power_profile(t_start, T_RESOLUTION_PROFILE, driving_profile, temp_ambient_df, v_cell_df,
    #                             i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    cap_aged, aging_states, temp_cell, soc, t_start, _ = (
        bat.apply_power_profile(t_start, T_RESOLUTION_PROFILE, driving_profile, temp_ambient_df,
                                cap_aged, aging_states, temp_cell, soc))

    # no more trips today - we will determine what to do in rest of the current day in the iteration of the next day

    # return (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
    #         p_grid_df, grid_params, num_infos, num_warnings, num_errors)
    return cap_aged, aging_states, temp_cell, soc, t_start, grid_params, num_infos, num_warnings, num_errors


# def simulate_trip_day(scenario, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
#                       v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
#                       p_grid_df, grid_input_data, grid_params, num_infos, num_warnings, num_errors):
def simulate_trip_day(scenario, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                      grid_input_data, grid_params, num_infos, num_warnings, num_errors):
    # we start the day at [Home/Remote], drive [Home/Remote -> Remote/Home] (+ charge on the way), stay at [Remote/Home]
    # for the sake of simplicity, treat both locations same, i.e., charging at remote is possible as if it was at home
    # sc_id = scenario[sc.ID]
    sc_trip = scenario[sc.TRIP]
    dep_range_h = sc_trip[sc.DEPARTURE]
    dep_h, dep_m, dep_hour_float = drv.get_random_departure(dep_range_h)

    # while the EV rests, do things according to HOME charging strategy until the earliest configured departure
    t_earliest_departure = drv.get_earliest_departure_unix_ts(date, dep_range_h, TIMEZONE)
    t_actual_departure = drv.get_earliest_departure_unix_ts(date, dep_hour_float, TIMEZONE)
    # (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
    #  p_grid_df, grid_params, num_infos, num_warnings, num_errors) = simulate_rest(
    #     scenario, sc.HOME, t_start, t_earliest_departure, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
    #     v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
    #     p_grid_df, grid_input_data, grid_params, num_infos, num_warnings, num_errors, True)
    (cap_aged, aging_states, temp_cell, soc, t_start, grid_params, num_infos, num_warnings, num_errors) = simulate_rest(
        scenario, sc.HOME, t_start, t_earliest_departure, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
        grid_input_data, grid_params, num_infos, num_warnings, num_errors, True)

    # wait until actual departure
    rest_duration = t_actual_departure - t_start
    # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
    #     bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df,
    #                     p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
        t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, cap_aged, aging_states, temp_cell, soc)

    # departure
    # for long distance trips, heat vehicle battery to 20°C if it is colder than that
    t_conditioning_trip = temp_ambient_df.loc[t_start:t_start + (36 * 60 * 60)].copy()
    t_conditioning_trip[t_conditioning_trip < TEMP_TRIP_MIN] = TEMP_TRIP_MIN
    # for charging, keep it at 32°C
    t_conditioning_fast_charging = TEMP_FAST_CHARGING

    n_remaining = DRIVING_PROFILE_TRIP_REPEAT
    # insert DRIVING_PROFILE_TRIP_REPEAT x DRIVING_PROFILE_TRIP, charge in between
    while True:
        # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc,t_start,n_rep =\
        #     bat.apply_power_profile_repeat(t_start, T_RESOLUTION_PROFILE, DRIVING_PROFILE_TRIP, n_remaining,
        #                                    t_conditioning_trip, None, TRIP_V_MIN, v_cell_df, i_cell_df, p_cell_df,
        #                                    temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)
        cap_aged, aging_states, temp_cell, soc, t_start, n_rep = bat.apply_power_profile_repeat(
            t_start, T_RESOLUTION_PROFILE, DRIVING_PROFILE_TRIP, n_remaining,
            t_conditioning_trip, None, TRIP_V_MIN, cap_aged, aging_states, temp_cell, soc)
        n_remaining = n_remaining - n_rep
        if n_remaining <= 0:
            break
        # else: fast charging
        t_chg_start = t_start
        chg_p_ev, chg_p_cell, chg_v_lim, chg_i_co = get_charging_ppvi(sc_trip)
        # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
        #     bat.apply_cp_cv(t_start, T_RESOLUTION_ACTIVE, chg_v_lim, chg_p_cell,chg_i_co,t_conditioning_fast_charging,
        #                     v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
        #                     cap_aged, aging_states, temp_cell, soc))
        cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df, _ = bat.apply_cp_cv(
            t_start, T_RESOLUTION_ACTIVE, chg_v_lim, chg_p_cell, chg_i_co, t_conditioning_fast_charging,
            None, cap_aged, aging_states, temp_cell, soc)

        # grid_params, p_grid_df = calc_grid_params_ex_ante(scenario, grid_params, grid_input_data, p_cell_df,p_grid_df,
        #                                                   t_chg_start, t_start)
        grid_params = calc_grid_params_ex_ante(scenario, grid_params, grid_input_data, p_cell_df, t_chg_start, t_start)

    # no more trips today - we will determine what to do in rest of the current day in the iteration of the next day

    # return (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
    #         p_grid_df, grid_params, num_infos, num_warnings, num_errors)
    return cap_aged, aging_states, temp_cell, soc, t_start, grid_params, num_infos, num_warnings, num_errors


# def simulate_no_car_use_day(scenario, date, t_start, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
#                             v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
#                             num_infos, num_warnings, num_errors):
#
#     # do nothing - we will determine what to do in the current day in the iteration of the next day
#
#     return (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
#             num_infos, num_warnings, num_errors)


# def simulate_rest(scenario, loc, t_start, t_earliest_departure, temp_ambient_df, cap_aged, aging_states,temp_cell,soc,
#                   v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, p_grid_df, grid_input_data, grid_params,
#                   num_infos, num_warnings, num_errors, trip_planned=False):
def simulate_rest(scenario, loc, t_start, t_earliest_departure, temp_ambient_df, cap_aged, aging_states, temp_cell, soc,
                  grid_input_data, grid_params, num_infos, num_warnings, num_errors, trip_planned=False):
    # determine where we are and what strategy to use
    sc_loc = scenario[loc]  # loc = location, sc.HOME, sc.WORK, or sc.FREE
    chg_strat_loc = sc_loc[sc.CHG_STRATEGY]

    allow_v2g = False
    if ((chg_strat_loc == sc.CHG_STRAT.V2G_OPT_EMISSION) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_COST)
            or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_REN) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_PV)):
        allow_v2g = True

    if trip_planned and not allow_v2g:
        chg_soc_low = 1.0  # for ..._IF_LOW charging strategies --> always charge
    elif sc.CHG_SOC_LOW in sc_loc:
        chg_soc_low = sc_loc[sc.CHG_SOC_LOW]
    else:
        chg_soc_low = 0.0  # not used

    t_when_charging = temp_ambient_df.loc[t_start:t_earliest_departure + (24 * 60 * 60)].copy()
    t_when_charging[t_when_charging < TEMP_CHARGING_MIN] = TEMP_CHARGING_MIN  # ToDo: can we make this "cost" something?

    t_chg_start = t_start

    p_cell_df = None  # added for fast model

    # determine what to do between right after the last trip and the next (earliest) departure
    if chg_strat_loc == sc.CHG_STRAT.NONE:
        pass  # do not charge
    elif ((chg_strat_loc == sc.CHG_STRAT.EARLY) or (chg_strat_loc == sc.CHG_STRAT.EARLY_IF_LOW)
          or (chg_strat_loc == sc.CHG_STRAT.LATE) or (chg_strat_loc == sc.CHG_STRAT.LATE_IF_LOW)
          or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_FREQ)):  # for freq. control, base charging strategy is LATE_IF_LOW
        need_to_charge = True
        if ((chg_strat_loc == sc.CHG_STRAT.EARLY_IF_LOW) or (chg_strat_loc == sc.CHG_STRAT.LATE_IF_LOW)
                or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_FREQ)):
            if soc > chg_soc_low:
                need_to_charge = False
        if need_to_charge:
            # get charging parameters
            chg_p_ev, chg_p_cell, chg_v_lim, chg_i_co = get_charging_ppvi(sc_loc, trip_planned)

            if ((chg_strat_loc == sc.CHG_STRAT.LATE) or (chg_strat_loc == sc.CHG_STRAT.LATE_IF_LOW)
                    or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_FREQ)):
                # estimate charging duration, add tolerance
                duration_chg_s = get_charging_duration_estimation(soc, chg_p_cell, chg_v_lim, cap_aged)
                t_chg_begin = t_earliest_departure - duration_chg_s

                if chg_strat_loc == sc.CHG_STRAT.V2G_OPT_FREQ:
                    # frequency control until t_chg_begin

                    # get freq in range [t_start, t_chg_begin)
                    ts = np.arange(t_start, t_chg_begin, FREQUENCY_CONTROL_RESOLUTION_S)
                    freq_roi = input_data_helper.get_freq_data(grid_input_data.get(COL_INPUT_DATA_FREQUENCY), ts)

                    # calculate power
                    df = freq_roi.copy() - FREQUENCY_CONTROL_FREQ_NOMINAL
                    df[abs(df) < FREQUENCY_CONTROL_DEAD_BAND] = 0.0
                    df[df > FREQUENCY_CONTROL_MAX_DELTA] = FREQUENCY_CONTROL_MAX_DELTA
                    df[df < -FREQUENCY_CONTROL_MAX_DELTA] = -FREQUENCY_CONTROL_MAX_DELTA
                    p_cell = df / FREQUENCY_CONTROL_MAX_DELTA * chg_p_cell * FREQUENCY_CONTROL_P_FAC

                    # apply power - ToDo: consider only using frequency control when T > TEMP_CHARGING_MIN
                    # (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc,
                    #  t_start) = bat.apply_power_profile(t_start, FREQUENCY_CONTROL_RESOLUTION_S, p_cell,
                    #                                    t_when_charging, v_cell_df, i_cell_df, p_cell_df, temp_cell_df,
                    #                                     soc_df, cap_aged, aging_states, temp_cell, soc)
                    # cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df = bat.apply_power_profile(
                    #     t_start, FREQUENCY_CONTROL_RESOLUTION_S, p_cell, t_when_charging,
                    #     cap_aged, aging_states, temp_cell, soc)
                    cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df = bat.apply_power_profile_soc_lim(
                        t_start, FREQUENCY_CONTROL_RESOLUTION_S, p_cell, t_when_charging,
                        cap_aged, aging_states, temp_cell, soc, FREQUENCY_CONTROL_SOC_MIN)
                else:
                    # Wait until t_chg_begin
                    wait_duration = t_chg_begin - t_start
                    # (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc,
                    #  t_start) = bat.apply_pause(
                    #     t_start, T_RESOLUTION_REST, wait_duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
                    #     temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)
                    cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
                        t_start, T_RESOLUTION_REST, wait_duration, temp_ambient_df,
                        cap_aged, aging_states, temp_cell, soc)
            # else:  # start charging as early as possible, i.e., right after arrival

            # start charging
            # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
            #     bat.apply_cp_cv(t_start, T_RESOLUTION_ACTIVE, chg_v_lim, chg_p_cell, chg_i_co, t_when_charging,
            #                     v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states,
            #                     temp_cell, soc, t_end_max=t_earliest_departure))
            cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df, _ = bat.apply_cp_cv(
                t_start, T_RESOLUTION_ACTIVE, chg_v_lim, chg_p_cell, chg_i_co, t_when_charging, None,
                cap_aged, aging_states, temp_cell, soc, t_end_max=t_earliest_departure)
        elif chg_strat_loc == sc.CHG_STRAT.V2G_OPT_FREQ:
            # frequency control until t_earliest_departure
            # get freq in range [t_start, t_earliest_departure)
            ts = np.arange(t_start, t_earliest_departure, FREQUENCY_CONTROL_RESOLUTION_S)
            freq_roi = input_data_helper.get_freq_data(grid_input_data.get(COL_INPUT_DATA_FREQUENCY), ts)

            # calculate power
            _, chg_p_cell, _, _ = get_charging_ppvi(sc_loc, trip_planned)
            df = freq_roi.copy() - FREQUENCY_CONTROL_FREQ_NOMINAL
            df[abs(df) < FREQUENCY_CONTROL_DEAD_BAND] = 0.0
            df[df > FREQUENCY_CONTROL_MAX_DELTA] = FREQUENCY_CONTROL_MAX_DELTA
            df[df < -FREQUENCY_CONTROL_MAX_DELTA] = -FREQUENCY_CONTROL_MAX_DELTA
            p_cell = df / FREQUENCY_CONTROL_MAX_DELTA * chg_p_cell * FREQUENCY_CONTROL_P_FAC

            # apply power - ToDo: consider only using frequency control when T > TEMP_CHARGING_MIN
            # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            #     bat.apply_power_profile(t_start, FREQUENCY_CONTROL_RESOLUTION_S, p_cell, t_when_charging, v_cell_df,
            #                             i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states,
            #                             temp_cell, soc)
            # cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df = bat.apply_power_profile(
            #     t_start, FREQUENCY_CONTROL_RESOLUTION_S, p_cell, t_when_charging,
            #     cap_aged, aging_states, temp_cell, soc)
            cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df = bat.apply_power_profile_soc_lim(
                t_start, FREQUENCY_CONTROL_RESOLUTION_S, p_cell, t_when_charging,
                cap_aged, aging_states, temp_cell, soc, FREQUENCY_CONTROL_SOC_MIN)
    elif ((chg_strat_loc == sc.CHG_STRAT.V1G_OPT_EMISSION) or (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_COST)
          or (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_REN) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_EMISSION)
          or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_COST) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_REN)):

        # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        #     smart_charging_old(scenario, sc_loc, chg_strat_loc, chg_soc_low, t_start, t_earliest_departure,
        #                        temp_ambient_df, t_when_charging, cap_aged, aging_states, temp_cell, soc, v_cell_df,
        #                        i_cell_df, p_cell_df, temp_cell_df, soc_df, grid_input_data, allow_v2g, trip_planned)

        # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        #     smart_charging(scenario, sc_loc, chg_strat_loc, chg_soc_low, t_start, t_earliest_departure,
        #                    temp_ambient_df, t_when_charging, cap_aged, aging_states, temp_cell, soc, v_cell_df,
        #                    i_cell_df, p_cell_df, temp_cell_df, soc_df, grid_input_data, allow_v2g, trip_planned)
        cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df = smart_charging(
            scenario, sc_loc, chg_strat_loc, chg_soc_low, t_start, t_earliest_departure, temp_ambient_df,
            t_when_charging, cap_aged, aging_states, temp_cell, soc, grid_input_data, allow_v2g, trip_planned)
    elif chg_strat_loc == sc.CHG_STRAT.V2G_OPT_PV:
        # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        #     solar_charging(sc_loc, chg_soc_low, t_start, t_earliest_departure,
        #                    temp_ambient_df, t_when_charging, cap_aged, aging_states, temp_cell, soc, v_cell_df,
        #                    i_cell_df, p_cell_df, temp_cell_df, soc_df, grid_input_data, trip_planned)
        cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df = solar_charging(
            sc_loc, chg_soc_low, t_start, t_earliest_departure, temp_ambient_df, t_when_charging, cap_aged,
            aging_states, temp_cell, soc, grid_input_data, trip_planned)
    else:
        num_warnings = num_warnings + 1
        logging.log.warning("Charging strategy %s not implemented for %s" % (chg_strat_loc, loc))

    # grid_params, p_grid_df = calc_grid_params_ex_ante(scenario, grid_params, grid_input_data, p_cell_df, p_grid_df,
    #                                                   t_chg_start, t_start)
    grid_params = calc_grid_params_ex_ante(scenario, grid_params, grid_input_data, p_cell_df, t_chg_start, t_start)

    # in case the process stopped before the earliest departure, wait until it
    rest_duration = t_earliest_departure - t_start
    # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = (
    #     bat.apply_pause(t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, v_cell_df, i_cell_df,
    #                     p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc))
    cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
        t_start, T_RESOLUTION_REST, rest_duration, temp_ambient_df, cap_aged, aging_states, temp_cell, soc)

    # return (v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start,
    #         p_grid_df, grid_params, num_infos, num_warnings, num_errors)
    return cap_aged, aging_states, temp_cell, soc, t_start, grid_params, num_infos, num_warnings, num_errors


# def smart_charging_old(scenario, sc_loc, chg_strat_loc, chg_soc_low, t_start, t_earliest_departure, temp_ambient_df,
#                        t_when_charging, cap_aged, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df,
#                        temp_cell_df, soc_df, grid_input_data, allow_v2g, trip_planned):
#     # slice into 5 minute intervals, aligned to hourly/15-min intervals -> first and last might be shorter
#     t_interval_arr_list = get_optimization_intervals(t_start, t_earliest_departure)
#     grid_conditions = calculate_grid_conditions(scenario, chg_strat_loc, grid_input_data, t_interval_arr_list)
#     chg_p_ev, chg_p_cell, chg_v_lim, chg_i_co = get_charging_ppvi(sc_loc, trip_planned)
#     v_lim_low = bat.get_ocv_from_soc(chg_soc_low)
#     num_intervals = len(t_interval_arr_list)
#     charging_urgency = pd.Series(0.0, index=range(num_intervals))  # by default, there is no urgency to charge
#     if trip_planned:
#         # calculate duration needed to charge, increase grid_condition to make it more likely to charge
#         duration_chg_s = get_charging_duration_estimation(chg_soc_low, chg_p_cell, chg_v_lim, cap_aged)
#         # t_chg_begin = t_earliest_departure - duration_chg_s * 3.0
#         min_charging_intervals = duration_chg_s / CHG_OPTIMIZE_INTERVAL_S
#         if min_charging_intervals >= num_intervals:
#             charging_urgency = pd.Series(1.0, index=range(num_intervals))  # no time for V2G or smart charging
#         else:
#             urgent_intervals = int(min_charging_intervals * 3.0)  # x3 to leave some space for V2G/smart charging
#             dx = 1.0 / float(urgent_intervals)
#             urgent_ser = pd.Series(np.arange(0.0, 1.0, dx) + dx,
#                                    index=np.arange(num_intervals - urgent_intervals, num_intervals))
#             charging_urgency.loc[:] = urgent_ser
#     # FIX ME better scheduling
#     for i in range(num_intervals):
#         t_interval = t_interval_arr_list[i]
#         grid_condition = grid_conditions.iloc[i] + charging_urgency.iloc[i]
#         # t_interval_start = t_interval[0]
#         t_interval_end = t_interval[1]
#         p_opt_cell = calculate_p_chg_opt(grid_condition, soc, allow_v2g, chg_soc_low, chg_p_ev)
#
#         if p_opt_cell > 0.0:  # charge
#             v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start \
#                 = bat.apply_cp_cv(t_start, T_RESOLUTION_ACTIVE, chg_v_lim, p_opt_cell, chg_i_co, t_when_charging,
#                                   v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states,
#                                   temp_cell, soc, t_end_max=t_earliest_departure)
#         elif p_opt_cell < 0.0:  # discharge
#             v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start \
#                 = bat.apply_cp_cv(t_start, T_RESOLUTION_ACTIVE, v_lim_low, p_opt_cell, -chg_i_co, temp_ambient_df,
#                                   v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states,
#                                   temp_cell, soc, t_end_max=t_earliest_departure)
#         # else: p_opt_cell == 0.0 -> do nothing
#
#         # wait until next interval (important in case charging/discharging stopped earlier - or if we did nothing)
#         duration = t_interval_end - t_start
#         v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
#             bat.apply_pause(t_start, T_RESOLUTION_REST, duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
#                             temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)
#
#     return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start


# def solar_charging(sc_loc, chg_soc_low, t_start, t_earliest_departure, temp_ambient_df,
#                    t_when_charging, cap_aged, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df,
#                    temp_cell_df, soc_df, grid_input_data, trip_planned):
def solar_charging(sc_loc, chg_soc_low, t_start, t_earliest_departure, temp_ambient_df,
                   t_when_charging, cap_aged, aging_states, temp_cell, soc, grid_input_data, trip_planned):
    load_profile_df = grid_input_data.get(COL_INPUT_DATA_LOAD_PROFILE)
    load_profile_dt = load_profile_df.index[1] - load_profile_df.index[0]
    # slice into load_profile_dt second intervals, aligned to load_profile_dt -> first and last might be shorter
    t_interval_arr_list = get_optimization_intervals(t_start, t_earliest_departure, dt=load_profile_dt)
    num_intervals = len(t_interval_arr_list)
    p_cell_df = None  # added for fast model
    if num_intervals <= 0:
        # return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start
        return cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df

    # get solar data
    t_interval_start_list = np.array(t_interval_arr_list)[:, 0].tolist()
    t_interval_end_list = np.array(t_interval_arr_list)[:, 1].tolist()
    t_interval_start_ser = pd.Series(t_interval_start_list, index=t_interval_start_list)
    t_analyze_from = t_interval_start_list[0]
    t_analyze_to = t_interval_start_list[-1] + 2 * load_profile_dt
    el_gen_dem_data_df = grid_input_data.get(COL_INPUT_DATA_EL_GEN_DEM)
    ixs_analyze = pd.Index(np.arange(t_analyze_from, t_analyze_to, load_profile_dt))
    t_analyze = pd.Series(ixs_analyze, index=ixs_analyze)
    t_fetch_ser = pd.concat([t_interval_start_ser, t_analyze], axis=0).drop_duplicates().sort_index()
    pv_df = input_data_helper.get_el_gen_pv_data(el_gen_dem_data_df, t_fetch_ser) * PV_POWER_PEAK_KW

    # get load profile data
    try:
        load_df = input_data_helper.get_load_profile_data(load_profile_df, t_fetch_ser)
    except Exception:  # prevent program termination -> we want to continue with the other scenarios regardless
        logging.log.error("Python Error in solar_charging:\n%s" % traceback.format_exc())
        print("debug")

    # get charging preferences
    chg_p_ev, chg_p_cell, chg_v_lim, chg_i_co = get_charging_ppvi(sc_loc, trip_planned)
    # soc_high = bat.get_soc_from_ocv(chg_v_lim)
    v_lim_low = bat.get_ocv_from_soc(chg_soc_low)

    # for each interval: analyze when to charge and when to discharge
    battery_full = False
    battery_empty = False
    charge_full_now = False  # once set True, do not set False again
    i_co_pv_chg = 0.0  # don't stop charging/discharging because of low currents

    # do interpolation once and not in every loop in bat.apply_cp_cv -> much faster
    t_start_ixs = pd.Series(t_interval_start_list)
    temp_when_charging_ser = bat.interpolate_df(t_when_charging, t_start_ixs)
    temp_when_charging_list = temp_when_charging_ser.tolist()
    temp_ambient_ser = bat.interpolate_df(temp_ambient_df, t_start_ixs)
    temp_ambient_list = temp_ambient_ser.tolist()

    # consider (dis)charging losses, since we want p_grid to match p_pv
    p_fac_chg = bat.wltp_profiles.P_EV_KW_TO_P_CELL_W * CHG_EFFICIENCY
    p_fac_dischg = bat.wltp_profiles.P_EV_KW_TO_P_CELL_W / CHG_EFFICIENCY
    # calculate and limit/edit delta_p_df
    delta_p_df = (pv_df[t_interval_start_list[0]:t_interval_start_list[-1]]
                  - load_df[t_interval_start_list[0]:t_interval_start_list[-1]])
    delta_p_df[delta_p_df > chg_p_ev] = chg_p_ev  # limit charging power
    cond_pv_chg = (delta_p_df >= PV_CHARGE_MIN_KW)
    if trip_planned:
        # only allow charging
        delta_p_df[~cond_pv_chg] = 0.0  # discharging, or charging power too small
        delta_p_df[cond_pv_chg] = delta_p_df[cond_pv_chg] * p_fac_chg
    else:
        # regular operation
        # only allow discharging if not trip_planned
        cond_pv_dischg = (delta_p_df <= PV_DISCHARGE_MIN_KW)
        delta_p_df[(~cond_pv_dischg) & (~cond_pv_chg)] = 0.0  # power too small
        delta_p_df[delta_p_df < -chg_p_ev] = -chg_p_ev  # limit discharging power
        delta_p_df[cond_pv_chg] = delta_p_df[cond_pv_chg] * p_fac_chg
        delta_p_df[cond_pv_dischg] = delta_p_df[cond_pv_dischg] * p_fac_dischg

    for i in range(num_intervals):
        # t_interval = t_interval_arr_list[i]
        # t_interval_start = t_interval[0]
        # t_interval_end = t_interval[1]

        t_interval_start = t_interval_start_list[i]
        t_interval_end = t_interval_end_list[i]

        # if t_start != t_interval_start:
        #     print("debug")

        if soc < chg_soc_low:
            # battery SoC under minimum limit -> charge with chg_p_cell / chg_v_lim for next interval,
            p_opt_cell = chg_p_cell
        else:
            if trip_planned:
                # estimate charging duration, add tolerance
                duration_chg_s = get_charging_duration_estimation(soc, chg_p_cell, chg_v_lim, cap_aged)
                t_chg_begin = t_earliest_departure - duration_chg_s
                if t_start >= t_chg_begin:  # it is necessary to start full now
                    charge_full_now = True
                    break
                # else: max continue with PV charging / discharging

            # P charging and household discharging -> charge or discharge or leave p_opt_cell = 0.0
            # # delta_p_grid_kW = pv_df[t_interval_start] - load_df[t_interval_start]
            # delta_p_grid_kW = delta_p_df[t_interval_start]
            # if delta_p_grid_kW > PV_CHARGE_MIN_KW:
            #     # more PV is generated than needed by household load --> charge?
            #     if not battery_full:
            #         # battery hasn't reached upper limit -> charge!
            #         # consider charging losses, since we want p_grid to match p_pv
            #         delta_p_grid_kW = min(delta_p_grid_kW, chg_p_ev)  # limit charging power to nominal charging power
            #         p_opt_cell = delta_p_grid_kW * p_fac_chg
            #     else:
            #         p_opt_cell = 0.0
            # elif (delta_p_grid_kW < PV_DISCHARGE_MIN_KW) and not trip_planned:
            #     # more household load than generated by PV --> discharge?
            #     # only allow discharging if not trip_planned
            #     if not battery_empty:
            #         # battery hasn't reached lower limit -> discharge!
            #         # consider discharging losses, since we want p_grid to match p_pv
            #         delta_p_grid_kW = max(delta_p_grid_kW, -chg_p_ev)  # limit to nominal discharging power
            #         p_opt_cell = delta_p_grid_kW * p_fac_dischg
            #     else:
            #         p_opt_cell = 0.0
            # else:
            #     p_opt_cell = 0.0
            # # else: leave at p_opt_cell = 0.0

            p_opt_cell = delta_p_df[t_interval_start]
            if p_opt_cell > 0.0:
                # more PV is generated than needed by household load --> charge?
                if battery_full:
                    # battery has reached upper limit -> stop charging!
                    p_opt_cell = 0.0
            elif p_opt_cell < 0.0:
                # more household load than generated by PV --> discharge?
                if battery_empty:
                    # battery has reached lower limit -> stop discharging!
                    p_opt_cell = 0.0

        if p_opt_cell > 0.0:  # charge
            # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start \
            #    = bat.apply_cp_cv(t_start, t_resolution_active, chg_v_lim, p_opt_cell, i_co_pv_chg,  # t_when_charging,
            #                       temp_when_charging_list[i],  # try to speed up simulation -> tested, works
            #                       v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states,
            #                       temp_cell, soc, t_end_max=t_interval_end)

            if (t_interval_start % T_RESOLUTION_ACTIVE) != 0:
                if (t_interval_start % 2) != 0:
                    t_resolution_active = 1
                else:
                    t_resolution_active = 2
            else:
                t_resolution_active = T_RESOLUTION_ACTIVE

            cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df, _ = bat.apply_cp_cv(
                t_start, t_resolution_active, chg_v_lim, p_opt_cell, i_co_pv_chg,  # t_when_charging,
                temp_when_charging_list[i],  # try to speed up simulation -> tested, works
                p_cell_df,
                cap_aged, aging_states, temp_cell, soc, t_end_max=t_interval_end)
            # if i_cell_df.iloc[-1] <= chg_i_co:  # charging stopped because cut-off current limit was reached
            #     battery_full = True
            if t_start < t_interval_end:  # charging stopped because cut-off current limit was reached
                battery_full = True
            battery_empty = False
        elif p_opt_cell < 0.0:  # discharge
            # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start \
            #     = bat.apply_cp_cv(t_start, t_resolution_active, v_lim_low, p_opt_cell, -i_co_pv_chg,# temp_ambient_df,
            #                       temp_ambient_list[i],  # try to speed up simulation -> tested, works
            #                       v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states,
            #                       temp_cell, soc, t_end_max=t_interval_end)

            if (t_interval_start % T_RESOLUTION_ACTIVE) != 0:
                if (t_interval_start % 2) != 0:
                    t_resolution_active = 1
                else:
                    t_resolution_active = 2
            else:
                t_resolution_active = T_RESOLUTION_ACTIVE

            cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df, _ = bat.apply_cp_cv(
                t_start, t_resolution_active, v_lim_low, p_opt_cell, -i_co_pv_chg,  # temp_ambient_df,
                temp_ambient_list[i],  # try to speed up simulation -> tested, works
                p_cell_df,
                cap_aged, aging_states, temp_cell, soc, t_end_max=t_interval_end)
            # if i_cell_df.iloc[-1] >= chg_i_co:  # discharging stopped because cut-off current limit was reached
            #     battery_empty = True
            if t_start < t_interval_end:  # discharging stopped because cut-off current limit was reached
                battery_empty = True
            battery_full = False
        # else: p_opt_cell == 0.0 -> do nothing

        # wait until next interval (important in case charging/discharging stopped earlier - or if we did nothing)
        duration = t_interval_end - t_start
        if duration > 0:
            # if i == (num_intervals - 1):  # this was likely a BUG - the modulo operation is crazy fast anyway,leave it
            if (duration % T_RESOLUTION_REST) != 0:
                if (duration % T_RESOLUTION_ACTIVE) != 0:
                    t_resolution_rest = 1
                else:
                    t_resolution_rest = T_RESOLUTION_ACTIVE
            else:
                t_resolution_rest = T_RESOLUTION_REST
            # else:
            #     t_resolution_rest = T_RESOLUTION_REST

            # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            #     bat.apply_pause(t_start, t_resolution_rest, duration, temp_ambient_df, v_cell_df, i_cell_df,p_cell_df,
            #                     temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)
            cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
                t_start, t_resolution_rest, duration, temp_ambient_df, cap_aged, aging_states, temp_cell, soc)

    if charge_full_now:
        # charge full from t_start to t_earliest_departure
        t_resolution_active = T_RESOLUTION_ACTIVE
        if (t_start % T_RESOLUTION_ACTIVE) != 0:
            t_resolution_active = 1
        # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start \
        #     = bat.apply_cp_cv(t_start, t_resolution_active, chg_v_lim, chg_p_cell, chg_i_co, t_when_charging,
        #                       v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states,
        #                       temp_cell, soc, t_end_max=t_earliest_departure)
        cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df, _ = bat.apply_cp_cv(
            t_start, t_resolution_active, chg_v_lim, chg_p_cell, chg_i_co, t_when_charging, p_cell_df,
            cap_aged, aging_states, temp_cell, soc, t_end_max=t_earliest_departure)

    # return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start
    return cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df


# def smart_charging(scenario, sc_loc, chg_strat_loc, chg_soc_low, t_start, t_earliest_departure, temp_ambient_df,
#                    t_when_charging, cap_aged, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df,
#                    temp_cell_df, soc_df, grid_input_data, allow_v2g, trip_planned):
def smart_charging(scenario, sc_loc, chg_strat_loc, chg_soc_low, t_start, t_earliest_departure, temp_ambient_df,
                   t_when_charging, cap_aged, aging_states, temp_cell, soc, grid_input_data, allow_v2g, trip_planned):

    # slice into 5 minute intervals, aligned to hourly/15-min intervals -> first and last might be shorter
    t_interval_arr_list = get_optimization_intervals(t_start, t_earliest_departure)
    num_intervals = len(t_interval_arr_list)
    p_cell_df = None  # added for fast model
    if num_intervals <= 0:
        # return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start
        return cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df

    grid_conditions = calculate_relative_grid_conditions(scenario, chg_strat_loc, grid_input_data, t_interval_arr_list)
    # -> grid_conditions -> 1.0 in best case (best to charge), 0.0 in worst case (worst to charge), limited to [0, 1]

    chg_p_ev, chg_p_cell, chg_v_lim, chg_i_co = get_charging_ppvi(sc_loc, trip_planned)
    soc_high = bat.get_soc_from_ocv(chg_v_lim)
    if chg_soc_low >= 1.0:
        soc_transformed = -1.0  # SOC_LOW = 100% -> always charge (and avoid division by zero)
    elif soc_high <= chg_soc_low:
        soc_transformed = 0.5  # invalid scenario configuration - fallback to prevent division by zero
    else:
        soc_transformed = (soc - chg_soc_low) / (soc_high - chg_soc_low)  # scaled to tolerated SoC range - may be < 0
    # -> e.g., chg_soc_low = 40%:
    #   soc = 100% -> soc_transformed = 100%  -> if grid_conditions = 1.0, preference_to_charge is 0
    #   soc =  70% -> soc_transformed =  50%  -> if grid_conditions = 0.5, preference_to_charge is 0
    #   soc =  40% -> soc_transformed =   0%  -> if grid_conditions = 0.0, preference_to_charge is 0
    #   soc =   0% -> soc_transformed = -67%
    preference_to_charge: pd.Series = grid_conditions - soc_transformed  # 0.0: no need to charge, 1.0: always charge

    # only allow charging if preference_to_charge > 0, but limit selected intervals to energy needed to charge to 100%
    BAT_CAP = bat.wltp_profiles.bat_capacity_kWh
    INTERVALS_PER_HOUR = 3600.0 / CHG_OPTIMIZE_INTERVAL_S
    soc_high = bat.get_soc_from_ocv(chg_v_lim)
    E_charge_max = (bat.get_soe_from_soc(soc_high) - bat.get_soe_from_soc(soc)) * BAT_CAP * 1.05  # 5% tolerance
    p_cell_interval_schedule = pd.Series(0.0, index=grid_conditions.index)

    if allow_v2g and not trip_planned:
        E_discharge_max = (bat.get_soe_from_soc(soc) - bat.get_soe_from_soc(chg_soc_low)) * BAT_CAP
        num_dischg_intervals_max = math.floor(E_discharge_max / chg_p_ev * INTERVALS_PER_HOUR)
        preference_to_discharge = PREFERENCE_TO_DISCHARGE_BASE_FACTOR
        if chg_strat_loc in PREFERENCE_TO_DISCHARGE:
            preference_to_discharge = preference_to_discharge * PREFERENCE_TO_DISCHARGE.get(chg_strat_loc)
        preference_dischg = preference_to_charge[preference_to_charge < -(1.0 - preference_to_discharge)]
        preference_dischg = preference_dischg.sort_values(ascending=True)  # lowest first
        if preference_dischg.shape[0] > num_dischg_intervals_max:
            preference_dischg = preference_dischg.iloc[0:(num_dischg_intervals_max + 1)]  # select first n items
        p_cell_interval_schedule[preference_dischg.index] = -chg_p_cell
    else:
        num_dischg_intervals_max = 0

    num_chg_intervals_max = math.ceil(E_charge_max / chg_p_ev * INTERVALS_PER_HOUR) + num_dischg_intervals_max
    preference_chg = preference_to_charge[preference_to_charge > (1.0 - PREFERENCE_TO_CHARGE)]
    preference_chg = preference_chg.sort_values(ascending=False)  # highest first
    if preference_chg.shape[0] > num_chg_intervals_max:
        preference_chg = preference_chg.iloc[0:(num_chg_intervals_max + 1)]  # select first n items
    p_cell_interval_schedule[preference_chg.index] = chg_p_cell

    v_lim_low = bat.get_ocv_from_soc(chg_soc_low)

    battery_full = False
    battery_empty = False
    for i in range(num_intervals):
        t_interval = t_interval_arr_list[i]
        t_interval_start = t_interval[0]
        t_interval_end = t_interval[1]
        if t_start != t_interval_start:
            print("debug")
        p_opt_cell = p_cell_interval_schedule.loc[t_interval_start]
        t_resolution_active = T_RESOLUTION_ACTIVE
        if (t_interval_start % T_RESOLUTION_ACTIVE) != 0:
            t_resolution_active = 1
        if (p_opt_cell > 0.0) and not battery_full:  # charge
            # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start \
            #     = bat.apply_cp_cv(t_start, t_resolution_active, chg_v_lim, p_opt_cell, chg_i_co, t_when_charging,
            #                       v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states,
            #                       temp_cell, soc, t_end_max=t_interval_end)
            cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df, cu_lim_reached = bat.apply_cp_cv(
                t_start, t_resolution_active, chg_v_lim, p_opt_cell, chg_i_co, t_when_charging, p_cell_df,
                cap_aged, aging_states, temp_cell, soc, t_end_max=t_interval_end)
            # if i_cell_df.iloc[-1] <= chg_i_co:  # charging stopped because cut-off current limit was reached
            if cu_lim_reached:  # charging stopped because cut-off current limit was reached
                battery_full = True
            battery_empty = False
        elif (p_opt_cell < 0.0) and not battery_empty:  # discharge
            # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start \
            #     = bat.apply_cp_cv(t_start, t_resolution_active, v_lim_low, p_opt_cell, -chg_i_co, temp_ambient_df,
            #                       v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states,
            #                       temp_cell, soc, t_end_max=t_interval_end)
            cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df, cu_lim_reached = bat.apply_cp_cv(
                t_start, t_resolution_active, v_lim_low, p_opt_cell, -chg_i_co, temp_ambient_df,
                p_cell_df,
                cap_aged, aging_states, temp_cell, soc, t_end_max=t_interval_end)
            # if i_cell_df.iloc[-1] >= chg_i_co:  # discharging stopped because cut-off current limit was reached
            if cu_lim_reached:  # discharging stopped because cut-off current limit was reached
                battery_empty = True
            battery_full = False
        # else: p_opt_cell == 0.0 -> do nothing

        # wait until next interval (important in case charging/discharging stopped earlier - or if we did nothing)
        duration = t_interval_end - t_start
        t_resolution_rest = T_RESOLUTION_REST
        if (duration % T_RESOLUTION_REST) != 0:
            t_resolution_rest = 1
        # v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        #     bat.apply_pause(t_start, t_resolution_rest, duration, temp_ambient_df, v_cell_df, i_cell_df, p_cell_df,
        #                     temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)
        cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
            t_start, t_resolution_rest, duration, temp_ambient_df, cap_aged, aging_states, temp_cell, soc)

    # return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start
    return cap_aged, aging_states, temp_cell, soc, t_start, p_cell_df


# def calc_grid_params_ex_ante(scenario, grid_params, grid_input_data, p_cell_df, p_grid_df, t_chg_start, t_next):
def calc_grid_params_ex_ante(scenario, grid_params, grid_input_data, p_cell_df, t_chg_start, t_next):
    if (p_cell_df is None) or (len(p_cell_df) == 0):  # added for the fast model
        return grid_params  # nothing new happened

    new_ixs = p_cell_df[p_cell_df.index > t_chg_start].index
    if len(new_ixs) == 0:
        # return grid_params, p_grid_df  # nothing new happened
        return grid_params  # nothing new happened

    # new_p_grid = p_cell_df[new_ixs].copy() * bat.wltp_profiles.P_CELL_W_TO_P_EV_KW
    # all_ixs = p_grid_df.index.union(new_ixs)
    # p_grid_df = p_grid_df.reindex(all_ixs)
    # p_grid_df[new_ixs] = new_p_grid  # test if this works
    # altered the code above --^ to the code below --v for the fast model
    new_p_grid = p_cell_df[new_ixs].copy() * bat.wltp_profiles.P_CELL_W_TO_P_EV_KW
    p_grid_df = new_p_grid

    dt_s = -pd.Series(new_ixs, index=new_ixs).diff(-1)  # time periods in which p_cell/grid_df are applied
    dt_s.iloc[-1] = t_next - new_ixs[-1]
    cond_chg = (p_grid_df > 0.0)
    cond_dischg = (p_grid_df < 0.0)

    # charging (P > 0) -> p_grid is higher because of charger losses
    p_grid_df.loc[new_ixs[cond_chg]] = p_grid_df[new_ixs[cond_chg]] / CHG_EFFICIENCY
    # discharging (P < 0) -> p_grid is lower because of charger losses
    p_grid_df.loc[new_ixs[cond_dischg]] = p_grid_df[new_ixs[cond_dischg]] * CHG_EFFICIENCY

    # determine residual load (might be required for emission and price estimation if no historic data is available
    scale_shift_years = 0
    if sc.SHIFT_BY_YEARS in scenario:
        scale_shift_years = scenario.get(sc.SHIFT_BY_YEARS)
    el_gen_dem_data_df = grid_input_data.get(COL_INPUT_DATA_EL_GEN_DEM)
    el_gen_dem_roi = input_data_helper.get_el_gen_dem_data(el_gen_dem_data_df, new_ixs,
                                                           scale_shift_years=scale_shift_years)
    residual_roi = el_gen_dem_roi[input_data_helper.RESIDUAL_LOAD]

    if scale_shift_years == 0:
        # we don't transform historic energy demand/generation to future scenarios --> use historic emissions if
        # available, use estimations if no data is available for the time
        emission_roi = input_data_helper.get_emission_data(grid_input_data.get(COL_INPUT_DATA_EMISSIONS), new_ixs,
                                                           residual_load=residual_roi)
        price_roi = input_data_helper.get_price_data(grid_input_data.get(COL_INPUT_DATA_PRICE), new_ixs)
    else:
        # we shift renewable generation installation capacity to the future
        # -> useless to use emission data of the past, use estimated data for all entries
        emission_roi = input_data_helper.get_emission_estimate_based_on_residual_load(residual_roi)
        price_roi = input_data_helper.get_price_estimate_based_on_residual_load(residual_roi)

    # calculate grid energy, CO2 emissions, and electricity price
    (E_grid, el_cost, emissions,
     E_grid_chg, el_cost_chg, emissions_chg, t_residual_chg_s, residual_chg,
     E_grid_dischg, el_cost_dischg, emissions_dischg, t_residual_dischg_s, residual_dischg) = grid_params

    # E_grid_df = p_grid_df[new_ixs] * dt_s / 3600.0  # kW * s / 3600 -> kWh
    E_grid_df = p_grid_df * dt_s / 3600.0  # kW * s / 3600 -> kWh
    E_grid_delta = E_grid_df.sum()
    E_grid_chg_delta = E_grid_df[cond_chg].sum()
    E_grid_dischg_delta = E_grid_df[cond_dischg].sum()
    E_grid = E_grid + E_grid_delta  # should be >> 0 in the long term (driving consumes energy + V2G energy losses)
    E_grid_chg = E_grid_chg + E_grid_chg_delta  # always >= 0
    E_grid_dischg = E_grid_dischg + E_grid_dischg_delta  # always <= 0

    emissions_df = E_grid_df * emission_roi  # kWh * gCO2eq/kWh -> gCO2eq. emissions are always > 0
    emissions_delta = emissions_df.sum()
    emissions_chg_delta = emissions_df[cond_chg].sum()  # energy > 0 for charging. we emit CO2
    emissions_dischg_delta = emissions_df[cond_dischg].sum()  # energy < 0 for discharging. if avoid CO2
    emissions = emissions + emissions_delta  # should be >> 0 in the long term
    emissions_chg = emissions_chg + emissions_chg_delta  # always >= 0
    emissions_dischg = emissions_dischg + emissions_dischg_delta  # always <= 0

    el_cost_df = E_grid_df * price_roi  # kWh * ct/kWh -> ct
    el_cost_delta = el_cost_df.sum()
    el_cost_chg_delta = el_cost_df[cond_chg].sum()  # energy > 0 for charging. if price > 0, we need to pay money
    el_cost_dischg_delta = el_cost_df[cond_dischg].sum()  # energy < 0 for discharging. if price > 0, we receive money
    el_cost = el_cost + el_cost_delta  # likely to be > 0 in the long term (we pay money)
    el_cost_chg = el_cost_chg + el_cost_chg_delta
    el_cost_dischg = el_cost_dischg + el_cost_dischg_delta

    t_residual_chg_s = t_residual_chg_s + dt_s[cond_chg].sum()
    t_residual_dischg_s = t_residual_dischg_s + dt_s[cond_dischg].sum()
    res_t_prod_chg = residual_roi[cond_chg] * dt_s[cond_chg]
    res_t_prod_dischg = residual_roi[cond_dischg] * dt_s[cond_dischg]
    residual_chg = residual_chg + res_t_prod_chg.sum()
    residual_dischg = residual_dischg + res_t_prod_dischg.sum()

    grid_params = (E_grid, el_cost, emissions,
                   E_grid_chg, el_cost_chg, emissions_chg, t_residual_chg_s, residual_chg,
                   E_grid_dischg, el_cost_dischg, emissions_dischg, t_residual_dischg_s, residual_dischg)
    # return grid_params, p_grid_df
    return grid_params


def get_charging_ppvi(sc_loc, is_before_trip=False):
    if is_before_trip:
        return get_charging_ppvi_before_trip(sc_loc)
    else:
        p_ev = sc_loc[sc.CHG_P]
        p_cell = p_ev * bat.wltp_profiles.P_EV_KW_TO_P_CELL_W
        return p_ev, p_cell, sc_loc[sc.CHG_V_LIM], sc_loc[sc.CHG_I_CO]


def get_charging_ppvi_before_trip(sc_loc):
    p_ev = sc_loc[sc.CHG_P]
    p_cell = p_ev * bat.wltp_profiles.P_EV_KW_TO_P_CELL_W
    return p_ev, p_cell, V_CHG_LIMIT_BEFORE_TRIP, I_CHG_CUTOFF_BEFORE_TRIP


def get_charging_duration_estimation(soc_now, p_chg, v_lim, cap_remaining):
    if (v_lim <= 0.0) or (p_chg <= 0.0):
        return 0  # invalid input
    soc_lim = bat.get_soc_from_ocv(v_lim)
    ocv_now = bat.get_ocv_from_soc(soc_now)
    d_soc = soc_lim - soc_now
    i_chg_avg = p_chg / ((v_lim + ocv_now) / 2.0)
    time_chg_s = d_soc * cap_remaining / i_chg_avg * 3600.0 + T_SCHEDULED_CHARGING_BUFFER_S
    if time_chg_s < 0.0:
        return 0.0
    return time_chg_s


def get_optimization_intervals(t_start, t_earliest_departure, dt=CHG_OPTIMIZE_INTERVAL_S):
    if t_earliest_departure <= t_start:
        return []  # no intervals
    t_interval_arr_list = []
    t_start_use = t_start
    rem = (t_start_use % dt)
    if rem != 0:
        t_2 = t_start_use + dt - rem
        if t_2 >= t_earliest_departure:
            t_interval_arr_list.append([t_start_use, t_earliest_departure])  # only one interval
            return t_interval_arr_list
        t_interval_arr_list.append([t_start_use, t_2])
        t_start_use = t_2

    t_end_use = t_earliest_departure
    rem = (t_end_use % dt)
    if rem != 0:
        t_1 = t_end_use - rem
        end_append = [t_1, t_end_use]
        t_end_use = t_1
    else:
        end_append = None

    interval_start = np.array(np.arange(t_start_use, t_end_use, dt))
    intervals_mid = np.transpose([interval_start, interval_start + dt])
    t_interval_arr_list.extend(intervals_mid.tolist())
    if end_append is not None:
        t_interval_arr_list.append(end_append)

    return t_interval_arr_list


def calculate_grid_conditions(scenario, chg_strat_loc, grid_input_data, t_interval_arr_list):
    t_interval_start_list = np.array(t_interval_arr_list)[:, 0].tolist()
    t_interval_start_ser = pd.Series(t_interval_start_list, index=t_interval_start_list)

    # determine residual load (might be required for emission and price estimation if no historic data is available
    scale_shift_years = 0
    if sc.SHIFT_BY_YEARS in scenario:
        scale_shift_years = scenario.get(sc.SHIFT_BY_YEARS)
    el_gen_dem_data_df = grid_input_data.get(COL_INPUT_DATA_EL_GEN_DEM)
    el_gen_dem_roi = input_data_helper.get_el_gen_dem_data(el_gen_dem_data_df, t_interval_start_ser,
                                                           scale_shift_years=scale_shift_years)
    residual_roi = el_gen_dem_roi[input_data_helper.RESIDUAL_LOAD]

    if (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_EMISSION) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_EMISSION):
        best_case, worst_case = 50, 350  # 50, 600  # average emissions of electricity mix in gCO2eq/kWh, lower = better
        grid_val_df = input_data_helper.get_emission_data(grid_input_data.get(COL_INPUT_DATA_EMISSIONS),
                                                          t_interval_start_ser, residual_load=residual_roi)
    elif (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_COST) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_COST):
        best_case, worst_case = -5, 8  # -5, 25  # electricity price in ct/kWh, lower = better
        grid_val_df = input_data_helper.get_price_data(grid_input_data.get(COL_INPUT_DATA_PRICE),
                                                       t_interval_start_ser, residual_load=residual_roi)
    elif (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_REN) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_REN):
        best_case, worst_case = -20, 30  # -20, 70  # residual load in GW, lower (or more negative = excess) = better
        grid_val_df = residual_roi
    else:
        print("Error: calculate_grid_conditions() not implemented for chg_strat_loc = %s" % str(chg_strat_loc))
        return None
    grid_conditions = (grid_val_df - worst_case) / (best_case - worst_case)
    grid_conditions[grid_conditions > 1.0] = 1.0
    grid_conditions[grid_conditions < 0.0] = 0.0
    return grid_conditions  # if worst_case -> 0, if best_case -> 1 (limited to [0, 1])


def calculate_relative_grid_conditions(scenario, chg_strat_loc, grid_input_data, t_interval_arr_list):
    t_interval_start_list = np.array(t_interval_arr_list)[:, 0].tolist()
    t_interval_start_ser = pd.Series(t_interval_start_list, index=t_interval_start_list)
    # FOR DEBUGGING:
    t_interval_end_list = np.array(t_interval_arr_list)[:, 1].tolist()
    t_interval_end_ser = pd.Series(t_interval_end_list, index=t_interval_start_list)
    # noinspection PyTypeChecker
    gap_less: pd.Series = (t_interval_start_ser == t_interval_end_ser.shift(1))
    gap_less.iloc[0] = True
    # noinspection PyTypeChecker
    if any(t_interval_end_ser <= t_interval_start_ser) or (not all(gap_less)):
        print("debug")
    # -------------
    t_analyze_from = t_interval_start_list[0] - 6 * 24 * 3600  # analyze data from 6 days ago ...
    t_analyze_to = t_interval_start_list[0] + 1 * 24 * 3600  # .. to 24 h ahead ("forecast")

    # determine residual load (might be required for emission and price estimation if no historic data is available
    scale_shift_years = 0
    if sc.SHIFT_BY_YEARS in scenario:
        scale_shift_years = scenario.get(sc.SHIFT_BY_YEARS)
    el_gen_dem_data_df = grid_input_data.get(COL_INPUT_DATA_EL_GEN_DEM)

    emission_df = pd.Series(dtype=np.float64)
    price_df = pd.Series(dtype=np.float64)
    if (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_EMISSION) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_EMISSION):
        # average emissions of electricity mix in gCO2eq/kWh, lower = better
        emission_df = grid_input_data.get(COL_INPUT_DATA_EMISSIONS)
        dt = emission_df.index[1] - emission_df.index[0]
    elif (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_COST) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_COST):
        price_df = grid_input_data.get(COL_INPUT_DATA_PRICE)
        dt = price_df.index[1] - price_df.index[0]
    elif (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_REN) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_REN):
        dt = el_gen_dem_data_df.index[1] - el_gen_dem_data_df.index[0]
    else:
        print("Error: calculate_grid_conditions() not implemented for chg_strat_loc = %s" % str(chg_strat_loc))
        return None
    ixs_analyze = pd.Index(np.arange(t_analyze_from, t_analyze_to, dt))
    t_analyze = pd.Series(ixs_analyze, index=ixs_analyze)
    t_fetch_ser = pd.concat([t_interval_start_ser, t_analyze], axis=0).drop_duplicates().sort_index()
    el_gen_dem_fetch = input_data_helper.get_el_gen_dem_data(el_gen_dem_data_df, t_fetch_ser,
                                                             scale_shift_years=scale_shift_years)
    el_gen_dem_roi = el_gen_dem_fetch.loc[t_interval_start_ser.index, :]
    residual_roi = el_gen_dem_roi[input_data_helper.RESIDUAL_LOAD]
    el_gen_dem_analyze = el_gen_dem_fetch.loc[t_analyze.index, :]
    residual_analyze = el_gen_dem_analyze[input_data_helper.RESIDUAL_LOAD]

    if (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_EMISSION) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_EMISSION):
        # average emissions of electricity mix in gCO2eq/kWh, lower = better
        analyze_df = input_data_helper.get_emission_data(emission_df, ixs_analyze, residual_load=residual_analyze)
        best_case, worst_case = get_grid_condition_thresholds(analyze_df, True)
        grid_val_roi = input_data_helper.get_emission_data(emission_df, t_interval_start_ser,
                                                           residual_load=residual_roi)
    elif (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_COST) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_COST):
        analyze_df = input_data_helper.get_price_data(price_df, ixs_analyze, residual_load=residual_analyze)
        best_case, worst_case = get_grid_condition_thresholds(analyze_df, True)
        grid_val_roi = input_data_helper.get_price_data(price_df, t_interval_start_ser, residual_load=residual_roi)
    elif (chg_strat_loc == sc.CHG_STRAT.V1G_OPT_REN) or (chg_strat_loc == sc.CHG_STRAT.V2G_OPT_REN):
        best_case, worst_case = get_grid_condition_thresholds(residual_analyze, True)
        grid_val_roi = residual_roi
    else:
        print("Error: calculate_grid_conditions() not implemented for chg_strat_loc = %s" % str(chg_strat_loc))
        return None
    grid_conditions = (grid_val_roi - worst_case) / (best_case - worst_case)
    grid_conditions[grid_conditions > 1.0] = 1.0
    grid_conditions[grid_conditions < 0.0] = 0.0
    return grid_conditions  # if worst_case -> 0, if best_case -> 1 (limited to [0, 1])


def get_grid_condition_thresholds(df, lower_is_better):
    if lower_is_better:
        best_case = df.min()  # e.g., 100
        worst_case = df.max()  # e.g., 600
        # diff > 0 --> best_case will be increased by 10% of the difference, worst case decreased
    else:
        best_case = df.max()  # e.g., 600
        worst_case = df.min()  # e.g., 100
        # diff < 0 --> best_case will be decreased by 10% of the difference, worst case increased
        #
    diff = worst_case - best_case
    # if 0 was the best and 1000 was the worst, actually use 100 (10%) as the best and 900 (90%) as worst
    margin = 0.0  # 0.1
    best_case = best_case + margin * diff
    worst_case = worst_case - margin * diff
    return best_case, worst_case


# def calculate_p_chg_opt(grid_condition, soc, allow_v2g, chg_soc_low, chg_p_ev):
#     # ToDO these limits may be adjusted
#     preference_to_charge = grid_condition - soc  # the higher, the more we like to charge
#     p_chg = 0  # default: no charging, no discharging
#     if (preference_to_charge > 0.01) or (soc < chg_soc_low):  # or 0.0?
#         if preference_to_charge > 0.25:
#             p_chg = chg_p_ev  # strong preference to charge
#         else:
#             p_chg = chg_p_ev/2.0  # small preference to charge
#     elif allow_v2g and (soc > chg_soc_low):
#         if preference_to_charge < 0.25:
#             if preference_to_charge < 0.4:
#                 p_chg = -chg_p_ev  # strong preference to discharge
#             else:
#                 p_chg = -chg_p_ev/2.0  # small preference to discharge
#     p_chg_cell = p_chg * bat.wltp_profiles.P_EV_KW_TO_P_CELL_W
#     return p_chg_cell


if __name__ == "__main__":
    run()
