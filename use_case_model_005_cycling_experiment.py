# Simulation of the battery aging experiment conducted for the dissertation:
# "Robust electricity grids through intelligent, highly efficient bidirectional charging systems for electric vehicles"
# (Dissertation of Matthias Luh, 2024)
# ... explained in chapter 7.1
# ToDo: if the figure closes immediately after opening, debug and set a breakpoint at the 'print("debug")' statement

# FIXME: Adjust the EXPORT_PATH before starting this script! An interactive .html will be exported into this folder.

import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime
from enum import IntEnum

import bat_model_v01 as bat
import result_plot


# class for the aging type (how is the cell operated/aged? calendar/cyclic/profile aging)
class age_type(IntEnum):
    CALENDAR = 0
    CYCLIC = 1
    PROFILE = 2


# used in the heading
AGE_TYPE_TITLES = {age_type.CALENDAR: "calendar aging",
                   age_type.CYCLIC: "cyclic aging",
                   age_type.PROFILE: "profile aging"
                   }

# used in the filename
AGE_TYPE_FILENAMES = {age_type.CALENDAR: "CAL",
                      age_type.CYCLIC: "CYC",
                      age_type.PROFILE: "PRF"
                      }

# constants
T_RESOLUTION_ACTIVE = 5  # in seconds, temporal resolution for modeling an active cell (charging, discharging)
T_RESOLUTION_PROFILE = 1  # in seconds, temporal resolution for modeling a profile aging cell (discharging) -> need 1s!
T_RESOLUTION_REST = 60  # in seconds, temporal resolution for modeling a resting cell (idle)

bat.R_TH_CELL = 3  # in K/W, rough estimation of the thermal resistance - active liquid cooling: 3

CELL_STORAGE_TIME_DAYS = 685  # Nov 26, 2020 to Oct 12, 2022 -> 685 days
# CELL_STORAGE_VOLTAGE = 3.5558  # in V, +/- 50 mV, voltage at which the cell was stored before experiment (as arrived)
CELL_STORAGE_SOC = 0.267  # in % (1 = 100%), +/- 1%, SoC at which the cell was stored before the experiment
CELL_STORAGE_TEMPERATURE = 18
EXPERIMENT_START_TIMESTAMP = 1665593100  # Mi, 12.10.2022  16:45:00 UTC

TEMP_RT = 25  # room temperature in °C (for check-up)
# N_CHECKUPS_MAX = 28  # for the experiment described in Chapter 7.1 of the dissertation
N_CHECKUPS_MAX = 3  # for testing and debugging (much shorter simulation time)
FIRST_CHECKUP_INTERVAL_S = (7 * 24 * 3600)  # interval between the first and the second CU
NEXT_CHECKUP_INTERVAL_S = (21 * 24 * 3600)  # interval between all following check-ups
CYCLING_PAUSE = 5 * 60  # in seconds, pause after charging/discharging operations

# general settings:
I_CHG_CAL = 1.0  # in A, charging current used for calendar aging cells (to reach the desired voltage)
I_DISCHG_CAL = -1.0  # in A, discharging current used for calendar aging cells (to reach the desired voltage)
I_CHG_CUTOFF_CYC = 0.3  # in A, cut-off current used for cyclic aging cells (CC-CV charging)
I_DISCHG_CUTOFF_CYC = -0.3  # in A, cut-off current used for cyclic aging cells (CC-CV discharging)
I_CHG_CUTOFF_CAL = 0.15  # in A, cut-off current used for calendar aging cells (CC-CV charging)
I_DISCHG_CUTOFF_CAL = -0.15  # in A, cut-off current used for calendar aging cells (CC-CV discharging)
C_REMAINING_CU_END = 0.5 * bat.CAP_NOMINAL  # run experiment until capacity < C_REMAINING_CU_END after CU

TEMP_OT_ARR = [0, 10, 25, 40]

# calendar aging cells
V_CAL_AGE_ARR = [3.3, 3.736, 4.089, 4.2]  # the cells shall rest at these voltages ...
SOC_CAL_AGE_ARR = [10, 50, 90, 100]  # ... roughly equivalent to this SoC (of a new cell)

# cyclic aging cells: cycling voltage range (and approximate SoC range)
V_CYC_MIN_MAX_AGE_ARR = [[2.5, 4.2], [3.249, 4.2], [3.249, 4.092]]  # 0-100, 10-100, 10-90
SOC_CYC_MIN_MAX_AGE_ARR = [[0, 100], [10, 100], [10, 90]]  # 0-100, 10-100, 10-90
SOC_CYC_MIN_MAX_TEXT_ARR = ["0-100 %", "10-100 %", "10-90 %"]  # for plot titles
I_CYC_DISCHG_CHG_ARR = [[-1.0, 1.0], [-3.0, 1.0], [-3.0, 3.0], [-3.0, 5.0]]  # discharging/charging current array
C_RATE_TEXT_ARR = ["+0.33 Cc / -0.33 Cd", "+0.33 Cc / -1.0 Cd", "+1.0 Cc / -1.0 Cd", "+1.67 Cc / -1.0 Cd"]  # for titles

# profile aging cells: cycling voltage range (and approximate SoC range)
V_PRF_MIN_MAX_AGE_ARR = [[3.249, 4.2], [3.249, 4.092], [3.249, 4.092]]  # 10-100, 10-90, 10-90
SOC_PRF_MIN_MAX_AGE_ARR = [[10, 100], [10, 90], [10, 90]]  # 10-100, 10-90, 10-90
PROFILE_AGE_ARR = [bat.wltp_profiles.full, bat.wltp_profiles.full, bat.wltp_profiles.extra_high]  # which profile to use
I_PRF_DISCHG_CHG_ARR = [[-1.0, 1.0], [-1.0, 1.0], [-3.0, 5.0]]  # how to discharge before CUs & how to chg. in operation
PROFILE_TEXT_ARR = ["10-100 %, +0.33 Cc, WLTP 3b", "10-90 %, +0.33 Cc, WLTP 3b", "10-90 %, +1.67 Cc, WLTP 3b high"]

# number of ... items
N_TEMP = len(TEMP_OT_ARR)  # number of temperatures in the experiment
N_SOC = len(V_CAL_AGE_ARR)  # number of SoCs for calendar aging cells
N_SOC_RANGES = len(V_CYC_MIN_MAX_AGE_ARR)  # number of SoC ranges for calendar aging cells
N_C_RATES = len(I_CYC_DISCHG_CHG_ARR)  # number of chg/dischg rate combinations for cyclic aging cells
N_PROFILES = len(PROFILE_AGE_ARR)  # number of (driving) power profiles used
N_TOTAL = N_TEMP * (N_SOC + N_SOC_RANGES * N_C_RATES + N_PROFILES)  # number of conditions/cells to simulate
# (since the simulation does not consider random effects and the experiment is idealized in the simulation, only one
# cell is tested per operating condition ("parameter set" in the dissertation)

# --- plot formatting --------------------------------------------------------------------------------------------------
EXPORT_PATH = "D:\\bat\\analysis\\use_case_models\\images\\"  # ToDo: adjust path! Please create the folder in advance.
OPEN_IN_BROWSER = True  # Do you want to open the resulting .html file in the browser?
EXPORT_HTML = True  # Do you want to export the resulting .html file for later usage? (recommended)
EXPORT_IMAGE = "png"  # Export image for later usage? Set to "png", "jpg", "svg", "pdf", or None (to skip image export)
EXPORT_FILENAME = "use_case_model_005_cycling_experiment_"  # filename of the exported image

# padding/margins for plot
SUBPLOT_H_SPACING_REL = 0.25  # 0.25  # 0.2  # 0.12  # 0.03, was 0.04
SUBPLOT_V_SPACING_REL = 0.35  # 0.3  # 0.35  # 0.21  # was 0.035
SUBPLOT_LR_MARGIN = 30
SUBPLOT_TOP_MARGIN = 130  # 120  # 0
SUBPLOT_BOT_MARGIN = 0
SUBPLOT_PADDING = 0

# plot height per row, plot width
HEIGHT_PER_ROW = 300  # in px
PLOT_WIDTH = 1350  # 1850  # in px
# PLOT_HEIGHT = HEIGHT_PER_ROW * SUBPLOT_ROWS -> we need to figure this out dynamically for each plot

# Title: %s -> aging type
PLOT_TITLE_RE = "<b>Usable discharge capacity [Ah] over time [days] – %s</b>"
PLOT_TITLE_Y_POS_REL = 20.0  # 30.0  # space for title above plot

# axis labels
X_AXIS_TITLE = 'Date'
Y_AXIS_TITLE = "Capacity [Ah]"

# mouse hover template
PLOT_HOVER_TEMPLATE_CU = "<b>%{text}</b><br>Remaining usable discharge capacity: %{y:.4f} Ah<br><extra></extra>"
PLOT_TEXT_CU = f"CU #%u at %s"

# if you want all plots to have the same, easily comparable y-axis limits, you can enable and configure them here
USE_MANUAL_PLOT_LIMITS_CAL = True  # False
USE_MANUAL_PLOT_LIMITS_CYC = True  # True
USE_MANUAL_PLOT_LIMITS_PRF = True  # False
MANUAL_PLOT_LIMITS_Y_CAL = [2.2, 3.1]  # [2.25, 3.0]
MANUAL_PLOT_LIMITS_Y_CYC = [1.4, 3.1]  # [1.42, 3.02]
MANUAL_PLOT_LIMITS_Y_PRF = [1.4, 3.1]  # [1.42, 3.02]

# font sizes
TITLE_FONT_SIZE = 17
SUBPLOT_TITLE_FONT_SIZE = 16
AXIS_FONT_SIZE = 16
AXIS_TICK_FONT_SIZE = 14

# figure styling template
FIGURE_TEMPLATE = "custom_theme"  # "custom_theme", "plotly_white", "plotly", "none", ...

# colors
BG_COLOR = 'rgba(255, 255, 255, 127)'  # '#fff'
MAJOR_GRID_COLOR = '#bbb'
MINOR_GRID_COLOR = '#e8e8e8'  # '#ddd'

COLOR_BLACK = 'rgb(0,0,0)'
COLOR_BLUE = 'rgb(29,113,171)'
COLOR_CYAN = 'rgb(22,180,197)'
COLOR_ORANGE = 'rgb(242,121,13)'
COLOR_RED = 'rgb(203,37,38)'

# line and marker styling, fill colors
TRACE_OPACITY = 0.8
TRACE_LINE_WIDTH = 1.5
MARKER_OPACITY = 0.8  # 75
MARKER_STYLE = dict(size=5, opacity=MARKER_OPACITY, line=None, symbol='circle')
AGE_FILL_COLORS = ['rgba(203,37,38,0.1)',  # q_loss_sei_total
                   'rgba(242,121,13,0.1)',  # q_loss_cyclic_total
                   'rgba(22,180,197)',  # q_loss_cyclic_low_total
                   'rgba(29,113,171, 0.1)']  # q_loss_plating_total
TEMPERATURE_COLORS = {0: COLOR_BLUE, 10: COLOR_CYAN, 25: COLOR_ORANGE, 40: COLOR_RED}

# create custom theme from default plotly theme
pio.templates["custom_theme"] = pio.templates["plotly"]
pio.templates["custom_theme"]['layout']['paper_bgcolor'] = BG_COLOR
pio.templates["custom_theme"]['layout']['plot_bgcolor'] = BG_COLOR
pio.templates["custom_theme"]['layout']['hoverlabel']['namelength'] = -1
pio.templates['custom_theme']['layout']['xaxis']['gridcolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['gridcolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['zerolinecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['xaxis']['linecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['linecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['zerolinecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['xaxis']['title']['standoff'] = 8
pio.templates['custom_theme']['layout']['yaxis']['title']['standoff'] = 8
pio.templates['custom_theme']['layout']['xaxis']['title']['font']['size'] = AXIS_FONT_SIZE
pio.templates['custom_theme']['layout']['yaxis']['title']['font']['size'] = AXIS_FONT_SIZE
pio.templates['custom_theme']['layout']['xaxis']['tickfont']['size'] = AXIS_TICK_FONT_SIZE
pio.templates['custom_theme']['layout']['yaxis']['tickfont']['size'] = AXIS_TICK_FONT_SIZE

# --- csv export settings (column labels) ------------------------------------------------------------------------------
COL_AGE_TYPE = "Aging type"
COL_TEMPERATURE = "Temperature [°C]"
COL_SOC_MIN = "SoC min [0..1]"
COL_SOC_MAX = "SoC max [0..1]"
COL_CHG_CURRENT = "Charging current [A]"
COL_DISCHG_CURRENT = "Discharging current [A]"
COL_PROFILE = "Profile"
COL_CAP_REMAINING = "Remaining capacity [Ah]"
COL_Q_LOSS_SEI = "q_loss_sei_total"
COL_Q_LOSS_CYC = "q_loss_cyclic_total"
COL_Q_LOSS_LOW = "q_loss_cyclic_low_total"
COL_Q_LOSS_PLA = "q_loss_plating_total"
COL_Q_CHG_TOTAL = "Q_chg_total"
COL_Q_DISCHG_TOTAL = "Q_dischg_total"
COL_E_CHG_TOTAL = "E_chg_total"
COL_E_DISCHG_TOTAL = "E_dischg_total"
CSV_EXPORT_COLUMNS = [COL_AGE_TYPE, COL_TEMPERATURE, COL_SOC_MIN, COL_SOC_MAX, COL_CHG_CURRENT, COL_DISCHG_CURRENT,
                      COL_PROFILE, COL_CAP_REMAINING, COL_Q_LOSS_SEI, COL_Q_LOSS_CYC, COL_Q_LOSS_LOW, COL_Q_LOSS_PLA,
                      COL_Q_CHG_TOTAL, COL_Q_DISCHG_TOTAL, COL_E_CHG_TOTAL, COL_E_DISCHG_TOTAL]


# main function
def run():
    # initialize arrays
    cap_remaining_arr = {  # remaining usable discharge capacity for all cells of the experiment
        age_type.CALENDAR: np.full((N_TEMP, N_SOC), None),
        age_type.CYCLIC: np.full((N_TEMP, N_SOC_RANGES, N_C_RATES), None),
        age_type.PROFILE: np.full((N_TEMP, N_PROFILES), None)
    }
    aging_states_arr = {  # aging types (SEI/cyclic I+II, plating) for all cells -> plotting not implemented yet
        age_type.CALENDAR: np.full((N_TEMP, N_SOC), None),
        age_type.CYCLIC: np.full((N_TEMP, N_SOC_RANGES, N_C_RATES), None),
        age_type.PROFILE: np.full((N_TEMP, N_PROFILES), None)
    }
    i = 0

    # prepare combined DataFrame (will export to csv)
    combined_df = pd.DataFrame(dtype=np.float64, columns=CSV_EXPORT_COLUMNS)
    combined_df_append_struct = pd.DataFrame(dtype=np.float64, columns=CSV_EXPORT_COLUMNS)

    # calendar aging
    for i_t in range(len(cap_remaining_arr.get(age_type.CALENDAR, []))):  # for all temperatures
        age_temp = TEMP_OT_ARR[i_t]
        for i_soc in range(N_SOC):  # for all SoCs/voltages
            age_v = V_CAL_AGE_ARR[i_soc]
            age_soc = SOC_CAL_AGE_ARR[i_soc]
            i = i + 1
            print("Simulating CAL %u°C %u%% (progress: %.1f %%)" % (age_temp, age_soc, i / N_TOTAL * 100))
            cap_aged_df, aging_states_df = run_calendar_aging(age_temp, age_v)
            cap_remaining_arr[age_type.CALENDAR][i_t][i_soc] = cap_aged_df
            aging_states_arr[age_type.CALENDAR][i_t][i_soc] = aging_states_df

            combined_df_append = combined_df_append_struct.copy()
            combined_df_append = combined_df_append.reindex(index=list(cap_aged_df.index.astype(int)))
            combined_df_append[COL_AGE_TYPE] = age_type.CALENDAR.value
            combined_df_append[COL_TEMPERATURE] = age_temp
            combined_df_append[COL_SOC_MIN] = age_soc
            combined_df_append[COL_SOC_MAX] = age_soc
            combined_df_append[COL_CHG_CURRENT] = 0.0
            combined_df_append[COL_DISCHG_CURRENT] = 0.0
            combined_df_append[COL_PROFILE] = -1
            combined_df_append[COL_CAP_REMAINING] = cap_aged_df.values
            combined_df_append[COL_Q_LOSS_SEI] = aging_states_df[COL_Q_LOSS_SEI]
            combined_df_append[COL_Q_LOSS_CYC] = aging_states_df[COL_Q_LOSS_CYC]
            combined_df_append[COL_Q_LOSS_LOW] = aging_states_df[COL_Q_LOSS_LOW]
            combined_df_append[COL_Q_LOSS_PLA] = aging_states_df[COL_Q_LOSS_PLA]
            combined_df_append[COL_Q_CHG_TOTAL] = aging_states_df[COL_Q_CHG_TOTAL]
            combined_df_append[COL_Q_DISCHG_TOTAL] = aging_states_df[COL_Q_DISCHG_TOTAL]
            combined_df_append[COL_E_CHG_TOTAL] = aging_states_df[COL_E_CHG_TOTAL]
            combined_df_append[COL_E_DISCHG_TOTAL] = aging_states_df[COL_E_DISCHG_TOTAL]
            combined_df = pd.concat([combined_df, combined_df_append], axis=0)

    # cyclic aging
    for i_t in range(len(cap_remaining_arr.get(age_type.CYCLIC, []))):  # for all temperatures
        age_temp = TEMP_OT_ARR[i_t]
        for i_soc_range in range(N_SOC_RANGES):  # for all SoC/voltage ranges
            age_v_range = V_CYC_MIN_MAX_AGE_ARR[i_soc_range]
            age_soc_range = SOC_CYC_MIN_MAX_AGE_ARR[i_soc_range]
            for i_c_rate in range(N_C_RATES):  # for all C-rate combinations
                age_c_rate = I_CYC_DISCHG_CHG_ARR[i_c_rate]
                i = i + 1
                print("Simulating CYC %u°C %u-%u%% +%u/-%u A (progress: %.1f %%)"
                      % (age_temp, age_soc_range[0], age_soc_range[1], age_c_rate[1], age_c_rate[0], i / N_TOTAL * 100))
                cap_aged_df, aging_states_df = run_cyclic_aging(age_temp, age_v_range, age_c_rate)
                cap_remaining_arr[age_type.CYCLIC][i_t][i_soc_range][i_c_rate] = cap_aged_df
                aging_states_arr[age_type.CYCLIC][i_t][i_soc_range][i_c_rate] = aging_states_df

                combined_df_append = combined_df_append_struct.copy()
                combined_df_append = combined_df_append.reindex(index=list(cap_aged_df.index.astype(int)))
                combined_df_append[COL_AGE_TYPE] = age_type.CYCLIC.value
                combined_df_append[COL_TEMPERATURE] = age_temp
                combined_df_append[COL_SOC_MIN] = age_soc_range[0]
                combined_df_append[COL_SOC_MAX] = age_soc_range[1]
                combined_df_append[COL_CHG_CURRENT] = age_c_rate[1]
                combined_df_append[COL_DISCHG_CURRENT] = age_c_rate[0]
                combined_df_append[COL_PROFILE] = -1
                combined_df_append[COL_CAP_REMAINING] = cap_aged_df.values
                combined_df_append[COL_Q_LOSS_SEI] = aging_states_df[COL_Q_LOSS_SEI]
                combined_df_append[COL_Q_LOSS_CYC] = aging_states_df[COL_Q_LOSS_CYC]
                combined_df_append[COL_Q_LOSS_LOW] = aging_states_df[COL_Q_LOSS_LOW]
                combined_df_append[COL_Q_LOSS_PLA] = aging_states_df[COL_Q_LOSS_PLA]
                combined_df_append[COL_Q_CHG_TOTAL] = aging_states_df[COL_Q_CHG_TOTAL]
                combined_df_append[COL_Q_DISCHG_TOTAL] = aging_states_df[COL_Q_DISCHG_TOTAL]
                combined_df_append[COL_E_CHG_TOTAL] = aging_states_df[COL_E_CHG_TOTAL]
                combined_df_append[COL_E_DISCHG_TOTAL] = aging_states_df[COL_E_DISCHG_TOTAL]
                combined_df = pd.concat([combined_df, combined_df_append], axis=0)

    # profile aging
    for i_t in range(len(cap_remaining_arr.get(age_type.PROFILE, []))):  # for all temperatures
        age_temp = TEMP_OT_ARR[i_t]
        for i_prf in range(N_PROFILES):  # for all driving profile/C-rate/SoC limit combinations
            age_v_range = V_PRF_MIN_MAX_AGE_ARR[i_prf]
            age_soc_range = SOC_PRF_MIN_MAX_AGE_ARR[i_prf]
            age_profile = PROFILE_AGE_ARR[i_prf]
            age_c_rate = I_PRF_DISCHG_CHG_ARR[i_prf]
            i = i + 1
            print("Simulating PRF %u°C #%u (%u-%u%% +%u A) (progress: %.1f %%)"
                  % (age_temp, i_prf, age_soc_range[0], age_soc_range[1], age_c_rate[1], i / N_TOTAL * 100))
            cap_aged_df, aging_states_df = run_profile_aging(age_temp, age_v_range, age_profile, age_c_rate)
            cap_remaining_arr[age_type.PROFILE][i_t][i_prf] = cap_aged_df
            aging_states_arr[age_type.PROFILE][i_t][i_prf] = aging_states_df

            combined_df_append = combined_df_append_struct.copy()
            combined_df_append = combined_df_append.reindex(index=list(cap_aged_df.index.astype(int)))
            combined_df_append[COL_AGE_TYPE] = age_type.PROFILE.value
            combined_df_append[COL_TEMPERATURE] = age_temp
            combined_df_append[COL_SOC_MIN] = age_soc_range[0]
            combined_df_append[COL_SOC_MAX] = age_soc_range[1]
            combined_df_append[COL_CHG_CURRENT] = age_c_rate[1]
            combined_df_append[COL_DISCHG_CURRENT] = age_c_rate[0]
            combined_df_append[COL_PROFILE] = i_prf
            combined_df_append[COL_CAP_REMAINING] = cap_aged_df.values
            combined_df_append[COL_Q_LOSS_SEI] = aging_states_df[COL_Q_LOSS_SEI]
            combined_df_append[COL_Q_LOSS_CYC] = aging_states_df[COL_Q_LOSS_CYC]
            combined_df_append[COL_Q_LOSS_LOW] = aging_states_df[COL_Q_LOSS_LOW]
            combined_df_append[COL_Q_LOSS_PLA] = aging_states_df[COL_Q_LOSS_PLA]
            combined_df_append[COL_Q_CHG_TOTAL] = aging_states_df[COL_Q_CHG_TOTAL]
            combined_df_append[COL_Q_DISCHG_TOTAL] = aging_states_df[COL_Q_DISCHG_TOTAL]
            combined_df_append[COL_E_CHG_TOTAL] = aging_states_df[COL_E_CHG_TOTAL]
            combined_df_append[COL_E_DISCHG_TOTAL] = aging_states_df[COL_E_DISCHG_TOTAL]
            combined_df = pd.concat([combined_df, combined_df_append], axis=0)

    # generate, save, and show plots
    fig_list = generate_result_figures(cap_remaining_arr, aging_states_arr)

    # save combined csv
    run_timestring = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_filename_csv = EXPORT_FILENAME + run_timestring + ".csv"
    combined_df[COL_AGE_TYPE] = combined_df[COL_AGE_TYPE].astype(int)
    combined_df[COL_TEMPERATURE] = combined_df[COL_TEMPERATURE].astype(int)
    combined_df[COL_SOC_MIN] = combined_df[COL_SOC_MIN].astype(int)
    combined_df[COL_SOC_MAX] = combined_df[COL_SOC_MAX].astype(int)
    combined_df[COL_CHG_CURRENT] = combined_df[COL_CHG_CURRENT].astype(int)  # optional
    combined_df[COL_DISCHG_CURRENT] = combined_df[COL_DISCHG_CURRENT].astype(int)  # optional
    combined_df[COL_PROFILE] = combined_df[COL_PROFILE].astype(int)
    combined_df.to_csv(EXPORT_PATH + export_filename_csv, index=True, index_label="timestamp",
                       sep=";", float_format="%.4f", na_rep="nan")

    print("debug here")


# initialize cell
def init_experiment_cell():
    cap_aged, aging_states, temp_cell, soc = bat.init(storage_time_days=CELL_STORAGE_TIME_DAYS,
                                                      storage_soc=CELL_STORAGE_SOC,
                                                      storage_temperature=CELL_STORAGE_TEMPERATURE)
    t_start = EXPERIMENT_START_TIMESTAMP

    # wait 2 hours at room temperature
    _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
        t_start, T_RESOLUTION_REST, (2 * 3600), TEMP_RT, None, None, None, None, None,
        cap_aged, aging_states, temp_cell, soc)

    cap_aged_df = pd.Series(dtype=np.float64)
    aging_states_df = pd.DataFrame(columns=[COL_Q_LOSS_SEI, COL_Q_LOSS_CYC, COL_Q_LOSS_LOW, COL_Q_LOSS_PLA,
                                            COL_Q_CHG_TOTAL, COL_Q_DISCHG_TOTAL, COL_E_CHG_TOTAL, COL_E_DISCHG_TOTAL],
                                   dtype=np.float64)

    return cap_aged_df, aging_states_df, cap_aged, aging_states, temp_cell, soc, t_start


# run experiment for one calendar aging cell
def run_calendar_aging(age_temp, age_v):
    cap_aged_df, aging_states_df, cap_aged, aging_states, temp_cell, soc, t_start = init_experiment_cell()

    # initial check-up
    t_next_cu = t_start + FIRST_CHECKUP_INTERVAL_S
    _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_checkup(
        t_start, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, age_v, I_CHG_CAL, I_DISCHG_CAL, I_CHG_CUTOFF_CAL,
        I_DISCHG_CUTOFF_CAL, age_temp, None, None, None, None, None, cap_aged, aging_states, temp_cell, soc)
    cap_aged_df[t_start] = cap_aged
    aging_states_df.loc[t_start, :] = aging_states

    for i_cu in range(2, N_CHECKUPS_MAX + 1):
        # calendar aging
        _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
            t_start, T_RESOLUTION_REST, (t_next_cu - t_start), age_temp, None, None, None, None, None,
            cap_aged, aging_states, temp_cell, soc)

        # check-up
        t_next_cu = t_start + NEXT_CHECKUP_INTERVAL_S
        _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_checkup(
            t_start, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, age_v, I_CHG_CAL, I_DISCHG_CAL, I_CHG_CUTOFF_CAL,
            I_DISCHG_CUTOFF_CAL, age_temp, None, None, None, None, None, cap_aged, aging_states, temp_cell, soc)
        cap_aged_df[t_start] = cap_aged
        aging_states_df.loc[t_start, :] = aging_states

        if cap_aged < C_REMAINING_CU_END:
            break

    return cap_aged_df, aging_states_df


# run experiment for one cyclic aging cell
def run_cyclic_aging(age_temp, age_v_range, age_c_rate):
    cap_aged_df, aging_states_df, cap_aged, aging_states, temp_cell, soc, t_start = init_experiment_cell()

    # initial check-up
    t_next_cu = t_start + FIRST_CHECKUP_INTERVAL_S
    _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_checkup(
        t_start, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, age_v_range[0], age_c_rate[1], age_c_rate[0], I_CHG_CUTOFF_CYC,
        I_DISCHG_CUTOFF_CYC, age_temp, None, None, None, None, None, cap_aged, aging_states, temp_cell, soc)
    cap_aged_df[t_start] = cap_aged
    aging_states_df.loc[t_start, :] = aging_states

    for i_cu in range(2, N_CHECKUPS_MAX + 1):
        # cyclic aging
        _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_cycles(
            None, t_start, t_next_cu, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, age_v_range[1], age_v_range[0],
            age_c_rate[1], age_c_rate[0], I_CHG_CUTOFF_CYC, I_DISCHG_CUTOFF_CYC, CYCLING_PAUSE, True, age_temp,
            None, None, None, None, None, cap_aged, aging_states, temp_cell, soc)

        if cap_aged < C_REMAINING_CU_END:
            break

        # check-up
        t_next_cu = t_start + NEXT_CHECKUP_INTERVAL_S
        _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_checkup(
            t_start, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, age_v_range[0], age_c_rate[1], age_c_rate[0],
            I_CHG_CUTOFF_CYC, I_DISCHG_CUTOFF_CYC, age_temp,
            None, None, None, None, None, cap_aged, aging_states, temp_cell, soc)
        cap_aged_df[t_start] = cap_aged
        aging_states_df.loc[t_start, :] = aging_states

        if cap_aged < C_REMAINING_CU_END:
            break

    return cap_aged_df, aging_states_df


# run experiment for one cell aging with a (driving) power profile
def run_profile_aging(age_temp, age_v_range, age_profile, age_c_rate):
    cap_aged_df, aging_states_df, cap_aged, aging_states, temp_cell, soc, t_start = init_experiment_cell()

    # initial check-up
    t_next_cu = t_start + FIRST_CHECKUP_INTERVAL_S
    _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_checkup(
        t_start, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, age_v_range[0], age_c_rate[1], age_c_rate[0], I_CHG_CUTOFF_CYC,
        I_DISCHG_CUTOFF_CYC, age_temp, None, None, None, None, None, cap_aged, aging_states, temp_cell, soc)
    cap_aged_df[t_start] = cap_aged
    aging_states_df.loc[t_start, :] = aging_states

    for i_cu in range(2, N_CHECKUPS_MAX + 1):
        # profile aging
        _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_profile_cycles(
            None, t_start, t_next_cu,
            T_RESOLUTION_ACTIVE, T_RESOLUTION_PROFILE, T_RESOLUTION_REST, age_profile, age_v_range[1], age_v_range[0],
            age_c_rate[1], I_CHG_CUTOFF_CYC, CYCLING_PAUSE, True, age_temp,
            None, None, None, None, None, cap_aged, aging_states, temp_cell, soc)

        # check-up
        t_next_cu = t_start + NEXT_CHECKUP_INTERVAL_S
        _, _, _, _, _, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_checkup(
            t_start, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, age_v_range[0], age_c_rate[1], age_c_rate[0],
            I_CHG_CUTOFF_CYC, I_DISCHG_CUTOFF_CYC, age_temp,
            None, None, None, None, None, cap_aged, aging_states, temp_cell, soc)
        cap_aged_df[t_start] = cap_aged
        aging_states_df.loc[t_start, :] = aging_states

        if cap_aged < C_REMAINING_CU_END:
            break

    return cap_aged_df, aging_states_df


# generate all empty figure templates
def generate_result_figures(cap_remaining_arr, aging_states_arr):  # FIXME: aging_states_arr not implemented yet
    # generate empty figures
    fig_list = {age_type.CALENDAR: None, age_type.CYCLIC: None, age_type.PROFILE: None}
    if age_type.CALENDAR in cap_remaining_arr:
        # build up calendar aging figure:
        # columns: temperatures from lowest to highest
        # rows: SoC from lowest to highest
        a_type = age_type.CALENDAR
        subplot_titles = ['---' for _ in range(0, N_TEMP * N_SOC)]
        for i_temp in range(0, N_TEMP):
            t_text = f"%u°C" % TEMP_OT_ARR[i_temp]
            for i_soc in range(0, N_SOC):
                soc_text = "%u %%" % SOC_CAL_AGE_ARR[i_soc]
                i_title = (i_soc * N_TEMP) + i_temp
                subplot_titles[i_title] = t_text + ", " + soc_text
        title_text = PLOT_TITLE_RE % AGE_TYPE_TITLES.get(a_type)
        fig_list[a_type] = generate_base_figure(N_SOC, N_TEMP, title_text, subplot_titles, a_type)

    if age_type.CYCLIC in cap_remaining_arr:
        # build up calendar aging figure:
        # columns: temperatures from lowest to highest
        # rows (outer loop): C-rate (unsorted - in my experiment: from lowest to highest)
        # rows (inner loop): SoC range (unsorted - in my experiment: from widest to narrowest)
        a_type = age_type.CYCLIC
        n_rows = N_C_RATES * N_SOC_RANGES
        subplot_titles = ['---' for _ in range(0, N_TEMP * n_rows)]
        for i_temp in range(0, N_TEMP):
            t_text = f"%u°C" % TEMP_OT_ARR[i_temp]
            for i_cr in range(0, N_C_RATES):
                c_rate_text = C_RATE_TEXT_ARR[i_cr]
                for i_sr in range(0, N_SOC_RANGES):
                    i_row = (i_cr * N_SOC_RANGES) + i_sr
                    soc_range_text = SOC_CYC_MIN_MAX_TEXT_ARR[i_sr]
                    i_title = (i_row * N_TEMP) + i_temp
                    subplot_titles[i_title] = t_text + ", " + soc_range_text + ",<br>" + c_rate_text
        title_text = PLOT_TITLE_RE % AGE_TYPE_TITLES.get(a_type)
        fig_list[a_type] = generate_base_figure(n_rows, N_TEMP, title_text, subplot_titles, a_type)

    if age_type.PROFILE in cap_remaining_arr:
        # build up profile aging figure:
        # columns: temperatures from lowest to highest
        # rows: profile number (rising)
        a_type = age_type.PROFILE

        subplot_titles = ['---' for _ in range(0, N_TEMP * N_PROFILES)]
        for i_temp in range(0, N_TEMP):
            t_text = f"%u°C" % TEMP_OT_ARR[i_temp]
            for i_prf in range(0, N_PROFILES):
                profile_text = PROFILE_TEXT_ARR[i_prf]
                i_title = (i_prf * N_TEMP) + i_temp
                subplot_titles[i_title] = t_text + ",<br>" + profile_text
        title_text = PLOT_TITLE_RE % AGE_TYPE_TITLES.get(a_type)
        fig_list[a_type] = generate_base_figure(N_PROFILES, N_TEMP, title_text, subplot_titles, a_type)

    # fill figures
    if age_type.CALENDAR in cap_remaining_arr:
        # columns: temperatures from lowest to highest
        # rows: SoC from lowest to highest
        a_type = age_type.CALENDAR
        this_fig: go.Figure = fig_list[a_type]
        for i_temp in range(0, N_TEMP):
            color = TEMPERATURE_COLORS.get(TEMP_OT_ARR[i_temp])
            for i_soc in range(0, N_SOC):
                add_result_trace(this_fig, i_soc, i_temp, cap_remaining_arr[age_type.CALENDAR][i_temp][i_soc].index,
                                 cap_remaining_arr[age_type.CALENDAR][i_temp][i_soc].values, color)

    if age_type.CYCLIC in cap_remaining_arr:
        # build up calendar aging figure:
        # columns: temperatures from lowest to highest
        # rows (outer loop): C-rate (unsorted - in my experiment: from lowest to highest)
        # rows (inner loop): SoC range (unsorted - in my experiment: from widest to narrowest)
        a_type = age_type.CYCLIC
        this_fig: go.Figure = fig_list[a_type]
        for i_temp in range(0, N_TEMP):
            color = TEMPERATURE_COLORS.get(TEMP_OT_ARR[i_temp])
            for i_cr in range(0, N_C_RATES):
                for i_sr in range(0, N_SOC_RANGES):
                    i_row = (i_cr * N_SOC_RANGES) + i_sr
                    add_result_trace(this_fig, i_row, i_temp,
                                     cap_remaining_arr[age_type.CYCLIC][i_temp][i_sr][i_cr].index,
                                     cap_remaining_arr[age_type.CYCLIC][i_temp][i_sr][i_cr].values, color)

    if age_type.PROFILE in cap_remaining_arr:
        # build up profile aging figure:
        # columns: temperatures from lowest to highest
        # rows: profile number (rising)
        a_type = age_type.PROFILE
        this_fig: go.Figure = fig_list[a_type]
        for i_temp in range(0, N_TEMP):
            color = TEMPERATURE_COLORS.get(TEMP_OT_ARR[i_temp])
            for i_prf in range(0, N_PROFILES):
                add_result_trace(this_fig, i_prf, i_temp, cap_remaining_arr[age_type.PROFILE][i_temp][i_prf].index,
                                 cap_remaining_arr[age_type.PROFILE][i_temp][i_prf].values, color)

    for a_type in age_type:
        if a_type not in cap_remaining_arr:
            continue
        if a_type not in fig_list:
            continue
        this_fig: go.Figure = fig_list.get(a_type)
        filename = EXPORT_FILENAME + AGE_TYPE_FILENAMES.get(a_type)
        result_plot.export_figure(this_fig, EXPORT_HTML, EXPORT_IMAGE, EXPORT_PATH, filename, OPEN_IN_BROWSER)
        # this_fig.show(validate=False)

    return fig_list


# add trace/line to figure
def add_result_trace(fig, i_row, i_col, x, y, color):
    x_plot = pd.to_datetime(x, unit="s", origin='unix')
    data_range = range(0, x_plot.shape[0])
    text_data = pd.Series("", index=data_range)
    for i in data_range:
        text_data.loc[i] = PLOT_TEXT_CU % (i + 1, x_plot[i].strftime('%Y-%m-%d'))

    this_marker_style = MARKER_STYLE.copy()
    this_marker_style["color"] = color

    fig.add_trace(
        go.Scatter(x=x_plot, y=y, showlegend=False, mode='markers+lines', marker=this_marker_style,
                   text=text_data, hovertemplate=PLOT_HOVER_TEMPLATE_CU,
                   line=dict(color=color, width=TRACE_LINE_WIDTH)),
        row=(i_row + 1), col=(i_col + 1))


# generate (one) empty figure template
def generate_base_figure(n_rows, n_cols, plot_title, subplot_titles, a_type):
    subplot_h_spacing = (SUBPLOT_H_SPACING_REL / n_cols)
    subplot_v_spacing = (SUBPLOT_V_SPACING_REL / n_rows)
    plot_height = HEIGHT_PER_ROW * n_rows
    plot_title_y_pos = 1.0 - PLOT_TITLE_Y_POS_REL / plot_height

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes="all",
                        horizontal_spacing=subplot_h_spacing, vertical_spacing=subplot_v_spacing,
                        subplot_titles=subplot_titles)

    fig.update_xaxes(ticks="outside",
                     tickcolor=MAJOR_GRID_COLOR,
                     ticklen=10,
                     showticklabels=True,
                     minor=dict(griddash='dot',
                                gridcolor=MINOR_GRID_COLOR,
                                ticklen=4)
                     )
    fig.update_yaxes(ticks="outside",
                     tickcolor=MAJOR_GRID_COLOR,
                     ticklen=10,
                     showticklabels=True,
                     minor=dict(griddash='dot',
                                gridcolor=MINOR_GRID_COLOR,
                                ticklen=4),
                     title_text=Y_AXIS_TITLE,
                     # title={'text': Y_AXIS_TITLE, 'font': dict(size=AXIS_FONT_SIZE)},
                     )

    if a_type == age_type.CALENDAR:
        if USE_MANUAL_PLOT_LIMITS_CAL:
            fig.update_yaxes(range=MANUAL_PLOT_LIMITS_Y_CAL)
    elif a_type == age_type.CYCLIC:
        if USE_MANUAL_PLOT_LIMITS_CYC:
            fig.update_yaxes(range=MANUAL_PLOT_LIMITS_Y_CYC)
    elif a_type == age_type.PROFILE:
        if USE_MANUAL_PLOT_LIMITS_PRF:
            fig.update_yaxes(range=MANUAL_PLOT_LIMITS_Y_PRF)

    for i_col in range(0, n_cols):
        fig.update_xaxes(title_text=X_AXIS_TITLE, row=n_rows, col=(i_col + 1))

    fig.update_layout(title={'text': plot_title,
                             'font': dict(size=TITLE_FONT_SIZE, color='black'),
                             'y': plot_title_y_pos,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template=FIGURE_TEMPLATE,
                      autosize=True,
                      height=plot_height,
                      width=PLOT_WIDTH,
                      legend=dict(x=0, y=0),
                      margin=dict(l=SUBPLOT_LR_MARGIN, r=SUBPLOT_LR_MARGIN,
                                  t=SUBPLOT_TOP_MARGIN, b=SUBPLOT_BOT_MARGIN,
                                  pad=SUBPLOT_PADDING)
                      )
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    return fig


if __name__ == "__main__":
    run()
