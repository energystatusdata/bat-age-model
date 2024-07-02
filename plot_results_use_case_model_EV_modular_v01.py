import numpy as np
import pandas as pd
import datetime
import os
import re
import pytz
import plotly.graph_objects as go

import logger
import result_plot

# ToDo:
#   - consider splitting up into multiple figures for diss
#     (maybe in addition to a plot containing all lines -> 2-3 subplots next to each other)

# --- import ---
# IMPORT_PATH = "D:\\bat\\analysis\\use_case_models\\images\\"
# IMPORT_PATH = "G:\\_diss_bat_use_case_sim_results\\"
IMPORT_PATH = "E:\\_diss_bat_use_case_sim_results\\"
IMPORT_FILENAME_BASE = r"use_case_model_007_modular_driving_sc(\d{3})_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2}).csv"
IMPORT_FILENAME_B_BASE = r"use_case_model_007_modular_driving_sc(\d{3})c_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2}).csv"
IMPORT_FILENAME_B_WRITE_BASE = f"use_case_model_007_modular_driving_sc%03uc_%04u-%02u-%02u_%02u-%02u.csv"
# use_case_model_007_modular_driving_sc020_2024-05-24_18-04.csv
CSV_SEP = ";"
COL_TIME = "timestamp"
COL_TIME_TEXT = "timestamp_text"
COL_CAP_REMAINING = "Remaining capacity [Ah]"
# COL_Q_LOSS_REL_SEI = "q_loss_sei_total"
# COL_Q_LOSS_REL_CYCLIC = "q_loss_cyclic_total"
# COL_Q_LOSS_REL_CYCLIC_LOW = "q_loss_cyclic_low_total"
# COL_Q_LOSS_REL_PLATING = "q_loss_plating_total"
IMPORT_COLUMNS = [COL_TIME, COL_CAP_REMAINING]
# IMPORT_COLUMNS_B = [COL_TIME, COL_CAP_REMAINING, COL_TIME_TEXT]
IMPORT_COLUMNS_B = {COL_TIME: float, COL_CAP_REMAINING: float, COL_TIME_TEXT: str}
# IMPORT_DOWNSAMPLE_N = 2880

# --- export ---
EXPORT_PATH = "D:\\bat\\analysis\\use_case_models\\images\\result_plots\\"
OPEN_IN_BROWSER = True
EXPORT_HTML = True
EXPORT_IMAGE = "svg"  # None / "pdf" / "png" / "svg"? ...
EXPORT_FILENAME_BASE = "use_case_model_007_modular_driving_results_%u"

# --- logging ---
logging_filename = "D:\\bat\\analysis\\use_case_models\\log\\use_case_model_007_plot.txt"
logging = logger.bat_logger(logging_filename)

# --- file dict ---
fd_filename = "filename"  # filename of the result file (compressed or uncompressed)
fd_filename_compressed = "filename_compressed"
fd_date = "date"  # date of the result file
fd_compressed = "compressed"  # True if this is a compressed version of a result file (only remaining capacity)

# --- plot settings ---
TIMEZONE_DEFAULT = 'Europe/Berlin'
TZ_INFO_DEFAULT = pytz.timezone(TIMEZONE_DEFAULT)

INSERT_LEGEND_TITLE = True
SHOW_SCENARIO_AT_LINE_END = False
PRESERVE_RATIO = True  # if more than one column is used, the height will be reduced to approximately preserve the ratio
SHOW_ALL_KNOWN_SCENARIOS = True  # if True, show all entries of SC_LINE_SETTINGS in legend
# One array per figure, one array per column, e.g. "[[[1, 2], [3, 4]]]". -> 1 figure with 2 columns: [1, 2] and [3, 4]
# Set to None to plot all, e.g., "[[[1, 2], [3 ,4]], [None]]". 1st figure: as above, 2nd figure: 1 column with all plots

# # for diss - overview of all scenarios
# PLOT_SCENARIOS = [[None]]  # plot all
# PLOT_SCENARIOS_BG = [[[]]]  # plot none in the background
# PLOT_TITLES = [["Overview of all simulated scenarios"]]
# # PLOT_SCENARIOS_BG = None  # plot none in the background (both is fine)

# for diss - individual figures
PLOT_SCENARIOS = [[[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11, 19]],
                  [[12, 13, 14, 23, 24, 25]],
                  [[15, 16, 17, 18, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33]],
                  # [None]
                  ]
PLOT_SCENARIOS_BG = [[[], [0, 1, 2, 6, 7, 8]],
                     [[0, 19]],
                     [[0, 19]],
                     # [[]]
                     ]
PLOT_TITLES = [["a) Conventional and scheduled charging", "b) Conditional conventional and scheduled charging"],
               ["Smart charging (V1G) strategies"],
               ["Bidirectional charging (V2G) strategies"],
               # [""]
               ]

PLOT_TEXT_FONTSIZE = 18
PLOT_TEXT_X = "x"
PLOT_TEXT_Y = "y"
PLOT_TEXT_STRING = "str"
PLOT_TEXT_COLOR = "color"
PLOT_TEXT_ARR_DEFAULT =\
    [{PLOT_TEXT_X: datetime.datetime(2035, 1, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 103.75,
      PLOT_TEXT_STRING: "<b>initial capacity</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_GRAY},
     {PLOT_TEXT_X: datetime.datetime(2035, 1, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 99.25,
      PLOT_TEXT_STRING: "<b>100% SoH</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_GREEN_DARK},
     {PLOT_TEXT_X: datetime.datetime(2026, 7, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 79.25,
      PLOT_TEXT_STRING: "<b>80% SoH</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_YELLOW_DARK},
     {PLOT_TEXT_X: datetime.datetime(2026, 7, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 70.75,
      PLOT_TEXT_STRING: "<b>70% SoH</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_RED_DARK},
    ]
PLOT_TEXT_ARR_SMALL =\
    [{PLOT_TEXT_X: datetime.datetime(2035, 1, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 104.2,
      PLOT_TEXT_STRING: "<b>initial capacity</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_GRAY},
     {PLOT_TEXT_X: datetime.datetime(2035, 1, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 99,
      PLOT_TEXT_STRING: "<b>100% SoH</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_GREEN_DARK},
     {PLOT_TEXT_X: datetime.datetime(2029, 7, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 79,
      PLOT_TEXT_STRING: "<b>80% SoH</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_YELLOW_DARK},
     {PLOT_TEXT_X: datetime.datetime(2029, 7, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 71,
      PLOT_TEXT_STRING: "<b>70% SoH</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_RED_DARK},
    ]
PLOT_TEXT_ARR_SMALL_EXTENDED = PLOT_TEXT_ARR_SMALL.copy()
PLOT_TEXT_ARR_SMALL_EXTENDED.extend(
    [{PLOT_TEXT_X: datetime.datetime(2031, 4, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 75.9,  # 75,
      PLOT_TEXT_STRING: "<b>common EOL</b>",  # "<b>common<br>EOL<br>criteria</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_GRAY},
     {PLOT_TEXT_X: datetime.datetime(2028, 9, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 74.1,  # 75,
      PLOT_TEXT_STRING: "<b>criteria</b>",  # "<b>common<br>EOL<br>criteria</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_GRAY},
     {PLOT_TEXT_X: datetime.datetime(2039, 1, 1, tzinfo=TZ_INFO_DEFAULT),
      PLOT_TEXT_Y: 92.5,
      PLOT_TEXT_STRING: "<b>capacity<br>fade</b>",
      PLOT_TEXT_COLOR: result_plot.COLOR_GRAY},
    ])
PLOT_TEXTS = [[PLOT_TEXT_ARR_SMALL_EXTENDED,
               PLOT_TEXT_ARR_SMALL
               ],
              [PLOT_TEXT_ARR_DEFAULT],
              [PLOT_TEXT_ARR_DEFAULT],
              # [PLOT_TEXT_ARR_DEFAULT]
              ]

# FIXME: arrows


# PLOT_SCENARIOS = [[[32, 33]]]  # for debugging
# PLOT_SCENARIOS_BG = [[[]]]  # for debugging
# PLOT_TITLES = [[""]]

# PLOT_SCENARIOS = [[[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11, 19]], [None]]  # for diss - part 1
# PLOT_SCENARIOS_BG = [[[], [0, 1, 2, 6, 7, 8]], [[]]]
# PLOT_TITLES = [["a) Conventional and scheduled charging", "b) Conditional conventional and scheduled charging"], [""]]

# # for diss - part 2: smart unidirectional charging (V1G) - AND - bidirectional charging (V2G)
# PLOT_SCENARIOS = [[[12, 13, 14]], [[15, 16, 17, 18, 32, 33]]]
# PLOT_SCENARIOS_BG = [[[0, 19]], [[0, 19]]]
# PLOT_TITLES = [[""], [""]]

# for diss - part 2: smart unidirectional charging (V1G) - AND - bidirectional charging (V2G)
# PLOT_SCENARIOS = [[[12, 13, 14, 23, 24, 25]], [[15, 16, 17, 18, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33]]]
# PLOT_SCENARIOS_BG = [[[0, 19]], [[0, 19]]]
# PLOT_TITLES = [[""], [""]]
# PLOT_SCENARIOS = [[[12, 13, 14, 23, 24, 25], [15, 16, 17, 18, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33]]]
# PLOT_SCENARIOS_BG = [[[0, 19], [0, 19]]]
# PLOT_TITLES = [["a) smart unidirectional charging (V1G)"], ["b) bidirectional charging (V2G)"]]

# PLOT_SCENARIOS = [[[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11, 19]]]  # for diss
# PLOT_SCENARIOS_BG = [[[], [0, 1, 2, 6, 7, 8]]]
# PLOT_TITLES = [["a) conventional and scheduled charging", "b) conditional conventional and scheduled charging"]]

# PLOT_SCENARIOS = [[[0, 2], [3, 5]], [[1, 4]]]  # for debugging
# PLOT_SCENARIOS_BG = [[[], [0, 2]], [[]]]
# PLOT_TITLES = [["a) conventional and scheduled charging", "b) conditional conventional and scheduled charging"]]

# Title: aging type
# PLOT_TITLE = "<b>Remaining usable capacity [%]</b>"

X_AXIS_TITLE = "Year"
Y_AXIS_TITLE = "Remaining usable capacity (SoH) [%]"
# LINE_NAME_BASE = "Scenario %u"
LINE_NAME_BASE = "%u: %s"
LINE_NAME_BASE_SINGLE_DIGIT = "%u:   %s"

X_TICK_YEARS_MAJOR = 2
X_TICK_MONTHS_MINOR = 6

TITLE_FONT_SIZE = 24  # 22
SUBPLOT_TITLE_FONT_SIZE = 19
AXIS_FONT_SIZE = 19  # 20
AXIS_TICK_FONT_SIZE = 17  # 18
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 20  # 16

CAP_NOMINAL = 3.0
if SHOW_SCENARIO_AT_LINE_END:
    X_LIM_ARR = [[datetime.datetime(2025, 1, 1, tzinfo=TZ_INFO_DEFAULT),
                  datetime.datetime(2046, 12, 31, tzinfo=TZ_INFO_DEFAULT)]]
else:
    X_LIM_ARR = [[datetime.datetime(2025, 1, 1, tzinfo=TZ_INFO_DEFAULT),
                  datetime.datetime(2045, 1, 1, tzinfo=TZ_INFO_DEFAULT)]]
Y_LIM_ARR = [[69, 104.95]]

# --- result_plot overwrite settings ---
result_plot.SUBPLOT_TOP_MARGIN = 25
result_plot.SUBPLOT_LR_MARGIN = 0  # 60  # 50  # 40
result_plot.PLOT_WIDTH = 1150  # original value: 1350
result_plot.HEIGHT_PER_ROW = 550  # 600  # 540  # in px - for individual V1G/V2G overview
# result_plot.HEIGHT_PER_ROW = 750  # 600  # 540  # in px - for "all scenarios"
result_plot.SUBPLOT_H_SPACING_REL = 0.74  # 0.67  # in 100%
result_plot.PLOT_HOVER_TEMPLATE = "<b>%{text}</b><br>%{y:.2f}%<br><extra></extra>"
result_plot.pio.templates['custom_theme']['layout']['title']['font']['size'] = TITLE_FONT_SIZE
result_plot.pio.templates['custom_theme']['layout']['xaxis']['title']['font']['size'] = AXIS_FONT_SIZE
result_plot.pio.templates['custom_theme']['layout']['yaxis']['title']['font']['size'] = AXIS_FONT_SIZE
result_plot.pio.templates['custom_theme']['layout']['xaxis']['tickfont']['size'] = AXIS_TICK_FONT_SIZE
result_plot.pio.templates['custom_theme']['layout']['yaxis']['tickfont']['size'] = AXIS_TICK_FONT_SIZE
result_plot.pio.templates['custom_theme']['layout']['annotationdefaults']['font']['size'] = SUBPLOT_TITLE_FONT_SIZE
result_plot.pio.templates['custom_theme']['layout']['legend']['font']['size'] = LEGEND_FONT_SIZE

# figure additional element settings
Y_LIMITS = [103, 100, 80, 70]
Y_LIMIT_COLORS = [result_plot.COLOR_GRAY, result_plot.COLOR_GREEN_DARK, result_plot.COLOR_YELLOW_DARK,
                  result_plot.COLOR_RED_DARK]
LIMIT_LINE_WIDTH = 2.5
LIMIT_ALPHA = 0.65

# background lines
BG_ALPHA = 0.37
BG_COLOR_MIX = np.array([0, 0, 0, 1.0])
BG_COLOR_MIX_FACTOR = 0.5
BG_LINE_WIDTH = 0.75

# color tools
re_pat_color_rgb = re.compile(r"rgb\((\d+),\s?(\d+),\s?(\d+)\)")
re_pat_color_rgba = re.compile(r"rgba\((\d+),\s?(\d+),\s?(\d+),\s?(\d*\.*\d+)\)")


NORMAL_LINE_WIDTH = 2.0
THICK_LINE_WIDTH = 3.25

# --- scenario plot definitions ---
SC_LINE_SETTINGS = {
    0: {'color': result_plot.COLOR_RED_DARK, 'dash': 'solid', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'EARLY to 100%'},
    1: {'color': result_plot.COLOR_RED_DARK, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'EARLY to 90%'},
    2: {'color': result_plot.COLOR_RED_DARK, 'dash': 'dash', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'EARLY to 80%'},
    3: {'color': result_plot.COLOR_RED_LIGHT, 'dash': 'solid', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'EARLY_low to 100% if < 40%'},
    4: {'color': result_plot.COLOR_RED_LIGHT, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'EARLY_low to 90% if < 40%'},
    5: {'color': result_plot.COLOR_RED_LIGHT, 'dash': 'dash', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'EARLY_low to 80% if < 40%'},
    6: {'color': result_plot.COLOR_ORANGE_DARK, 'dash': 'solid', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'LATE to 100%'},
    7: {'color': result_plot.COLOR_ORANGE_DARK, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'LATE to 90%'},
    8: {'color': result_plot.COLOR_ORANGE_DARK, 'dash': 'dash', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'LATE to 80%'},
    9: {'color': result_plot.COLOR_ORANGE_LIGHT, 'dash': 'solid', 'line_width': NORMAL_LINE_WIDTH,
        'shortname': 'LATE_low to 100% if < 40%'},
    10: {'color': result_plot.COLOR_ORANGE_LIGHT, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'LATE_low to 90% if < 40%'},
    11: {'color': result_plot.COLOR_ORANGE_LIGHT, 'dash': 'dash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'LATE_low to 80% if < 40%'},
    12: {'color': result_plot.COLOR_GREEN_INTENSE_LIGHT, 'dash': 'longdash', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V1G_emission 25-90% (H/W)'},
    13: {'color': result_plot.COLOR_PURPLE_INTENSE_LIGHT, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V1G_cost 25-90% (H/W)'},
    14: {'color': result_plot.COLOR_BLUE_LIGHT, 'dash': 'longdash', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V1G_renewable 25-90% (H/W)'},
    15: {'color': result_plot.COLOR_GREEN_INTENSE_DARK, 'dash': 'longdash', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V2G_emission 25-90% (H/W)'},
    16: {'color': result_plot.COLOR_PURPLE_INTENSE_DARK, 'dash': 'longdash', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V2G_cost 25-90% (H/W)'},
    17: {'color': result_plot.COLOR_BLUE_DARK, 'dash': 'longdash', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V2G_renewable 25-90% (H/W)'},
    18: {'color': result_plot.COLOR_PINK_INTENSE, 'dash': 'dash', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V2G_freq 40-80% (H/W)'},
    19: {'color': result_plot.COLOR_ORANGE_LIGHT, 'dash': 'dot', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'LATE_low to 60% if < 40%'},
    20: {'color': result_plot.COLOR_BLUE_DARK, 'dash': 'dashdot', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V2G_renewable 25-75% (H/W)'},
    21: {'color': result_plot.COLOR_BLUE_DARK, 'dash': 'dashdot', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V2G_renewable 25-75% (H/W)'},
    22: {'color': result_plot.COLOR_BLUE_DARK, 'dash': 'dot', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V2G_renewable 25-60% (H/W)'},
    23: {'color': result_plot.COLOR_GREEN_INTENSE_LIGHT, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V1G_emission 25-90%'},
    24: {'color': result_plot.COLOR_PURPLE_INTENSE_LIGHT, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V1G_cost 25-90%'},
    25: {'color': result_plot.COLOR_BLUE_LIGHT, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V1G_renewable 25-90%'},
    26: {'color': result_plot.COLOR_GREEN_INTENSE_DARK, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V2G_emission 25-90%'},
    27: {'color': result_plot.COLOR_PURPLE_INTENSE_DARK, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V2G_cost 25-90%'},
    28: {'color': result_plot.COLOR_BLUE_DARK, 'dash': 'longdash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V2G_renewable 25-90%'},
    29: {'color': result_plot.COLOR_BLUE_DARK, 'dash': 'dashdot', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V2G_renewable 25-75%'},
    30: {'color': result_plot.COLOR_BLUE_DARK, 'dash': 'dashdot', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V2G_renewable 25-75%'},
    31: {'color': result_plot.COLOR_BLUE_DARK, 'dash': 'dot', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V2G_renewable 25-60%'},
    32: {'color': result_plot.COLOR_YELLOW_DARK, 'dash': 'dash', 'line_width': THICK_LINE_WIDTH,
         'shortname': 'V2H_PV 40-80% (H/W)'},
    33: {'color': result_plot.COLOR_YELLOW_DARK, 'dash': 'dash', 'line_width': NORMAL_LINE_WIDTH,
         'shortname': 'V2H_PV 40-80% (PT)'},  # (part-time)

}


def run():
    start_timestamp = datetime.datetime.now()
    logging.log.info(os.path.basename(__file__))

    # generate dict of files
    sc_result_file_dict = get_file_dict()

    # read column of interest from files
    remaining_cap_dict = read_remaining_cap_data(sc_result_file_dict)

    for i_fig in range(len(PLOT_SCENARIOS)):
        # plot
        n_cols = len(PLOT_SCENARIOS[i_fig])
        if PRESERVE_RATIO:
            abs_height = result_plot.HEIGHT_PER_ROW / n_cols
        else:
            abs_height = None
        fig: go.Figure = result_plot.generate_base_figure(1, n_cols, "", PLOT_TITLES[i_fig], [Y_AXIS_TITLE],
                                                          x_lim_arr=X_LIM_ARR, y_lim_arr=Y_LIM_ARR,
                                                          x_tick_side="bottom",
                                                          abs_height=abs_height, x_axis_title=X_AXIS_TITLE)
        fig = plot_enhance(fig, n_cols, i_fig)
        if (PLOT_SCENARIOS_BG is not None) and (len(PLOT_SCENARIOS_BG) > i_fig):
            plot_scenarios_background = PLOT_SCENARIOS_BG[i_fig]
        else:
            plot_scenarios_background = []
        fig = plot_remaining_cap_data(fig, remaining_cap_dict, PLOT_SCENARIOS[i_fig], plot_scenarios_background)

        # export / open
        logging.log.info("Exporting plot...")
        export_filename = EXPORT_FILENAME_BASE % i_fig
        result_plot.export_figure(fig, EXPORT_HTML, EXPORT_IMAGE, EXPORT_PATH, export_filename, OPEN_IN_BROWSER)

    stop_timestamp = datetime.datetime.now()
    logging.log.info("\nScript runtime: %s h:mm:ss.ms" % str(stop_timestamp - start_timestamp))

    # for debugging:
    # plt.plot(result_df.index, result_df.values, "r")
    # plt.grid(True)
    # plt.show(block=False)
    # plt.pause(.1)


def plot_remaining_cap_data(fig, remaining_cap_dict, plot_scenario_col_array, plot_scenario_bg_col_array):
    logging.log.info("Sorting scenarios...")
    sc_ids = list(remaining_cap_dict.keys())
    sc_soh_end = [0.0 for _ in range(len(sc_ids))]
    for i in range(len(sc_ids)):
        this_df = remaining_cap_dict[sc_ids[i]]
        if len(this_df) > 0:
            sc_soh_end[i] = this_df[COL_CAP_REMAINING].values[-1]
    sc_ids_sorted = [sc_id for _, sc_id in sorted(zip(sc_soh_end, sc_ids), reverse=True)]
    logging.log.info("Plotting scenarios...")

    # for each column
    n_cols = len(plot_scenario_col_array)
    for i_col in range(n_cols):
        # plot_sc_ids_bg = []
        # if (plot_scenario_bg_col_array is not None) and (len(plot_scenario_bg_col_array) > i_col):
        #     plot_sc_ids_bg = plot_scenario_bg_col_array[i_col]
        #
        # if plot_sc_ids_bg is None:
        #     plot_sc_ids_bg_sorted = []  # no background plots
        # else:
        #     plot_sc_ids_bg_sorted = []
        #     for sc_id in sc_ids_sorted:
        #         if sc_id in plot_sc_ids_bg:
        #             plot_sc_ids_bg_sorted.append(sc_id)
        #
        # plot_sc_ids = plot_scenario_col_array[i_col]
        # if plot_sc_ids is None:
        #     plot_sc_ids_sorted = sc_ids_sorted.copy()
        # else:
        #     plot_sc_ids_sorted = []
        #     for sc_id in sc_ids_sorted:
        #         if sc_id in plot_sc_ids:
        #             plot_sc_ids_sorted.append(sc_id)
        plot_sc_ids_bg = []
        if (plot_scenario_bg_col_array is not None) and (len(plot_scenario_bg_col_array) > i_col):
            plot_sc_ids_bg = plot_scenario_bg_col_array[i_col]
            if plot_sc_ids_bg is None:
                plot_sc_ids_bg = []  # don't plot anything in the background

        plot_sc_ids_fg = plot_scenario_col_array[i_col]
        if plot_sc_ids_fg is None:
            plot_sc_ids_sorted = sc_ids_sorted.copy()  # plot all available in the foreground
            plot_sc_ids_fg = plot_sc_ids_sorted
        else:
            plot_sc_ids_sorted = []
            for sc_id in sc_ids_sorted:
                if (sc_id in plot_sc_ids_fg) or (sc_id in plot_sc_ids_bg):
                    plot_sc_ids_sorted.append(sc_id)

        # create/rename legend:
        legend_name = f"legend%u" % (i_col + 2)
        # subplot_width = (result_plot.PLOT_WIDTH * (1.0 - result_plot.SUBPLOT_H_SPACING_REL * (n_cols - 2)) / n_cols)
        hsr = result_plot.SUBPLOT_H_SPACING_REL
        if (n_cols > 1) and (hsr > 0):
            # subplot_h_space_width = (result_plot.PLOT_WIDTH * result_plot.SUBPLOT_H_SPACING_REL) / (n_cols - 1)
            subplot_h_space_width = result_plot.PLOT_WIDTH / (n_cols * (1.0 - hsr) / hsr + 1)
        else:
            subplot_h_space_width = 0
        # subplot_width = (result_plot.PLOT_WIDTH - (n_cols - 1) * subplot_h_space_width) / n_cols
        subplot_width = (result_plot.PLOT_WIDTH - (n_cols - 1) * subplot_h_space_width) / n_cols
        if i_col < (n_cols - 1):
            dy = 0.115  # 0.09
        else:
            dy = 0.015
        legend_pos_x = (((subplot_width + subplot_h_space_width) * i_col  # account for previous columns
                        + subplot_width  # account for current column plot
                        + dy * result_plot.PLOT_WIDTH)  # additional margin -> spacing from plot
                        / result_plot.PLOT_WIDTH)  # position is relative
        legend_pos_y = 0
        if INSERT_LEGEND_TITLE and (plot_sc_ids_fg is not None):
            legend_title = "<b>Scenarios:</b><br>ID: strategy and SoC"
        else:
            legend_title = None
        fig.update_layout(
            {legend_name: dict(x=legend_pos_x, y=legend_pos_y, itemwidth=50, font=dict(size=LEGEND_FONT_SIZE),
                               title=legend_title)}
        )
        # fig.update_layout(
        #     {legend_name: dict(x=(i_col + 1)/n_cols, y=0, xanchor="right", itemwidth=50)}
        # )

        # for sc_id, result_df in remaining_cap_dict.items():
        for sc_id in plot_sc_ids_sorted:
            zorder = None
            if sc_id in SC_LINE_SETTINGS:
                this_line_setting = SC_LINE_SETTINGS[sc_id]
                this_color = this_line_setting["color"]
                dash = this_line_setting["dash"]
                line_width = this_line_setting["line_width"]
                # name = LINE_NAME_BASE % sc_id
                if sc_id < 10:
                    name = LINE_NAME_BASE_SINGLE_DIGIT % (sc_id, this_line_setting["shortname"])
                else:
                    name = LINE_NAME_BASE % (sc_id, this_line_setting["shortname"])
            else:
                logging.log.warning("Warning: scenario %03u not in SC_LINE_SETTINGS - using default line style" % sc_id)
                this_color = result_plot.COLOR_BLACK
                dash = "solid"
                line_width = 2
                # name = LINE_NAME_BASE % sc_id
                if sc_id < 10:
                    name = LINE_NAME_BASE_SINGLE_DIGIT % (sc_id, "???")
                else:
                    name = LINE_NAME_BASE % (sc_id, "???")

            if (sc_id not in plot_sc_ids_fg) and (sc_id in plot_sc_ids_bg):
                # put this into the background and gray out the line
                re_match = re_pat_color_rgba.fullmatch(this_color)
                rgba_arr = np.array([127, 127, 127, 1.0])  # error
                if re_match:
                    rgba_arr = np.array([int(re_match.group(1)), int(re_match.group(2)), int(re_match.group(3)),
                                         float(re_match.group(4))])  # r, g, b, a
                else:
                    re_match = re_pat_color_rgb.fullmatch(this_color)
                    if re_match:
                        rgba_arr[0] = int(re_match.group(1))
                        rgba_arr[1] = int(re_match.group(2))
                        rgba_arr[2] = int(re_match.group(3))

                rgba_arr = BG_COLOR_MIX + (1.0 - BG_COLOR_MIX_FACTOR) * rgba_arr
                rgba_arr[3] = BG_ALPHA
                this_color = f"rgba(%u,%u,%u,%f)" % (int(rgba_arr[0]), int(rgba_arr[1]), int(rgba_arr[2]),
                                                     float(rgba_arr[3]))
                zorder = -1
                line_width = BG_LINE_WIDTH
                # noinspection PyTypeChecker
                name = ("<span style='color:#%02X%02X%02X%02X'>%s</span>"
                        % (round(rgba_arr[0]), round(rgba_arr[1]), round(rgba_arr[2]), round(rgba_arr[3] * 255.0),
                           name))

            result_df = remaining_cap_dict[sc_id]
            if len(result_df) > 0:
                timestamp_data = result_df.index
                datetime_data = pd.to_datetime(timestamp_data.astype(np.int64), unit="s",
                                               utc=True).tz_convert(TIMEZONE_DEFAULT)
                sc_id_text = "Sc %03u: " % sc_id
                if COL_TIME_TEXT in result_df.columns:
                    text_data = sc_id_text + result_df[COL_TIME_TEXT]
                else:
                    text_data = sc_id_text + datetime_data.strftime('%Y-%m-%d')
                # x_data = timestamp_data.values
                x_data = datetime_data.values
                y_data = result_df[COL_CAP_REMAINING].values / CAP_NOMINAL * 100.0
                show_line = True
            else:
                # empty DataFrame -> show in legend anyway, this is intentional
                x_data = []
                y_data = []
                text_data = []
                show_line = False
                this_color = 'rgba(255,255,255,0.0)'

            # add line
            result_plot.add_result_trace(fig, 0, i_col, x_data, y_data, this_color, False, show_line,
                                         text_data=text_data, line_dash=dash, line_width=line_width,
                                         show_legend=True, name=name, zorder=zorder, legend_name=legend_name)
            if SHOW_SCENARIO_AT_LINE_END:  # not sure if this works as expected with multiple columns
                # add text
                x_text = x_data[-1] + np.timedelta64(365 * 24 * 3600 * 1000 * 1000 * 1000)  # add 1 year
                y_text = y_data[-1]
                fig.add_annotation(xref="x", yref="y", x=x_text, y=y_text, align="left",
                                   showarrow=False, text=name, bgcolor=result_plot.BG_COLOR,
                                   font=dict(size=ANNOTATION_FONT_SIZE, color=this_color))

        # change color of "background" labels in legend:
        pass

    return fig


def plot_enhance(fig: go.Figure, n_cols, i_fig):
    x_data = X_LIM_ARR[0]
    major_tickvals = [datetime.datetime(y, 1, 1, tzinfo=TZ_INFO_DEFAULT)
                      for y in range(X_LIM_ARR[0][0].year, X_LIM_ARR[0][1].year + 1, X_TICK_YEARS_MAJOR)]
    major_tick_labels = [major_tickvals[i].year for i in range(0, len(major_tickvals))]
    minor_tickvals = [datetime.datetime(y, m, 1, tzinfo=TZ_INFO_DEFAULT) for m in range(1, 13, X_TICK_MONTHS_MINOR)
                      for y in range(X_LIM_ARR[0][0].year, X_LIM_ARR[0][1].year + 1)]
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    for i_col in range(n_cols):
        for i in range(len(Y_LIMITS)):
            color = color_add_alpha(Y_LIMIT_COLORS[i], LIMIT_ALPHA)
            result_plot.add_result_trace(fig, 0, i_col, x_data, [Y_LIMITS[i], Y_LIMITS[i]], color,
                                         False, True, line_width=LIMIT_LINE_WIDTH)

        fig.update_xaxes(tickvals=major_tickvals,
                         ticktext=major_tick_labels,
                         minor=dict(tickvals=minor_tickvals),
                         title_standoff=17,
                         row=1, col=(i_col + 1))
        fig.update_yaxes(title_standoff=17,
                         row=1, col=(i_col + 1))

        if (PLOT_TEXTS is not None) and (len(PLOT_TEXTS) > i_fig):
            if (PLOT_TEXTS[i_fig] is not None) and (len(PLOT_TEXTS[i_fig]) > i_col):
                for this_annotation in PLOT_TEXTS[i_fig][i_col]:
                    if this_annotation is not None:
                        fig.add_annotation(dict(font=dict(color=this_annotation[PLOT_TEXT_COLOR],
                                                          size=PLOT_TEXT_FONTSIZE),
                                                x=this_annotation[PLOT_TEXT_X],
                                                y=this_annotation[PLOT_TEXT_Y],
                                                showarrow=False,
                                                text=this_annotation[PLOT_TEXT_STRING],

                                                # textangle=0,
                                                xanchor='center',
                                                xref="x",
                                                yref="y"),
                                           row=1, col=(i_col + 1))

    # fig.update_legends(x=1.01, y=0, itemwidth=50)
    return fig


def read_remaining_cap_data(sc_result_file_dict):
    logging.log.info("Reading scenarios...")
    remaining_cap_dict = {}
    for sc_id, sc_dict in sc_result_file_dict.items():
        logging.log.debug("Reading scenario %03u" % sc_id)
        # result_df = pd.read_csv(IMPORT_PATH + sc_dict[fd_filename], header=0, sep=CSV_SEP,
        #                         usecols=IMPORT_COLUMNS, skiprows=lambda i: i % IMPORT_DOWNSAMPLE_N)
        if sc_dict[fd_compressed]:
            result_df = pd.read_csv(IMPORT_PATH + sc_dict[fd_filename], header=0, sep=CSV_SEP,
                                    usecols=list(IMPORT_COLUMNS_B.keys()), engine="pyarrow", dtype=IMPORT_COLUMNS_B)
        else:
            # result_df = pd.read_csv(IMPORT_PATH + sc_dict[fd_filename], header=0, sep=CSV_SEP,
            #                         usecols=IMPORT_COLUMNS, engine="pyarrow")
            result_df = pd.DataFrame()
            read_dx = pd.read_csv(IMPORT_PATH + sc_dict[fd_filename], header=0, sep=CSV_SEP,
                                  usecols=IMPORT_COLUMNS, chunksize=100000)
            for chunk_df in read_dx:
                chunk_df.dropna(axis=0, inplace=True)
                result_df = pd.concat([result_df, chunk_df], axis=0)

        # result_df.dropna(inplace=True)

        result_df.set_index(COL_TIME, inplace=True)
        if not sc_dict[fd_compressed]:
            # convert date to text
            timestamp_data = result_df.index
            datetime_data = pd.to_datetime(timestamp_data.astype(np.int64), unit="s",
                                           utc=True).tz_convert(TIMEZONE_DEFAULT)
            text_data = datetime_data.strftime('%Y-%m-%d')
            result_df[COL_TIME_TEXT] = text_data

            # export compressed version
            result_df.to_csv(IMPORT_PATH + sc_dict[fd_filename_compressed], index=True, sep=CSV_SEP)
        remaining_cap_dict[sc_id] = result_df

    if SHOW_ALL_KNOWN_SCENARIOS:
        for sc_id in SC_LINE_SETTINGS.keys():
            if sc_id not in remaining_cap_dict.keys():
                # add empty DataFrame, so we know that there is no data
                remaining_cap_dict[sc_id] = pd.DataFrame()

    return remaining_cap_dict


def get_file_dict():
    load_all = False
    load_scenarios = np.array([], int)
    for i_fig in range(len(PLOT_SCENARIOS)):
        for i_col in range(len(PLOT_SCENARIOS[i_fig])):
            if PLOT_SCENARIOS[i_fig][i_col] is None:
                load_all = True
                break
            if any(PLOT_SCENARIOS[i_fig][i_col]) is None:
                load_all = True
                break
            load_scenarios = np.union1d(load_scenarios, PLOT_SCENARIOS[i_fig][i_col])
        if load_all:
            break
    for i_fig in range(len(PLOT_SCENARIOS_BG)):
        for i_col in range(len(PLOT_SCENARIOS_BG[i_fig])):
            if PLOT_SCENARIOS_BG[i_fig][i_col] is None:
                break
            if any(PLOT_SCENARIOS_BG[i_fig][i_col]) is None:
                break
            load_scenarios = np.union1d(load_scenarios, PLOT_SCENARIOS_BG[i_fig][i_col])

    res_csv = {}  # find .csv files: use_case_model_007_modular_driving_sc020_2024-05-24_18-04.csv -> find latest
    with os.scandir(IMPORT_PATH) as iterator:
        re_pat_csv = re.compile(IMPORT_FILENAME_BASE)  # raw file
        re_pat_csv_b = re.compile(IMPORT_FILENAME_B_BASE)  # condensed file (only remaining capacity)
        for entry in iterator:
            filename = entry.name
            re_match_csv = re_pat_csv.fullmatch(filename)
            re_match_csv_b = re_pat_csv_b.fullmatch(filename)
            if re_match_csv_b:
                sc_id = int(re_match_csv_b.group(1))
                if (not load_all) and (sc_id not in load_scenarios):
                    # logging.log.debug("Skipping scenario %u (not in PLOT_SCENARIOS list)" % sc_id)
                    continue
                year = int(re_match_csv_b.group(2))
                month = int(re_match_csv_b.group(3))
                day = int(re_match_csv_b.group(4))
                hour = int(re_match_csv_b.group(5))
                minute = int(re_match_csv_b.group(6))
                dt_timestamp = datetime.datetime(year, month, day, hour, minute)
                if (sc_id not in res_csv) or (dt_timestamp >= res_csv[sc_id][fd_date]):  # intentional '>='
                    res_csv[sc_id] = {fd_filename: filename, fd_date: dt_timestamp,
                                      fd_filename_compressed: filename, fd_compressed: True}
            elif re_match_csv:
                sc_id = int(re_match_csv.group(1))
                if (not load_all) and (sc_id not in load_scenarios):
                    logging.log.debug("Skipping scenario %u (not in PLOT_SCENARIOS list)" % sc_id)
                    continue
                year = int(re_match_csv.group(2))
                month = int(re_match_csv.group(3))
                day = int(re_match_csv.group(4))
                hour = int(re_match_csv.group(5))
                minute = int(re_match_csv.group(6))
                dt_timestamp = datetime.datetime(year, month, day, hour, minute)
                filename_compressed = IMPORT_FILENAME_B_WRITE_BASE % (sc_id, year, month, day, hour, minute)
                if (sc_id not in res_csv) or (dt_timestamp > res_csv[sc_id][fd_date]):  # intentional '>'
                    res_csv[sc_id] = {fd_filename: filename, fd_date: dt_timestamp,
                                      fd_filename_compressed: filename_compressed, fd_compressed: False}
    return res_csv


def color_add_alpha(rgb_string, alpha):
    re_match = re_pat_color_rgb.fullmatch(rgb_string)
    rgb_arr = np.array([127, 127, 127])  # fallback
    if re_match:
        rgb_arr = np.array([int(re_match.group(1)), int(re_match.group(2)), int(re_match.group(3))])  # r, g, b
    return f"rgba(%u,%u,%u,%f)" % (int(rgb_arr[0]), int(rgb_arr[1]), int(rgb_arr[2]), alpha)


if __name__ == "__main__":
    run()
