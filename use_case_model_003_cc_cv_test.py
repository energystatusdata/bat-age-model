# example: various CC-CV charging/discharging and ambient temperature variation tests
# ToDo: if the figure closes immediately after opening, debug and set a breakpoint at the 'print("debug")' statement

# import math
# import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from datetime import datetime

import bat_model_v01 as bat


matplotlib.use('TkAgg')

# constants
T_RESOLUTION_ACTIVE = 5  # in seconds, temporal resolution for modeling an active cell (charging, discharging)
T_RESOLUTION_REST = 60  # in seconds, temporal resolution for modeling a resting cell (idle)
TEMP_AMBIENT = 10
bat.R_TH_CELL = 3  # in K/W, rough estimation of the thermal resistance - active liquid cooling: 3


def run():
    cap_aged, aging_states, temp_cell, soc = bat.init()

    # rest for 12 hours
    t_start = 0
    duration = (12 * 3600)
    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
        t_start, T_RESOLUTION_REST, duration, TEMP_AMBIENT, None, None, None, None, None,
        cap_aged, aging_states, temp_cell, soc)

    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_cycles(
        3, t_start, None, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, 4.2, 2.5, 2.0, -4.0, 0.2, -0.4, (5 * 60), True,
        TEMP_AMBIENT, v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    duration = (1 * 3600)
    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
        t_start, T_RESOLUTION_REST, duration, TEMP_AMBIENT, v_df, i_df, p_df, temp_df, soc_df,
        cap_aged, aging_states, temp_cell, soc)

    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_cc_cv(
        t_start, T_RESOLUTION_ACTIVE, 3.7, 5.0, 0.15, TEMP_AMBIENT,
        v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    duration = (0.5 * 3600)
    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
        t_start, T_RESOLUTION_REST, duration, TEMP_AMBIENT, v_df, i_df, p_df, temp_df, soc_df,
        cap_aged, aging_states, temp_cell, soc)

    t_max = t_start + (5 * 3600)
    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_cycles(
        None, t_start, t_max, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, 4.2, 3.0, 2.0, -4.0, 0.2, -0.4, (5 * 60), False,
        TEMP_AMBIENT, v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_checkup(
        t_start, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, 3.0, 2.0, -4.0, 0.2, -0.4, TEMP_AMBIENT,
        v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc
    )

    # plot results
    fig = plt.figure()
    plt_p = fig.add_subplot(5, 1, 1)
    plt_i = fig.add_subplot(5, 1, 2, sharex=plt_p)
    plt_v = fig.add_subplot(5, 1, 3, sharex=plt_p)
    plt_soc = fig.add_subplot(5, 1, 4, sharex=plt_p)
    plt_t = fig.add_subplot(5, 1, 5, sharex=plt_p)

    time = np.array(p_df.index) / 3600.0  # in hours
    plt_p.plot(time, p_df, c="red")
    plt_p.set_ylabel("Power [W]")
    plt_p.grid(True)

    plt_i.plot(time, i_df, c="orange")
    plt_i.set_ylabel("Current [A]")
    plt_i.grid(True)

    plt_v.plot(time, v_df, c="blue")
    plt_v.set_ylabel("Voltage [V]")
    plt_v.grid(True)

    plt_soc.plot(time, soc_df, c="yellow")
    plt_soc.set_ylabel("SoC [0..1]")
    plt_soc.grid(True)

    plt_t.plot(time, [TEMP_AMBIENT] * len(time), c="gray", linestyle='dashed')
    plt_t.plot(time, temp_df, c="green", linestyle='solid')
    # plt_t.set_title("Ambient temperature (dashed) and cell temperature (solid) [°C]")
    plt_t.set_ylabel("Temperature [°C]")
    plt_t.grid(True)

    plt_t.set_xlabel("time [h]")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(120)

    print("debug")  # set a breakpoint here


if __name__ == "__main__":
    run()
