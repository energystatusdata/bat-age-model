# example: a simple charge-discharge test using the aging model

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import bat_model_v01 as bat


# constants
T_RESOLUTION = 1  # in seconds, temporal resolution of modeling
T_STEP = 60  # in seconds, duration of a simulation step (soc, aging, ...)

TEMP_AMBIENT = 25


def run():
    cap_aged, aging_states, temp_cell, soc = bat.init()

    num_steps = 150
    dt_step_df = pd.Series(T_RESOLUTION,
                           index=(np.array(range(math.ceil(num_steps * T_STEP / T_RESOLUTION))) * T_RESOLUTION))
    p_set_df = pd.Series(0, index=dt_step_df.index)
    temp_ambient_df = pd.Series(TEMP_AMBIENT, index=dt_step_df.index)  # constant ambient temperature

    p_actual_df = pd.Series(np.nan, index=dt_step_df.index)
    v_cell_df = p_actual_df.copy()
    i_cell_df = p_actual_df.copy()
    temp_cell_df = p_actual_df.copy()
    soc_df = pd.Series(np.nan, index=(np.array(range(num_steps + 1)) * T_STEP))

    # let the cell start at ambient temperature
    temp_cell = temp_ambient_df.iloc[0]
    # define p_set profile
    # rest 10 seconds
    p_set_df[(dt_step_df.index >= 10) & (dt_step_df.index < 2510)] = 18  # charge for 2500 seconds with 18 W (constant)
    # rest 40 seconds
    p_set_df[(dt_step_df.index >= 2550) & (dt_step_df.index < 8600)] = -18  # discharge for 40 seconds with 20 W
    # rest 400 seconds

    print("start looping...")
    start_timestamp = datetime.now()
    for i in range(num_steps):
        soc_df[i * T_STEP] = soc
        ixs = ((dt_step_df.index >= i * T_STEP) & (dt_step_df.index < ((i + 1) * T_STEP)))
        cap_aged, aging_states, temp_cell, soc, v_cell_step_df, i_cell_step_df, p_actual_step_df, temp_cell_step_df = (
            bat.get_aging_step(dt_step_df[ixs], None, p_set_df[ixs], temp_ambient_df[ixs],
                               cap_aged, aging_states, temp_cell, soc))
        p_actual_df[ixs] = p_actual_step_df
        v_cell_df[ixs] = v_cell_step_df
        i_cell_df[ixs] = i_cell_step_df
        temp_cell_df[ixs] = temp_cell_step_df

    soc_df[num_steps * T_STEP] = soc
    stop_timestamp = datetime.now()
    print("finished looping (took %s h:mm:ss.ms)" % str(stop_timestamp - start_timestamp))

    # plot results
    fig = plt.figure()
    plt_p = fig.add_subplot(5, 1, 1)
    plt_i = fig.add_subplot(5, 1, 2)
    plt_v = fig.add_subplot(5, 1, 3)
    plt_soc = fig.add_subplot(5, 1, 4)
    plt_t = fig.add_subplot(5, 1, 5)

    plt_p.plot(p_set_df, c="black", linestyle='dashed')
    plt_p.plot(p_actual_df, c="red", linestyle='solid')
    plt_p.set_title("Cell set power (dashed) and actual power (solid) power [W]")
    plt_p.set_ylabel("Power [W]")
    plt_p.grid(True)

    plt_i.plot(i_cell_df, c="orange")
    plt_i.set_ylabel("Current [A]")
    plt_i.grid(True)

    plt_v.plot(v_cell_df, c="blue")
    plt_v.set_ylabel("Voltage [V]")
    plt_v.grid(True)

    plt_soc.plot(soc_df, c="yellow")
    plt_soc.set_ylabel("SoC [0..1]")
    plt_soc.grid(True)

    plt_t.plot(temp_ambient_df, c="green", linestyle='dashed')
    plt_t.plot(temp_cell_df, c="green", linestyle='solid')
    # plt_t.set_title("Ambient temperature (dashed) and cell temperature (solid) [°C]")
    plt_t.set_ylabel("Temperature [°C]")
    plt_t.grid(True)

    # plt.tight_layout()
    plt.show()

    print("debug")


if __name__ == "__main__":
    run()
