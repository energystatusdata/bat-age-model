# example: a simple repeated charge-discharge test using different power values with the aging model
# it can be used to determine the power-dependent SoC <-> SoE dependency of the battery cell

import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

import bat_model_v01 as bat


matplotlib.use('TkAgg')

# constants
TEMP_AMBIENT = 25


def run():
    # test for each power in the array, using the resolution and duration in the following arrays
    power_arr = [0.1, 0.22, 0.46, 1.0, 1.5, 2.2, 3.2, 4.6, 6.8, 10.0, 15.0, 22.0]  # , 32.0]
    T_RESOLUTION_ARR = [60, 60, 60, 30, 10, 10, 5, 5, 2, 1, 1, 1]  # , 1]
    # NUM_STEPS_ARR = np.array([500, 300, 150, 100, 80, 60, 40, 30, 20, 10, 5, 3, 1]) * 60  # [hours] * 60 = minutes
    NUM_STEPS_ARR = np.array([300, 200, 120, 120, 120, 120, 120, 120, 180, 300, 240, 240]) * 60  # [hours] * 60 = min.

    # initialize dataframes
    soc_soe_chg_df = pd.DataFrame(np.nan, columns=power_arr, index=(np.array(range(0, 102, 2)) / 100.0))
    soc_soe_dischg_df = soc_soe_chg_df.copy()

    for i_pwr in range(len(power_arr)):  # for each power to test
        pwr = power_arr[i_pwr]
        T_RESOLUTION = T_RESOLUTION_ARR[i_pwr]
        T_STEP = T_RESOLUTION

        cap_aged, aging_states, temp_cell, soc = bat.init()

        num_steps = NUM_STEPS_ARR[i_pwr]  # (500 * 60)  # 500 hours
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

        t_chg_start = 0.01 * num_steps * T_STEP
        t_chg_end = 0.49 * num_steps * T_STEP
        t_dischg_start = 0.51 * num_steps * T_STEP
        t_dischg_end = 0.99 * num_steps * T_STEP
        cond_chg = (dt_step_df.index >= t_chg_start) & (dt_step_df.index < t_chg_end)
        cond_dischg = (dt_step_df.index >= t_dischg_start) & (dt_step_df.index < t_dischg_end)
        cond_soc_chg = (soc_df.index >= t_chg_start) & (soc_df.index < t_chg_end)
        cond_soc_dischg = (soc_df.index >= t_dischg_start) & (soc_df.index < t_dischg_end)
        p_set_df[cond_chg] = pwr
        p_set_df[cond_dischg] = -pwr

        # for this test, reset cap_aged to cap_nominal and start at SoC = 0
        cap_aged = bat.CAP_NOMINAL
        soc = 0.0

        print("start looping...")
        start_timestamp = datetime.now()
        for i in range(num_steps):  # for each simulation step
            soc_df[i * T_STEP] = soc
            ixs = ((dt_step_df.index >= i * T_STEP) & (dt_step_df.index < ((i + 1) * T_STEP)))
            cap_aged, aging_states, temp_cell, soc, v_cell_step_df, i_cell_step_df, p_actual_step_df, temp_cell_step_df\
                = (bat.get_aging_step(dt_step_df[ixs], None, p_set_df[ixs], temp_ambient_df[ixs],
                                      cap_aged, aging_states, temp_cell, soc))
            p_actual_df[ixs] = p_actual_step_df
            v_cell_df[ixs] = v_cell_step_df
            i_cell_df[ixs] = i_cell_step_df
            temp_cell_df[ixs] = temp_cell_step_df

        soc_df[num_steps * T_STEP] = soc

        delta_e_chg_df = (p_actual_df[cond_chg] * dt_step_df[cond_chg]).cumsum() / 3600.0  # > 0
        delta_e_dischg_df = (p_actual_df[cond_dischg] * dt_step_df[cond_dischg]).cumsum() / 3600.0  # < 0
        e_chg = max(delta_e_chg_df)  # > 0
        e_dischg = -min(delta_e_dischg_df)  # > 0
        stop_timestamp = datetime.now()
        print("finished looping pwr = %.1f, (took %s h:mm:ss.ms)\n"
              "   total charging energy [Wh]: %.3f\n"
              "   total discharging energy [Wh]: %.3f"
              % (pwr, str(stop_timestamp - start_timestamp), e_chg, e_dischg))

        for this_soc in soc_soe_chg_df.index:  # find SoC in soc_df. charging -> SoC increasing
            ixs = soc_df[cond_soc_chg & (soc_df >= this_soc)].index
            if len(ixs) > 0:
                ix = ixs[0]
            else:
                ixs = soc_df[cond_soc_chg & (soc_df <= this_soc)].index
                if len(ixs) > 0:
                    ix = ixs[-1]
                else:
                    print("============ WARNING: didn't find SoC in DataFrame (charging) ============")
                    continue
            if ix in delta_e_chg_df:
                soe = delta_e_chg_df[ix] / e_chg  # > 0
                soc_soe_chg_df.loc[this_soc, pwr] = soe
            else:
                print("============ WARNING: delta_e_chg_df[%u] does not exist ============" % ix)
                continue

        for this_soc in soc_soe_dischg_df.index:  # find SoC in soc_df. discharging -> SoC decreasing
            ixs = soc_df[cond_soc_dischg & (soc_df <= this_soc)].index
            if len(ixs) > 0:
                ix = ixs[0]
            else:
                ixs = soc_df[cond_soc_dischg & (soc_df >= this_soc)].index
                if len(ixs) > 0:
                    ix = ixs[-1]
                else:
                    print("============ WARNING: didn't find SoC in DataFrame (discharging) ============")
                    continue
            if ix in delta_e_dischg_df:
                soe = 1.0 + delta_e_dischg_df[ix] / e_dischg  # 1 + (< 0) -> [0..1]
                soc_soe_dischg_df.loc[this_soc, pwr] = soe
            else:
                print("============ WARNING: delta_e_dischg_df[%u] does not exist ============" % ix)
                continue

        print("generation of soc_soe_df column complete")

        # plot results
        fig = plt.figure()
        plt_p = fig.add_subplot(5, 1, 1)
        plt_i = fig.add_subplot(5, 1, 2, sharex=plt_p)
        plt_v = fig.add_subplot(5, 1, 3, sharex=plt_p)
        plt_soc = fig.add_subplot(5, 1, 4, sharex=plt_p)
        plt_t = fig.add_subplot(5, 1, 5, sharex=plt_p)

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

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(.1)

    print("soc_soe_chg_df:\n" + soc_soe_chg_df.to_string())
    print("soc_soe_dischg_df:\n" + soc_soe_dischg_df.to_string())

    print("debug")


if __name__ == "__main__":
    run()
