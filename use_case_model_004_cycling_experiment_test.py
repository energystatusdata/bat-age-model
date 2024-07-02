# example: Preliminary tests for a simulation of the battery aging experiment conducted for the dissertation:
# "Robust electricity grids through intelligent, highly efficient bidirectional charging systems for electric vehicles"
# (Dissertation of Matthias Luh, 2024)
# ... explained in chapter 7.1
# the full experiment is simulated in use_case_model_005_cycling_experiment.py
# ToDo: if the figure closes immediately after opening, debug and set a breakpoint at the 'print("debug")' statement

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import bat_model_v01 as bat


matplotlib.use('TkAgg')

# constants
T_RESOLUTION_ACTIVE = 5  # in seconds, temporal resolution for modeling an active cell (charging, discharging)
T_RESOLUTION_PROFILE = 1  # in seconds, temporal resolution for modeling the profile (it is in seconds!)
T_RESOLUTION_REST = 60  # in seconds, temporal resolution for modeling a resting cell (idle)

bat.R_TH_CELL = 3  # in K/W, rough estimation of the thermal resistance - active liquid cooling: 3

CELL_STORAGE_TIME_DAYS = 685  # Nov 26, 2020 to Oct 12, 2022 -> 685 days
# CELL_STORAGE_VOLTAGE = 3.5558  # in V, +/- 50 mV, voltage at which the cell was stored before experiment (as arrived)
CELL_STORAGE_SOC = 0.267  # in % (1 = 100%), +/- 1%, SoC at which the cell was stored before the experiment
CELL_STORAGE_TEMPERATURE = 18

TEMP_RT = 25
N_CHECKUPS = 2  # 23  # 2 (for testing)
FIRST_CHECKUP_INTERVAL_S = (7 * 24 * 3600)
NEXT_CHECKUP_INTERVAL_S = (21 * 24 * 3600)
CYCLING_PAUSE = 5 * 60

# cyclic aging cell
TEMP_OT = 40
V_MAX_OT = 4.0921  # bat.get_ocv_from_soc(0.9)
V_MIN_OT = 3.249  # bat.get_ocv_from_soc(0.1)
I_CHG = 5.0
I_DISCHG = -3.0
I_CHG_CUTOFF = 0.3
I_DISCHG_CUTOFF = -0.3


def run():
    cap_aged, aging_states, temp_cell, soc = bat.init(storage_time_days=CELL_STORAGE_TIME_DAYS,
                                                      storage_soc=CELL_STORAGE_SOC,
                                                      storage_temperature=CELL_STORAGE_TEMPERATURE)
    t_start = 1665593100  # EXPERIMENT_START_TIMESTAMP -> Mi, 12.10.2022  16:45:00 UTC

    # wait 2 hours at room temperature
    duration = 2 * 3600
    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_pause(
        t_start, T_RESOLUTION_REST, duration, TEMP_RT, None, None, None, None, None,
        cap_aged, aging_states, temp_cell, soc)

    # initial check-up
    print("Starting check-Up 1 of %u" % N_CHECKUPS)
    t_cycling_end = t_start + FIRST_CHECKUP_INTERVAL_S
    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_checkup(
        t_start, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, V_MIN_OT, I_CHG, I_DISCHG, I_CHG_CUTOFF, I_DISCHG_CUTOFF,
        TEMP_OT, v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # cycling
    print("cycling...")
    # v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_cycles(
    #     None, t_start, t_cycling_end, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, V_MAX_OT, V_MIN_OT,
    #     I_CHG, I_DISCHG, I_CHG_CUTOFF, I_DISCHG_CUTOFF, CYCLING_PAUSE, True,
    #     TEMP_OT, v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc)
    # profile cycling
    v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_profile_cycles(
        None, t_start, t_cycling_end, T_RESOLUTION_ACTIVE, T_RESOLUTION_PROFILE, T_RESOLUTION_REST,
        bat.wltp_profiles.extra_high, V_MAX_OT, V_MIN_OT, I_CHG, I_CHG_CUTOFF, CYCLING_PAUSE, True, TEMP_OT,
        v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    for i_cu in range(2, N_CHECKUPS + 1):
        print("Check-Up %u of %u" % (i_cu, N_CHECKUPS))
        # check-up
        v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_checkup(
            t_start, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, V_MIN_OT, I_CHG, I_DISCHG, I_CHG_CUTOFF, I_DISCHG_CUTOFF,
            TEMP_OT, v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc)
        t_cycling_end = t_start + NEXT_CHECKUP_INTERVAL_S

        # cycling
        print("cycling...")
        # v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_cycles(
        #     None, t_start, t_cycling_end, T_RESOLUTION_ACTIVE, T_RESOLUTION_REST, V_MAX_OT, V_MIN_OT,
        #     I_CHG, I_DISCHG, I_CHG_CUTOFF, I_DISCHG_CUTOFF, CYCLING_PAUSE, True,
        #     TEMP_OT, v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc)
        # profile cycling
        v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = bat.apply_profile_cycles(
            None, t_start, t_cycling_end, T_RESOLUTION_ACTIVE, T_RESOLUTION_PROFILE, T_RESOLUTION_REST,
            bat.wltp_profiles.full, V_MAX_OT, V_MIN_OT, I_CHG, I_CHG_CUTOFF, CYCLING_PAUSE, True, TEMP_OT,
            v_df, i_df, p_df, temp_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # plot results
    fig = plt.figure()
    plt_p = fig.add_subplot(5, 1, 1)
    plt_i = fig.add_subplot(5, 1, 2, sharex=plt_p)
    plt_v = fig.add_subplot(5, 1, 3, sharex=plt_p)
    plt_soc = fig.add_subplot(5, 1, 4, sharex=plt_p)
    plt_t = fig.add_subplot(5, 1, 5, sharex=plt_p)

    # time = np.array(p_df.index) / 3600.0  # in hours
    time = pd.to_datetime(p_df.index, unit="s")
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

    plt_t.plot(time, temp_df, c="green", linestyle='solid')
    # plt_t.set_title("Ambient temperature (dashed) and cell temperature (solid) [°C]")
    plt_t.set_ylabel("Temperature [°C]")
    plt_t.grid(True)

    plt_t.set_xlabel("time [h]")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(120)

    print("debug")


if __name__ == "__main__":
    run()
