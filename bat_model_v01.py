# Battery degradation model (capacity fade) derived from experimental cell aging test data, as described in:
# "Robust electricity grids through intelligent, highly efficient bidirectional charging systems for electric vehicles"
# (Dissertation of Matthias Luh, 2024)
#
# The model and the variables are explained in Chapters 7.2 and 7.3 of the dissertation.
# Note: The "..._fast.py" models run faster, but do not return a voltage/current/power/temperature profile of the cell.
#       That makes you a bit "blind", because you only get the capacity fade of the cell, but don't see what it did.
#       I suggest starting with the "regular" models until you are absolutely sure about what you do and that the
#       results are legitimate. Then, you can compare the results with the fast model and continue there.

import math
import typing
import pandas as pd
import numpy as np
import wltp_profiles  # may be needed in use case models - leave it in here!


# battery operation limits - the battery model will decrease the power/current to stay within the thresholds below:
V_CELL_MAX = 4.2  # in V, maximum allowed cell voltage (model specified for ≤ 4.2 V)
V_CELL_MIN = 2.5  # in V, minimum allowed cell voltage (model specified for ≥ 2.5 V - probably also works with ≥ 2 A)
I_CELL_MAX_CHG = 5.0  # in A, >0, maximum allowed cell charging current (specified for ≤ 5 A - might work with 6 A)
I_CELL_MIN_DISCHG = -6.0  # in A, <0, max. allowed cell discharging current (specified for ≥ -3 A - might work with -6)

# (simple!) internal impedance/resistance model: R_CELL = R_CELL_0 + R_CELL_AGE * (1 - C_remaining / C_nominal)
R_CELL_0 = 0.05  # in Ohm, internal resistance of a new cell
R_CELL_AGE = 0.1  # in Ohm, aging-dependent part ot the internal resistance of a new cell
# R_CELL_0 = 0.05, R_CELL_AGE = 0.1 -> if 75% (50%) of the capacity remaining, R_CELL = 0.075 (0.1) Ohm

# (simple!) thermal model -> thermal resistance of cell temperature to ambient/coolant and thermal capacity of cell
R_TH_CELL = 15  # in K/W, rough estimation of the thermal resistance - passive air cooling: 15, active liquid cooling: 3
C_TH_CELL = 30  # in J/K, rough estimation of the thermal capacity - regardless of the cooing: 10-50, e.g. 30 J/K


# constants -> do not change them unless you know what you do - the aging model will depend on it!
CAP_NOMINAL = 3  # in Ah, nominal capacity of the cell
E_NOMINAL = 11  # in Wh, nominal energy of the cell
V_NOMINAL = 3.6  # in V, nominal cell voltage
R_TH_C_TH_CELL = R_TH_CELL * C_TH_CELL
T_0_DEGC_IN_K = 273.15  # 0°C in Kelvin

# default values for the storage of the cell between production date and usage of the battery
CAP_INITIAL = CAP_NOMINAL * 1.03  # in Ah, actual capacity of the cell right after production (e.g., +3%)
STORAGE_TIME_DEFAULT = 30  # in days, default time between production date and usage of the battery
STORAGE_SOC_DEFAULT = 0.25  # in 100%, i.e., [0..1], storage soc between production and initial battery usage
STORAGE_TEMPERATURE_DEFAULT = 18  # in °C, average (or age-weighted) storage temperature between production and usage

# === aging parameters ===
# general aging parameters
T_REF_KELVIN = 25.0 + T_0_DEGC_IN_K  # reference temperature
V_REF = 3.73  # reference voltage
AGE_APPLY_PERIOD = 30  # in seconds, use average of this period for aging. If dt_resolution is larger, use the latter.

# SEI growth:
AGE_S0 = 1.49e-9   # base SEI growth rate -> higher = faster SEI growth / more aging
AGE_S1 = -2375     # SEI temperature dependency (negative: hotter = more aging) -> higher = steeper temp. dependency
AGE_S2 = 1.2       # SEI voltage dependency (pos.: higher voltage = more aging) -> higher = steeper voltage dependency
AGE_S3 = 1.78e-8   # SEI growth counterforce/limit -> higher = SEI growth becomes more limited
# cyclic wearout
AGE_W0 = 2.67e-7   # base wearout rate -> higher = more cyclic wearout
AGE_W1 = 2.25      # acceleration of wearout rate for an aged cell -> higher = aged cell will have higher wearout rate
AGE_W2 = 0.14      # acceleration tipping point -> higher = tipping point at higher q_loss. 0.14 = tipping at SoH = 86%
AGE_W3 = 9.5e-7    # wearout counterforce/limit -> higher = wearout rate becomes more limited
# extra wearout at low voltages (higher volume change? current collector corrosion? other processes / side reactions?)
AGE_C0 = 3.60e-5  # wearout rate at low voltages -> higher = more wearout
AGE_C1 = 1050     # temperature dependency (pos.: colder = more aging) -> higher = steeper temp. dependency
AGE_C2 = 3.2      # threshold voltage in V below which the losses occur
AGE_C3 = 2.47e-4  # loss counterforce/limit -> higher = loss rate becomes more limited
# lithium plating - effective resistance: r_eff_rel = p0 + (p1 * max(p2 - T_cell, 0))^p3 * exp(p4 * q_loss) * Cc_rel)
AGE_P0 = 0.07     # effective base resistance (relative) -> higher = plating occurs more easily at all temperatures
AGE_P1 = 0.029    # temperature dependent part of effective resistance -> higher = temp. dependent part is more relevant
AGE_P2 = 41.5 + T_0_DEGC_IN_K  # temp. in K above which AGE_P1 is irrelevant -> higher = p1 also relevant at higher T
AGE_P3 = 3.5      # temperature dependency -> higher = plating occurs more easily at colder temp. vs. medium temp.
AGE_P4 = 0.33     # aging dependency of effective res. (tipping behavior) -> higher = more R_eff increase for aged cell
AGE_P5 = 5.3e-8   # base plating rate -> higher = if plating occurs, lithium is plated faster
AGE_P6 = 2.15     # C-rate dependency of plating rate (Cc^p6) -> higher = steeper dependency if C-rate > 1, lower if < 1


# states (if we want multithreading support, it might be the best to pass them back- and forth to the user function
# instead of "storing" them here into a variable
# - SoC (<-> SoE)
# - T_cell_int
# - C_remaining
# - internal aging states:
#   - q_loss_sei_total
#   - q_loss_cyclic_total
#   - q_loss_cyclic_low_total  -> used to be q_loss_corrosion, but I'm not sure if it actually is corrosion
#   - q_loss_plating_total
#   - q_loss_total (optional, for user convenience -> no, is in C_remaining)

# inputs:
#   - P_set
#   - delta_time
#   - T_ambient/coolant
# outputs:
#   - SoC, SoE
#   - P_actual, V_cell, I_cell, T_cell_case
#   - states (including C_remaining, maybe extra for convenience)

# ToDo: this can be called for each simulation step of the model (e.g., for second-resolution, this dataframe could be a
#  one-minute step containing 60 one-second micro-steps. For a 10-second resolution, this dataframe could be a
#  two-minute step containing 12 ten-second micro-steps.
#  However, the step shouldn't be too long, since the voltage/current/temperature-dependent aging is only calculated
#  once per step (using the end values). How long is "long" depends on the maximum charge rate. 50-100 steps per
#  0-100% charge cycle should be fine, 20 are probably not enough, 1000 might be excessive. If the cell is unused, the
#  steps (and dt_df micro-steps) can also be large (e.g. hourly or daily).
#  If your application needs the SoC-feedback, this may also limit the maximum step duration, since the p_set_df may be
#  cut if the cell is full or empty (which you will see in p_actual_df). p_set_df is also limited if the necessary cell
#  current exceeds I_CELL_MAX_CHG / I_CELL_MAX_DISCHG.
# inputs:   dt_df, i_set_df, p_set_df, temp_amb_df, cap_aged_begin, aging_states, temp_cell, soc
#   dt_df           pandas.Series   delta time in s, how long are each of the corresponding values of variables applied?
#   i_set_df        pandas.Series   OR desired set power in A, will be limited if it violates the cell operation limits
#   p_set_df        pandas.Series   desired set power in W, will be limited if it violates the cell operation limits
#       --> only use one of p_set_df or i_set_df and set the other to None
#   temp_ambient_df pandas.Series   ambient (or coolant) temperature in °C -> R_TH_CELL "between" temp_ambient/temp_cell
#   cap_aged_begin  float           remaining usable capacity of the cell in Ah before step (cap_aged_end of last step)
#   aging_states    arr. of floats  internal aging states (different aging types [0...1]), use aging_states of last step
#   temp_cell       float           cell (case) temperature in °C at start (use temp_cell of the last time step)
#   soc             float           State of Charge [0...1] of the cell at start (use soc of the last time step)
# outputs:  cap_aged_end, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df
#   cap_aged_end    float           remaining usable capacity of the cell in Ah after step (-> cap_aged_begin of next)
#   aging_states    arr. of floats  internal aging states, use for next step
#   temp_cell       float           new cell (case) temperature in °C (use for temp_cell of next step)
#   soc             float           new capacity-based State of Charge [0...1] of cell (use for soc of next step)
#   v_cell_df       pandas.Series   cell voltage profile in V (over the index given in dt_df - informative, not needed)
#   i_cell_df       pandas.Series   actual cell current profile in A (over dt_df.index - informative, not needed)
#   p_cell_df       pandas.Series   actual cell power profile in W (over dt_df.index - informative, not needed)
#   temp_cell_df    pandas.Series   cell (case) temperature profile in °C (over dt_df.index - informative, not needed)
#   soc_df          pandas.Series   cell State of Charge [0...1] profile (over dt_df.index - informative, not needed)
def get_aging_step(dt_df, i_set_df, p_set_df, temp_amb_df, cap_aged_begin, aging_states, temp_cell, soc):  # deprecated
    if i_set_df is not None:
        use_power = False
        pi_set_df = i_set_df
    elif p_set_df is not None:
        use_power = True
        pi_set_df = p_set_df
    else:
        return aging_states, cap_aged_begin, temp_cell, soc, None, None, None, None  # invalid user input

    # create empty data frames for V_cell, I_cell, P_cell, T_cell, get aged resistance
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, r_cell = init_step(pi_set_df.index, cap_aged_begin)

    v_cell, i_cell = np.nan, np.nan
    for ix, pi_set in pi_set_df.items():  # fastest iteration if we need index and value and can't vectorize
        # p_actual (derived from p_set) and temp_ambient are the values applied to the cell DURING the timestep dt
        # v_cell, i_cell/i_set -> resulting values at the start of the timestep
        # temp_cell, aging states, cap_aged_end, temp_cell_end, soc_end, soe_end -> AFTER the complete simulation step

        ocv = get_ocv_from_soc(soc)  # calculate OCV from SoC (at the beginning of the timestep)
        if use_power:  # pi_set is p_set -> calculate current
            i_set = get_i_set_from_p_set(pi_set, ocv, r_cell)
        else:  # pi_set is i_set -> only check if current is valid
            i_set = get_limited_i_set(pi_set)

        soc, v_cell, p_actual, temp_cell = (  # apply electrical and thermal cell model
            cell_model(dt_df[ix], soc, ocv, i_set, temp_cell, temp_amb_df[ix], cap_aged_begin, r_cell))

        v_cell_df[ix], i_cell_df[ix], p_cell_df[ix], temp_cell_df[ix], soc_df[ix] =\
            v_cell, i_set, p_actual, temp_cell, soc  # store relevant values in Series

    # apply aging
    cap_aged_end, aging_states = apply_aging(cap_aged_begin, aging_states, dt_df.sum(), v_cell, i_cell, temp_cell)

    return cap_aged_end, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df


# charge the cell with constant current (CC) - constant voltage (CV) mode using the current limit i_lim and the voltage
#   limit v_lim, charged/discharged until the current amplitude is smaller than abs(i_cutoff). Generates timestamp
#   indexes in v/i/p/temp_cell_df from t_start until the end of the (dis)charging process. For charging, v_lim should
#   be > ocv (= get_ocv_from_soc(soc)), for discharging, v_lim should be < ocv (otherwise the function will return after
#   one step, which is ok as well). For charging, i_lim and i_cutoff should be > 0, for discharging, both should be < 0.
#   Let dt = dt_resolution. The first v/i/p/temp_cell_df.index is (t_start), the last .index is (t_start + n * dt).
#   The cell charges until (t_start + (n + 1) * dt).
#   t_start    (t_start + 1 * dt)    ...    (t_start + n * dt)    ...    (t_start + (n + 1) * dt)
#      |-- charging -------------------------------------------------------------------|
#      \-- included in v/i/p/temp_cell_df.index -----------------------/
#   For any index ix = (t_start + i * dt).
#       - i_cell_df[ix] is applied between (t_start + i * dt) and (t_start + (i + 1) * dt)
#       - v_cell_df[ix] is measured just before (t_start + (i + 1) * dt)
#       - p_cell_df[ix] is measured just before (t_start + (i + 1) * dt) and roughly equals the power in between i...i+1
#       - temp_cell_df[ix] is measured just before (t_start + (i + 1) * dt)
#       --> for accurate modeling, choose an appropriately small dt_resolution, for example 5...30 seconds for a C-rate
#           of 0.5...2.0 C or 5...120 seconds for a C-rate < 0.5 C.
# inputs:   t_start, dt_resolution, i_lim, v_lim, i_cutoff, temp_amb, cap_aged, aging_states, temp_cell, soc
#   t_start         int or float    timestamp in s (e.g., unixtimestamp), at which charging/discharging starts
#   dt_resolution   int or float    temporal resolution of the micro-steps (at which soc and aging is evaluated)
#   v_lim           float           charging/discharging "constant voltage" (CV), i.e., terminal (max./min.) voltage
#   i_lim           float           charging(+)/discharging(-) "constant current" (CC), i.e., maximum current
#   i_cutoff        float           chg.(+)/dischg.(-) cut-off current in CV phase, end if abs(current) < abs(i_cutoff)
#   temp_amb        float           ambient (or coolant) temperature in °C -> R_TH_CELL "between" temp_ambient/temp_cell
#   FIXME add description for v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df
#   cap_aged        float           remaining usable capacity of the cell in Ah at start (cap_aged of last step)
#   aging_states    arr. of floats  internal aging states (different aging types [0...1]), use aging_states of last step
#   temp_cell       float           cell (case) temperature in °C at start (use temp_cell of the last time step)
#   soc             float           State of Charge [0...1] of the cell at start (use soc of the last time step)
# outputs:  cap_aged_end, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df
#   cap_aged_end    float           remaining usable capacity of the cell in Ah after step (-> cap_aged_begin of next)
#   aging_states    arr. of floats  internal aging states, use for next step
#   temp_cell       float           new cell (case) temperature in °C (use for temp_cell of next step)
#   soc             float           new capacity-based State of Charge [0...1] of cell (use for soc of next step)
#   v_cell_df       pandas.Series   cell voltage profile in V (over the index given in dt_df - informative, not needed)
#   i_cell_df       pandas.Series   actual cell current profile in A (over dt_df.index - informative, not needed)
#   p_cell_df       pandas.Series   actual cell power profile in W (over dt_df.index - informative, not needed)
#   temp_cell_df    pandas.Series   cell (case) temperature profile in °C (over dt_df.index - informative, not needed)
#   soc_df          pandas.Series   cell State of Charge [0...1] profile (over dt_df.index - informative, not needed)
#   t_next          int or float    timestamp in s (e.g., unixtimestamp), can be used as t_start of next process
def apply_cc_cv(t_start, dt_resolution, v_lim, i_lim, i_cutoff, temp_amb, v_cell_df, i_cell_df, p_cell_df, temp_cell_df,
                soc_df, cap_aged, aging_states, temp_cell, soc, t_end_max=None):
    if cap_aged <= 0.0:  # cell has no usable capacity anymore
        return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start
    # check current
    v_lim = get_limited_v_set(v_lim)
    i_lim = get_limited_i_set(i_lim)

    # estimate how long the charging/discharging process takes and create time index range that is definitely longer
    max_duration_s = 2.0 * abs(cap_aged / i_lim) * 3600.0  # assume charging duration < 2x time for CC-chg. to 100%
    # ixs = range(int(t_start), int(t_start + math.ceil(max_duration_s + dt_resolution)), int(dt_resolution))
    t_end = t_start + max_duration_s
    if t_end_max is not None:
        if t_end > t_end_max:
            t_end = t_end_max
    # ixs = np.arange(t_start, t_end + dt_resolution, dt_resolution)
    ixs = np.arange(t_start, t_end, dt_resolution)

    # create empty data frames for V_cell, I_cell, P_cell, T_cell, get aged resistance
    v_new, i_new, p_new, temp_new, soc_new, r_cell = init_step(ixs, cap_aged)

    # temp_amb_use = temp_amb

    i_cell, ix_last_used = np.nan, t_start
    for ix in ixs:  # fastest iteration if we need the index and can't vectorize
        ocv = get_ocv_from_soc(soc)  # calculate OCV from SoC (at the beginning of the timestep)
        i_set = get_i_set_from_v_lim_i_lim(v_lim, i_lim, ocv, r_cell)
        end_run = False
        if i_lim > 0.0:
            if i_set < i_cutoff:
                end_run = True  # end of charge!
        elif i_lim < 0.0:
            if i_set > i_cutoff:
                end_run = True  # end of discharge!
        else:
            end_run = True  # end, since i_lim = 0

        if type(temp_amb) is pd.Series:
            # temp_amb_use = get_nearest_value_from_df(temp_amb, ix, temp_amb_use)  we already have temp_amb[ix]!
            temp_amb_use = temp_amb[ix]
        else:
            temp_amb_use = temp_amb

        soc, v_cell, p_actual, temp_cell = (  # apply electrical and thermal cell model
            cell_model(dt_resolution, soc, ocv, i_set, temp_cell, temp_amb_use, cap_aged, r_cell))

        v_new[ix], i_new[ix], p_new[ix], temp_new[ix], soc_new[ix] =\
            v_cell, i_set, p_actual, temp_cell, soc  # store relevant values in Series

        ix_last_used = ix
        if end_run:
            break

    # cut Series after ix_last_used
    v_new = v_new.loc[t_start:ix_last_used]  # when using .loc[a:b], b is included
    i_new = i_new.loc[t_start:ix_last_used]
    p_new = p_new.loc[t_start:ix_last_used]
    temp_new = temp_new.loc[t_start:ix_last_used]
    soc_new = soc_new.loc[t_start:ix_last_used]

    # apply aging
    cap_aged, aging_states = apply_aging_df(cap_aged, aging_states, dt_resolution, v_new, i_new, temp_new)

    if v_cell_df is None:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = v_new, i_new, p_new, temp_new, soc_new
    else:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = append_dataframes(
            [v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df], [v_new, i_new, p_new, temp_new, soc_new])

    t_next = ix_last_used + dt_resolution
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_next


# FIXME documentation
def apply_cp_cv(t_start, dt_resolution, v_lim, p_lim, i_cutoff, temp_amb, v_cell_df, i_cell_df, p_cell_df, temp_cell_df,
                soc_df, cap_aged, aging_states, temp_cell, soc, t_end_max=None):
    if cap_aged <= 0.0:  # cell has no usable capacity anymore
        return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start
    # check current
    v_lim = get_limited_v_set(v_lim)

    # estimate how long the charging/discharging process takes and create time index range that is definitely longer
    # assume charging duration < 2x time for CP-chg. to 100%
    max_duration_s = 2.0 * abs(cap_aged / (p_lim / V_NOMINAL)) * 3600.0
    # ixs = range(int(t_start), int(t_start + math.ceil(max_duration_s + dt_resolution)), int(dt_resolution))
    t_end = t_start + max_duration_s
    if t_end_max is not None:
        if t_end > t_end_max:
            t_end = t_end_max
    # ixs = np.arange(t_start, t_end + dt_resolution, dt_resolution)
    ixs = np.arange(t_start, t_end, dt_resolution)

    # create empty data frames for V_cell, I_cell, P_cell, T_cell, get aged resistance
    v_new, i_new, p_new, temp_new, soc_new, r_cell = init_step(ixs, cap_aged)

    if type(temp_amb) is pd.Series:  # todo: consider explicitly providing temp_amb type as a parameter -> safer*
        if temp_amb.shape[0] == len(ixs):  # same length -> copy ixs   (<-  *in rare scenarios, this could be a mistake)
            temp_amb = pd.Series(temp_amb.values, index=ixs)
        else:  # different length -> need interpolation (assume index is of same type)
            temp_amb = interpolate_df(temp_amb, ixs)
    else:
        temp_amb = pd.Series(temp_amb, index=ixs)

    # temp_amb_use = temp_amb

    i_cell, ix_last_used = np.nan, t_start
    for ix in ixs:  # fastest iteration if we need the index and can't vectorize
        ocv = get_ocv_from_soc(soc)  # calculate OCV from SoC (at the beginning of the timestep)
        i_set = get_i_set_from_v_lim_p_lim(v_lim, p_lim, ocv, r_cell)
        end_run = False
        if p_lim > 0.0:
            if i_set < i_cutoff:
                end_run = True  # end of charge!
        elif p_lim < 0.0:
            if i_set > i_cutoff:
                end_run = True  # end of discharge!
        else:
            end_run = True  # end, since p_lim = 0

        if type(temp_amb) is pd.Series:
            # temp_amb_use = get_nearest_value_from_df(temp_amb, ix, temp_amb_use)  we already have temp_amb[ix]!
            temp_amb_use = temp_amb[ix]
        else:
            temp_amb_use = temp_amb

        soc, v_cell, p_actual, temp_cell = (  # apply electrical and thermal cell model
            cell_model(dt_resolution, soc, ocv, i_set, temp_cell, temp_amb_use, cap_aged, r_cell))

        v_new[ix], i_new[ix], p_new[ix], temp_new[ix], soc_new[ix] =\
            v_cell, i_set, p_actual, temp_cell, soc  # store relevant values in Series

        ix_last_used = ix
        if end_run:
            break

    # cut Series after ix_last_used
    v_new = v_new.loc[t_start:ix_last_used]  # when using .loc[a:b], b is included
    i_new = i_new.loc[t_start:ix_last_used]
    p_new = p_new.loc[t_start:ix_last_used]
    temp_new = temp_new.loc[t_start:ix_last_used]
    soc_new = soc_new.loc[t_start:ix_last_used]

    # apply aging
    cap_aged, aging_states = apply_aging_df(cap_aged, aging_states, dt_resolution, v_new, i_new, temp_new)

    if v_cell_df is None:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = v_new, i_new, p_new, temp_new, soc_new
    else:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = append_dataframes(
            [v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df], [v_new, i_new, p_new, temp_new, soc_new])

    t_next = ix_last_used + dt_resolution
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_next


# FIXME documentation
def apply_power_profile(t_start, dt_resolution, p_set_df, temp_amb,
                        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc):
    if cap_aged <= 0.0:  # cell has no usable capacity anymore
        return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start

    if type(p_set_df) is pd.Series:
        p_set_df_use = p_set_df.copy()
        p_set_df_use.index = p_set_df_use.index + t_start - p_set_df_use.index[0]
    else:
        ixs = np.arange(t_start, t_start + len(p_set_df) * dt_resolution, dt_resolution)  # t_start
        p_set_df_use = pd.Series(p_set_df, index=ixs)
    # create empty data frames for V_cell, I_cell, P_cell, T_cell, get aged resistance
    v_new, i_new, p_new, temp_new, soc_new, r_cell = init_step(p_set_df_use.index, cap_aged)

    ixs = p_set_df_use.index
    if type(temp_amb) is pd.Series:  # todo: consider explicitly providing temp_amb type as a parameter -> safer*
        if temp_amb.shape[0] == len(
                ixs):  # same length -> copy ixs   (<-  *in rare scenarios, this could be a mistake)
            temp_amb = pd.Series(temp_amb.values, index=ixs)
        else:  # different length -> need interpolation (assume index is of same type)
            temp_amb = interpolate_df(temp_amb, ixs)
    else:
        temp_amb = pd.Series(temp_amb, index=ixs)

    # temp_amb_use = temp_amb

    for ix, pi_set in p_set_df_use.items():  # fastest iteration if we need index and value and can't vectorize
        ocv = get_ocv_from_soc(soc)  # calculate OCV from SoC (at the beginning of the timestep)
        i_set = get_i_set_from_p_set(pi_set, ocv, r_cell)

        if type(temp_amb) is pd.Series:
            # temp_amb_use = get_nearest_value_from_df(temp_amb, ix, temp_amb_use)  we already have temp_amb[ix]!
            temp_amb_use = temp_amb[ix]
        else:
            temp_amb_use = temp_amb

        soc, v_cell, p_actual, temp_cell = (  # apply electrical and thermal cell model
            cell_model(dt_resolution, soc, ocv, i_set, temp_cell, temp_amb_use, cap_aged, r_cell))

        v_new[ix], i_new[ix], p_new[ix], temp_new[ix], soc_new[ix] =\
            v_cell, i_set, p_actual, temp_cell, soc  # store relevant values in Series

    # apply aging
    cap_aged, aging_states = apply_aging_df(cap_aged, aging_states, dt_resolution, v_new, i_new, temp_new)

    if v_cell_df is None:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = v_new, i_new, p_new, temp_new, soc_new
    else:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = append_dataframes(
            [v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df], [v_new, i_new, p_new, temp_new, soc_new])

    t_next = v_cell_df.index[-1] + dt_resolution
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_next


# FIXME documentation
def apply_power_profile_soc_lim(t_start, dt_resolution, p_set_df, temp_amb,
                                v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
                                cap_aged, aging_states, temp_cell, soc, soc_min):
    if cap_aged <= 0.0:  # cell has no usable capacity anymore
        return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start

    if type(p_set_df) is pd.Series:
        p_set_df_use = p_set_df.copy()
        p_set_df_use.index = p_set_df_use.index + t_start - p_set_df_use.index[0]
    else:
        ixs = np.arange(t_start, t_start + len(p_set_df) * dt_resolution, dt_resolution)  # t_start
        p_set_df_use = pd.Series(p_set_df, index=ixs)
    # create empty data frames for V_cell, I_cell, P_cell, T_cell, get aged resistance
    v_new, i_new, p_new, temp_new, soc_new, r_cell = init_step(p_set_df_use.index, cap_aged)

    ixs = p_set_df_use.index
    if type(temp_amb) is pd.Series:  # todo: consider explicitly providing temp_amb type as a parameter -> safer*
        if temp_amb.shape[0] == len(
                ixs):  # same length -> copy ixs   (<-  *in rare scenarios, this could be a mistake)
            temp_amb = pd.Series(temp_amb.values, index=ixs)
        else:  # different length -> need interpolation (assume index is of same type)
            temp_amb = interpolate_df(temp_amb, ixs)
    else:
        temp_amb = pd.Series(temp_amb, index=ixs)

    # temp_amb_use = temp_amb

    for ix, pi_set in p_set_df_use.items():  # fastest iteration if we need index and value and can't vectorize
        ocv = get_ocv_from_soc(soc)  # calculate OCV from SoC (at the beginning of the timestep)
        if (soc < soc_min) and (pi_set < 0.0):  # discharging, but soc < limit => not allowed
            i_set = 0.0
        else:
            i_set = get_i_set_from_p_set(pi_set, ocv, r_cell)

        if type(temp_amb) is pd.Series:
            # temp_amb_use = get_nearest_value_from_df(temp_amb, ix, temp_amb_use)  we already have temp_amb[ix]!
            temp_amb_use = temp_amb[ix]
        else:
            temp_amb_use = temp_amb

        soc, v_cell, p_actual, temp_cell = (  # apply electrical and thermal cell model
            cell_model(dt_resolution, soc, ocv, i_set, temp_cell, temp_amb_use, cap_aged, r_cell))

        v_new[ix], i_new[ix], p_new[ix], temp_new[ix], soc_new[ix] =\
            v_cell, i_set, p_actual, temp_cell, soc  # store relevant values in Series

    # apply aging
    cap_aged, aging_states = apply_aging_df(cap_aged, aging_states, dt_resolution, v_new, i_new, temp_new)

    if v_cell_df is None:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = v_new, i_new, p_new, temp_new, soc_new
    else:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = append_dataframes(
            [v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df], [v_new, i_new, p_new, temp_new, soc_new])

    t_next = v_cell_df.index[-1] + dt_resolution
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_next


# FIXME: documentation - important: v_min is measured AFTER a profile. The profile is repeated if v_max > ocv > v_min
def apply_power_profile_repeat(t_start, dt_resolution, p_set_df, n_repeat_max, temp_amb, v_max, v_min, v_cell_df,
                               i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc):
    # avoid endless loops:
    if (((v_min is None) and (v_max is None) and (n_repeat_max is None))  # no stop condition --> endless loop
            or ((v_min is not None) and (v_min < V_CELL_MIN))  # minimum voltage will never be reached --> endless loop
            or ((v_max is not None) and (v_max > V_CELL_MAX))):  # maximum voltage will never be reached --> ...
        return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc

    n_rep = 0
    while True:
        ocv = get_ocv_from_soc(soc)
        if (v_min is not None) and (ocv <= v_min):
            break
        if (v_max is not None) and (ocv >= v_max):
            break
        if (n_repeat_max is not None) and (n_rep >= n_repeat_max):
            break
        if cap_aged <= 0.0:  # cell has no usable capacity anymore
            break
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_power_profile(t_start, dt_resolution, p_set_df, temp_amb, v_cell_df, i_cell_df, p_cell_df,
                                temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)
        n_rep = n_rep + 1

    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start, n_rep


# apply charge/discharge cycles to the cell  # FIXME add description
def apply_cycles(n_cycles_max, t_start, t_end_max,  dt_resolution_active, dt_resolution_rest, v_max, v_min,
                 i_chg, i_dischg, i_chg_cutoff, i_dischg_cutoff, rest_duration, start_charging, temp_amb,
                 v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc):
    if t_end_max is not None:
        if t_start >= t_end_max:
            return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, soc_df, cap_aged, aging_states, temp_cell, soc

    if n_cycles_max is not None:
        if n_cycles_max <= 0:
            return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, soc_df, cap_aged, aging_states, temp_cell, soc

    if cap_aged <= 0.0:  # cell has no usable capacity
        return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, soc_df, cap_aged, aging_states, temp_cell, soc

    if not start_charging:
        # discharge to v_min with i_dischg (cutoff current: i_dischg_cutoff)
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_cc_cv(t_start, dt_resolution_active, v_min, i_dischg, i_dischg_cutoff, temp_amb,
                        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

        # rest
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_pause(t_start, dt_resolution_rest, rest_duration, temp_amb,
                        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    i = 0
    # for i in range(n_cycles_max):
    while True:
        # charge to v_max with i_chg (cutoff current: i_chg_cutoff)
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_cc_cv(t_start, dt_resolution_active, v_max, i_chg, i_chg_cutoff, temp_amb,
                        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

        # rest
        v_new, i_new, p_new, temp_new, soc_new, cap_aged, aging_states, temp_cell, soc, t_start = apply_pause(
            t_start, dt_resolution_rest, rest_duration, temp_amb, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
            cap_aged, aging_states, temp_cell, soc)

        if cap_aged <= 0.0:  # cell has no usable capacity anymore
            break

        if not start_charging:
            # check end condition
            i = i + 1
            if t_end_max is not None:
                if t_start >= t_end_max:
                    break
            if n_cycles_max is not None:
                if i >= n_cycles_max:
                    break

        if (start_charging or (n_cycles_max is not None and (i < (n_cycles_max - 1)))
                or (t_end_max is not None and (t_start < t_end_max))):
            # discharge to v_min with i_dischg (cutoff current: i_dischg_cutoff)
            v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
                apply_cc_cv(t_start, dt_resolution_active, v_min, i_dischg, i_dischg_cutoff, temp_amb, v_cell_df,
                            i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

            # rest
            v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
                apply_pause(t_start, dt_resolution_rest, rest_duration, temp_amb, v_cell_df,
                            i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

            if cap_aged <= 0.0:  # cell has no usable capacity anymore
                break

        if start_charging:
            # check end condition
            i = i + 1
            if t_end_max is not None:
                if t_start >= t_end_max:
                    break
            if n_cycles_max is not None:
                if i >= n_cycles_max:
                    break

    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start


def apply_profile_cycles(n_charge_cycles_max, t_start, t_end_max,
                         dt_resolution_charging, dt_resolution_profile, dt_resolution_rest,
                         p_set_df, v_max, v_min, i_chg, i_chg_cutoff, rest_duration, start_charging, temp_amb,
                         v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc):
    if t_end_max is not None:
        if t_start >= t_end_max:
            return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, soc_df, cap_aged, aging_states, temp_cell, soc

    if n_charge_cycles_max is not None:
        if n_charge_cycles_max <= 0:
            return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, soc_df, cap_aged, aging_states, temp_cell, soc

    if cap_aged <= 0.0:  # cell has no usable capacity anymore
        return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, soc_df, cap_aged, aging_states, temp_cell, soc

    if not start_charging:
        # apply profiles until surpassing v_min
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start, _ = \
            apply_power_profile_repeat(t_start, dt_resolution_profile, p_set_df, None, temp_amb, v_max, v_min,
                                       v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
                                       cap_aged, aging_states, temp_cell, soc)

        # rest
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_pause(t_start, dt_resolution_rest, rest_duration, temp_amb, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

        if cap_aged <= 0.0:  # cell has no usable capacity anymore
            return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, soc_df, cap_aged, aging_states, temp_cell, soc

    i = 0
    # for i in range(n_charge_cycles_max):
    while True:
        # charge to v_max with i_chg (cutoff current: i_chg_cutoff)
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_cc_cv(t_start, dt_resolution_charging, v_max, i_chg, i_chg_cutoff, temp_amb, v_cell_df, i_cell_df,
                        p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

        # rest
        v_new, i_new, p_new, temp_new, soc_new, cap_aged, aging_states, temp_cell, soc, t_start = apply_pause(
            t_start, dt_resolution_rest, rest_duration, temp_amb, v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
            cap_aged, aging_states, temp_cell, soc)

        if cap_aged <= 0.0:  # cell has no usable capacity anymore
            break

        if not start_charging:
            # check end condition
            i = i + 1
            if t_end_max is not None:
                if t_start >= t_end_max:
                    break
            if n_charge_cycles_max is not None:
                if i >= n_charge_cycles_max:
                    break

        if (start_charging or (n_charge_cycles_max is not None and (i < (n_charge_cycles_max - 1)))
                or (t_end_max is not None and (t_start < t_end_max))):
            # apply profiles until surpassing v_min
            v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start, _ =\
                apply_power_profile_repeat(t_start, dt_resolution_profile, p_set_df, None, temp_amb, v_max, v_min,
                                           v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df,
                                           cap_aged, aging_states, temp_cell, soc)

            # rest
            v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
                apply_pause(t_start, dt_resolution_rest, rest_duration, temp_amb, v_cell_df, i_cell_df, p_cell_df,
                            temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

            if cap_aged <= 0.0:  # cell has no usable capacity anymore
                break

        if start_charging:
            # check end condition
            i = i + 1
            if t_end_max is not None:
                if t_start >= t_end_max:
                    break
            if n_charge_cycles_max is not None:
                if i >= n_charge_cycles_max:
                    break

    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start


def apply_checkup(t_start, dt_resolution_active, dt_resolution_rest, v_min_op, i_chg_op, i_dischg_op,
                  i_chg_cutoff_op, i_dischg_cutoff_op, temp_amb_op,
                  v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc):
    rest_duration = 5 * 60  # 5 minutes in s
    temp_change_duration = 45 * 60  # 45 minutes in s
    temp_wait_duration = 45 * 60  # 45 minutes in s
    v_prepare_set = 2.6  # V
    v_max_cu = 4.2  # V
    v_min_cu = 2.5  # V
    i_chg_cu = 1.0  # A
    i_dischg_cu = -1.0  # A
    i_chg_cu_cutoff = 0.15  # A
    i_dischg_cu_cutoff = -0.15  # A
    temp_amb_cu = 25  # in °C
    soc_eis = [0.1, 0.3, 0.5, 0.7, 0.9]  # * 100%, nominal SoC for EIS and pulse
    duration_eis = 30 * 60  # 30 minutes in s, EIS/pulse not modeled - cell rests for this period at the given SoCs

    # change temperature from OT to RT
    n_temp_change = math.ceil(temp_change_duration / dt_resolution_rest)
    temp_amb_df = pd.Series(np.linspace(temp_amb_op, temp_amb_cu, num=n_temp_change, endpoint=False))
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        apply_pause(t_start, dt_resolution_rest, temp_change_duration, temp_amb_df, v_cell_df, i_cell_df, p_cell_df,
                    temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # wait for temperature stabilization
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        apply_pause(t_start, dt_resolution_rest, temp_wait_duration, temp_amb_cu, v_cell_df, i_cell_df, p_cell_df,
                    temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # PREPARE: discharge to v_prepare_set with operational current
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        apply_cc_cv(t_start, dt_resolution_active, v_prepare_set, i_dischg_op, i_dischg_cutoff_op, temp_amb_cu,
                    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # PREPARE: rest 5 minutes
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        apply_pause(t_start, dt_resolution_rest, rest_duration, temp_amb_cu, v_cell_df, i_cell_df, p_cell_df,
                    temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # PREPARE: discharge to v_min_cu with check-up current
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        apply_cc_cv(t_start, dt_resolution_active, v_min_cu, i_dischg_cu, i_dischg_cu_cutoff, temp_amb_cu, v_cell_df,
                    i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # PREPARE: rest 5 minutes
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        apply_pause(t_start, dt_resolution_rest, rest_duration, temp_amb_cu, v_cell_df, i_cell_df, p_cell_df,
                    temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # capacity check
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        apply_cycles(1, t_start, None,  dt_resolution_active, dt_resolution_rest, v_max_cu, v_min_cu,
                     i_chg_cu, i_dischg_cu, i_chg_cu_cutoff, i_dischg_cu_cutoff, rest_duration, True, temp_amb_cu,
                     v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # charge to EIS points and wait
    for i_soc in range(len(soc_eis)):
        # charge to soc_target
        ocv_target = get_ocv_from_soc(soc_eis[i_soc])
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_cc_cv(t_start, dt_resolution_active, ocv_target, i_chg_cu, i_chg_cu_cutoff, temp_amb_cu, v_cell_df,
                        i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

        # rest
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_pause(t_start, dt_resolution_rest, rest_duration + duration_eis, temp_amb_cu, v_cell_df, i_cell_df,
                        p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # change temperature from RT to OT
    n_temp_change = math.ceil(temp_change_duration / dt_resolution_rest)
    temp_amb_df = pd.Series(np.linspace(temp_amb_cu, temp_amb_op, num=n_temp_change, endpoint=False))
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_pause(t_start, dt_resolution_rest, temp_change_duration, temp_amb_df, v_cell_df, i_cell_df, p_cell_df,
                        temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # wait for temperature stabilization
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        apply_pause(t_start, dt_resolution_rest, temp_wait_duration, temp_amb_op, v_cell_df, i_cell_df, p_cell_df,
                    temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # discharge to EIS points and wait
    for i_soc in range(len(soc_eis) - 1, -1, -1):
        # discharge to soc_target
        ocv_target = get_ocv_from_soc(soc_eis[i_soc])
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_cc_cv(t_start, dt_resolution_active, ocv_target, i_dischg_cu, i_dischg_cu_cutoff, temp_amb_op,
                        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

        # rest
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
            apply_pause(t_start, dt_resolution_rest, rest_duration + duration_eis, temp_amb_op, v_cell_df, i_cell_df,
                        p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    # FOLLOW-UP: charge to v_min_op with operational current
    v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start = \
        apply_cc_cv(t_start, dt_resolution_active, v_min_op, i_chg_op, i_chg_cutoff_op, temp_amb_op, v_cell_df,
                    i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc)

    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start


# Let the cell rest for duration seconds at the current SoC. Generates timestamp indexes in v/i/p/temp_cell_df from
#   t_start until t_last (= t_start + duration - tau).
#   If duration is a multiple of dt_resolution, tau is dt_resolution. If not tau is between 0 and dt_resolution, i.e.,
#   the last timestep is shorter than dt_resolution, so the next index after the pandas.Series generated by this
#   function can start at (t_start + duration).
#   t_start    (t_start + 1 * dt)    ...    (t_start + n * dt)  (t_start + duration)
#      |-- resting ------------------------------------------------------|
#      \-- included in v/i/p/temp_cell_df.index --------------/
#   For any index ix = (t_start + i * dt).
#       - i_cell_df[ix] = 0 is applied between (t_start + i * dt) and the next index, typically (t_start + (i + 1) * dt)
#       - v_cell_df[ix] is measured just before the next index
#       - p_cell_df[ix] is measured just before the next index ix_n and roughly equals the power in between ix and ix_n
#       - temp_cell_df[ix] is measured just before the next index
#       --> An appropriate dt_resolution is 1...24 hours (*3600 for seconds!) while resting if the cell temperature is
#           already in steady state near temp_amb, 1...10 minutes (*60 for seconds!) might be preferable if a
#           temperature change (from temp_cell to temp_amb) is expected. Smaller is possible (e.g., 1, 10, 30 seconds).
# inputs:   t_start, dt_resolution, duration, temp_amb, cap_aged, aging_states, temp_cell, soc
#   t_start         int or float    timestamp in s (e.g., unixtimestamp), at which charging/discharging starts
#   dt_resolution   int or float    temporal resolution of the micro-steps (at which soc and aging is evaluated)
#   duration        int or float    duration in seconds, for which the cell shall rest
#   temp_amb        float           ambient (or coolant) temperature in °C -> R_TH_CELL "between" temp_ambient/temp_cell
#   FIXME add description for v_cell_df,
#   cap_aged        float           remaining usable capacity of the cell in Ah at start (cap_aged of last step)
#   aging_states    arr. of floats  internal aging states (different aging types [0...1]), use aging_states of last step
#   temp_cell       float           cell (case) temperature in °C at start (use temp_cell of the last time step)
#   soc             float           State of Charge [0...1] of the cell at start (use soc of the last time step)
# outputs:  cap_aged_end, aging_states, temp_cell, soc, v_cell_df, i_cell_df, p_cell_df, temp_cell_df
#   cap_aged_end    float           remaining usable capacity of the cell in Ah after step (-> cap_aged_begin of next)
#   aging_states    arr. of floats  internal aging states, use for next step
#   temp_cell       float           new cell (case) temperature in °C (use for temp_cell of next step)
#   soc             float           new capacity-based State of Charge [0...1] of cell (use for soc of next step)
#   v_cell_df       pandas.Series   cell voltage profile in V (= OCV over dt_df.index - informative, not needed)
#   i_cell_df       pandas.Series   actual cell current profile in A (= 0 over dt_df.index - informative, not needed)
#   p_cell_df       pandas.Series   actual cell power profile in W (= 0 over dt_df.index - informative, not needed)
#   temp_cell_df    pandas.Series   cell (case) temperature profile in °C (over dt_df.index - informative, not needed)
#   soc_df          pandas.Series   cell State of Charge [0...1] profile (over dt_df.index - informative, not needed)
#   t_next          int or float    timestamp in s (e.g., unixtimestamp), can be used as t_start of next process
def apply_pause(t_start, dt_resolution, duration, temp_amb: typing.Union[int, float, pd.Series],
                v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc):
    if duration <= 0:
        return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_start

    n = math.ceil((duration - dt_resolution) / dt_resolution)
    t_last = t_start + n * dt_resolution
    # ixs = range(int(t_start), int(t_start + duration), int(dt_resolution))
    ixs = np.arange(t_start, t_start + duration, dt_resolution)
    dt = dt_resolution

    if type(temp_amb) is pd.Series:  # todo: consider explicitly providing temp_amb type as a parameter -> safer*
        if temp_amb.shape[0] == len(ixs):  # same length -> copy ixs   (<-  *in rare scenarios, this could be a mistake)
            temp_amb = pd.Series(temp_amb.values, index=ixs)
        else:  # different length -> need interpolation (assume index is of same type)
            temp_amb = interpolate_df(temp_amb, ixs)
    else:
        temp_amb = pd.Series(temp_amb, index=ixs)

    # temp_amb_use = temp_amb

    # create pre-filled data frames for V_cell, I_cell, P_cell, T_cell, get ocv
    v_new, i_new, p_new, temp_new, soc_new, v_cell = init_step_rest(ixs, soc)
    for ix in ixs:  # fastest iteration if we need the index and can't vectorize
        if ix == t_last:
            dt = t_start + duration - t_last
        else:
            dt = dt_resolution

        if type(temp_amb) is pd.Series:
            # temp_amb_use = get_nearest_value_from_df(temp_amb, ix, temp_amb_use)  we already have temp_amb[ix]!
            temp_amb_use = temp_amb[ix]
        else:
            temp_amb_use = temp_amb

        # apply thermal cell model, store temperature in Series
        temp_cell = cell_model_rest(dt, temp_cell, temp_amb_use)
        temp_new[ix] = temp_cell

        # # calculate aging for every timestep
        # cap_aged_end, aging_states = apply_aging(cap_aged, aging_states, dt_resolution, v_cell, 0.0, temp_cell)

    # apply aging
    cap_aged, aging_states = apply_aging_df(cap_aged, aging_states, dt_resolution, v_new, i_new, temp_new)

    if v_cell_df is None:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = v_new, i_new, p_new, temp_new, soc_new
    else:
        v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df = append_dataframes(
            [v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df], [v_new, i_new, p_new, temp_new, soc_new])

    t_next = v_cell_df.index[-1] + dt
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, cap_aged, aging_states, temp_cell, soc, t_next


def get_i_set_from_p_set(p_set, ocv, r_cell):
    # calculate set-point current for p_set (at the beginning of the timestep)
    i_set = (-ocv + math.sqrt(ocv**2 + 4.0 * r_cell * p_set)) / (2.0 * r_cell)

    # calculate I_max_chg for V_CELL_MAX and I_min_dischg for V_CELL_MIN, limit i_cell if necessary
    if p_set > 0.0:  # power > 0 --> charging
        if i_set < 0.0:
            i_set = 0.0
        else:
            i_max_chg = min((V_CELL_MAX - ocv) / r_cell, I_CELL_MAX_CHG)
            if i_set > i_max_chg:
                if i_max_chg < 0.0:  # don't allow discharging when the cell should be charging
                    i_set = 0.0
                else:
                    i_set = i_max_chg
    elif p_set < 0.0:  # power < 0 --> discharging
        if i_set > 0.0:
            i_set = 0.0
        else:
            i_min_dischg = max((V_CELL_MIN - ocv) / r_cell, I_CELL_MIN_DISCHG)
            if i_set < i_min_dischg:
                if i_min_dischg > 0.0:  # don't allow charging when the cell should be discharging
                    i_set = 0.0
                else:
                    i_set = i_min_dischg
    else:
        i_set = 0.0
    return i_set


def get_i_set_from_v_lim_i_lim(v_lim, i_lim, ocv, r_cell):
    # calculate set-point current for CC/CV charging
    i_set = i_lim
    if i_lim > 0.0:  # charging
        if ocv >= v_lim:  # reached charging level
            i_set = 0.0
        else:
            i_v_lim = (v_lim - ocv) / r_cell
            if i_lim > i_v_lim:
                i_set = i_v_lim
            # if i_set > I_CELL_MAX_CHG:  --> should already be checked with get_limited_i_set() before
            #     i_set = I_CELL_MAX_CHG
    elif i_lim < 0.0:  # discharging
        if ocv <= v_lim:  # reached discharging level
            i_set = 0.0
        else:
            i_v_lim = (v_lim - ocv) / r_cell
            if i_lim < i_v_lim:
                i_set = i_v_lim
            # if i_set < I_CELL_MIN_DISCHG:  --> should already be checked with get_limited_i_set() before
            #     i_set = I_CELL_MIN_DISCHG
    else:
        i_set = 0.0  # i_lim = 0 or invalid

    return i_set


def get_i_set_from_v_lim_p_lim(v_lim, p_lim, ocv, r_cell):
    # calculate set-point current for p_set (at the beginning of the timestep)
    i_p_lim = (-ocv + math.sqrt(ocv**2 + 4.0 * r_cell * p_lim)) / (2.0 * r_cell)
    i_v_lim = (v_lim - ocv) / r_cell

    if i_p_lim > 0.0:
        if i_p_lim > i_v_lim:
            i_set = i_v_lim
        else:
            i_set = i_p_lim
    else:
        if i_p_lim < i_v_lim:
            i_set = i_v_lim
        else:
            i_set = i_p_lim

    # calculate I_max_chg for V_CELL_MAX and I_min_dischg for V_CELL_MIN, limit i_cell if necessary
    if p_lim > 0.0:  # power > 0 --> charging
        if i_p_lim < 0.0:
            i_set = 0.0
        else:
            i_max_chg = min((min(V_CELL_MAX, v_lim) - ocv) / r_cell, I_CELL_MAX_CHG)
            if i_p_lim > i_max_chg:
                if i_max_chg < 0.0:  # don't allow discharging when the cell should be charging
                    i_set = 0.0
                else:
                    i_set = i_max_chg
    elif p_lim < 0.0:  # power < 0 --> discharging
        if i_p_lim > 0.0:
            i_set = 0.0
        else:
            i_min_dischg = max((V_CELL_MIN - ocv) / r_cell, I_CELL_MIN_DISCHG)
            if i_p_lim < i_min_dischg:
                if i_min_dischg > 0.0:  # don't allow charging when the cell should be discharging
                    i_set = 0.0
                else:
                    i_set = i_min_dischg
    else:
        i_set = 0.0
    return i_set


def get_limited_i_set(i_set):
    if i_set > 0.0:
        if i_set > I_CELL_MAX_CHG:
            i_set = I_CELL_MAX_CHG
    elif i_set < 0.0:
        if i_set < I_CELL_MIN_DISCHG:
            i_set = I_CELL_MIN_DISCHG
    return i_set


def get_limited_v_set(v_set):
    if v_set > V_CELL_MAX:
        v_set = V_CELL_MAX
    elif v_set < V_CELL_MIN:
        v_set = V_CELL_MIN
    return v_set


def cell_model(dt, soc, ocv, i_set, temp_cell, temp_ambient, cap_aged, r_cell):
    # calculate new cell voltage, actual cell power and current (at the beginning of the timestep)
    dv_cell = r_cell * i_set
    v_cell = ocv + dv_cell
    p_actual = v_cell * i_set

    # calculate new SoC (at the end of the timestamp)
    if cap_aged > 0.0:
        soc = soc + i_set * dt / (cap_aged * 3600.0)  # soc in %, i-set in A, dt in s, cap_aged_begin in Ah

    # calculate thermal losses during the timestep and new cell temperature after the timestep
    p_loss = dv_cell * i_set  # R * I^2, but performance-optimized
    temp_cell = temp_cell + ((temp_ambient + R_TH_CELL * p_loss - temp_cell) / R_TH_C_TH_CELL) * dt
    return soc, v_cell, p_actual, temp_cell


# with i = 0 / p = 0 -> cell resting, temperature relaxing
def cell_model_rest(dt, temp_cell, temp_ambient):
    # calculate new cell temperature after the timestep
    # temp_cell = temp_cell + ((temp_ambient - temp_cell) / R_TH_C_TH_CELL) * dt
    # return temp_cell
    return temp_cell + ((temp_ambient - temp_cell) / R_TH_C_TH_CELL) * dt


def init_step(ixs, cap_aged):
    v_cell_df = pd.Series(0, index=ixs)
    i_cell_df = v_cell_df.copy()
    p_cell_df = v_cell_df.copy()
    temp_cell_df = v_cell_df.copy()
    soc_df = v_cell_df.copy()

    r_cell = get_r_cell_from_cap_aged(cap_aged)
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, r_cell


def init_step_rest(ixs, soc):
    ocv = get_ocv_from_soc(soc)  # calculate OCV from SoC (at the beginning of the timestep - stays constant)
    soc_df = pd.Series(soc, index=ixs)
    v_cell_df = pd.Series(ocv, index=ixs)
    i_cell_df = pd.Series(0.0, index=ixs)
    p_cell_df = i_cell_df.copy()
    temp_cell_df = i_cell_df.copy()
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df, ocv


# FIXME documentation
def apply_aging_df(cap_aged, aging_states, dt_resolution, v_cell_df, i_cell_df, temp_cell_df):
    # resample to AGE_APPLY_PERIOD to apply aging
    v_cell = v_cell_df.copy()  # not sure if this is necessary, but we don't want to alter the original data
    i_cell = i_cell_df.copy()
    temp_cell = temp_cell_df.copy()

    # convert timestamp to datetime to apply stuff
    try:
        v_cell.index = pd.to_datetime(v_cell.index, unit="s", origin='unix')
    except pd.errors.ParserError:  # Exception:
        print("ParserError converting %s .. %s" % (str(v_cell.index[0]), str(v_cell.index[-1])))
    except ValueError:  # Exception:
        print("ValueError converting %s .. %s" % (str(v_cell.index[0]), str(v_cell.index[-1])))
    i_cell.index = v_cell.index
    temp_cell.index = v_cell.index
    time_sum = pd.Series(dt_resolution, index=v_cell.index)

    # resample with time_resolution (use averaging)
    pd_resolution = f'%uS' % AGE_APPLY_PERIOD
    v_cell = v_cell.resample(pd_resolution).mean()
    i_cell = i_cell.resample(pd_resolution).mean()
    temp_cell = temp_cell.resample(pd_resolution).mean()
    time_sum = time_sum.resample(pd_resolution).sum()

    # if dt_resolution > AGE_APPLY_PERIOD:
    #     # -> fill with interpolation
    #     v_cell = v_cell.interpolate("linear")
    #     i_cell = i_cell.interpolate("linear")
    #     temp_cell = temp_cell.interpolate("linear")
    #     time_sum = time_sum.interpolate("linear")

    # v_cell.index = (v_cell.index - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta("1s")
    v_cell.index = (v_cell.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")  # we did not introduce timezones
    i_cell.index = v_cell.index
    temp_cell.index = v_cell.index
    time_sum.index = v_cell.index

    for ix in v_cell.index:
        # calculate aging for every timestep
        if time_sum[ix] == 0:
            continue  # dt_resolution > AGE_APPLY_PERIOD -->
        cap_aged, aging_states = apply_aging(cap_aged, aging_states, time_sum[ix],
                                             v_cell[ix], i_cell[ix], temp_cell[ix])

    return cap_aged, aging_states


# update aging for the timestep dt [s] during which the cell voltage was v_cell [V], the cell current i_cell [A] and the
# cell temperature temp_cell [°C]
def apply_aging(cap_aged_begin, aging_states, dt, v_cell, i_cell, temp_cell):
    if np.isnan(dt) or np.isnan(v_cell) or np.isnan(i_cell) or np.isnan(temp_cell) or (cap_aged_begin <= 0.0):
        return cap_aged_begin, aging_states  # invalid input or already at 0 Ah
    # else:

    temp_cell_kelvin = temp_cell + T_0_DEGC_IN_K
    dq_loss_step = 0.0
    (q_loss_sei_total, q_loss_cyclic_total, q_loss_cyclic_low_total, q_loss_plating_total,
     Q_chg_total, Q_dischg_total, E_chg_total, E_dischg_total) = aging_states
    q_loss_total = q_loss_sei_total + q_loss_cyclic_total + q_loss_cyclic_low_total + q_loss_plating_total

    # SEI layer growth
    sei_force = (AGE_S0 * np.exp(AGE_S1 * (1.0 / temp_cell_kelvin - 1.0 / T_REF_KELVIN))
                 * np.exp(AGE_S2 * (v_cell - V_REF)))
    diff_sei = sei_force - AGE_S3 * q_loss_sei_total  # = loss rate (per second)
    if diff_sei > 0.0:  # SEI losses can only be increased, not decreased in this model
        dq_loss_sei = diff_sei * dt  # loss rate (per second) * time
        q_loss_sei_total = min(q_loss_sei_total + dq_loss_sei, 1.0)  # relative losses cannot be > 1.0
        dq_loss_step = dq_loss_step + dq_loss_sei

    if i_cell != 0.0:
        # capacity and energy throughput statistics
        dQ = i_cell * dt / 3600.0  # A*s in Ah
        dE = dQ * v_cell
        if i_cell > 0.0:  # charging
            Q_chg_total = Q_chg_total + dQ
            E_chg_total = E_chg_total + dE
        elif i_cell < 0.0:  # discharging
            Q_dischg_total = Q_dischg_total - dQ
            E_dischg_total = E_dischg_total - dE

        # cyclic aging
        c_rate_rel = i_cell / cap_aged_begin  # i_cell in A, cap_aged_begin in Ah -> c_rate_rel in 1/h
        dq_abs = abs(c_rate_rel) * dt
        # Note that if C_remaining = C_nominal, c_rate_rel is 1 [unit: 1/h] for a C-rate of 1 C, and dt is in seconds.
        # That means that the cumulated dq_abs would roughly be 3600 (no unit) for a full charge.

        # cyclic wearout
        cyc_age_force = AGE_W0 * abs(v_cell - V_REF)  # * np.exp(AGE_W1 * max(q_loss_total - AGE_W2), 0)
        if q_loss_total > AGE_W2:
            cyc_age_force = cyc_age_force * np.exp(AGE_W1 * (q_loss_total - AGE_W2))
        diff_cyc = cyc_age_force - AGE_W3 * q_loss_cyclic_total  # = loss rate (per charge in C-rate * s)
        if diff_cyc > 0.0:  # wearout losses can only be increased, not decreased in this model
            dq_loss_cyc = diff_cyc * dq_abs  # loss rate (per charge) * charge (C-rate * s)
            q_loss_cyclic_total = min(q_loss_cyclic_total + dq_loss_cyc, 1.0)  # relative losses cannot be > 1.0
            dq_loss_step = dq_loss_step + dq_loss_cyc

        # extra wearout at low voltages
        if v_cell < AGE_C2:
            low_age_force = AGE_C0 * np.exp(AGE_C1 * (1.0 / temp_cell_kelvin - 1.0 / T_REF_KELVIN)) * (AGE_C2 - v_cell)
            diff_low = low_age_force - AGE_C3 * q_loss_cyclic_low_total  # = loss rate (per charge in C-rate * s)
            if diff_low > 0.0:  # wearout losses at low voltages can only be increased, not decreased in this model
                dq_loss_low = diff_low * dq_abs  # loss rate (per charge) * charge (C-rate * s)
                q_loss_cyclic_low_total = min(q_loss_cyclic_low_total + dq_loss_low, 1.0)  # rel. losses cannot be > 1.0
                dq_loss_step = dq_loss_step + dq_loss_low

        if c_rate_rel > 0.0:  # charging: c_rate_rel = c_chg_rate_rel
            # lithium plating -> only when charging
            r_eff = AGE_P0
            if temp_cell_kelvin < AGE_P2:
                r_eff = r_eff + (AGE_P1 * (AGE_P2 - temp_cell_kelvin))**AGE_P3
            r_eff = r_eff * np.exp(AGE_P4 * q_loss_total)
            v_anode = get_v_anode_from_v_cell(v_cell)
            v_plating = v_anode - r_eff * c_rate_rel  # here: c_rate_rel = c_chg_rate_rel
            if v_plating < 0.0:
                # plating can occur. lithium stripping and intercalation of plated lithium is not modeled
                dq_plating = abs(v_plating) * AGE_P5 * dt * c_rate_rel**AGE_P6
                q_loss_plating_total = min(q_loss_plating_total + dq_plating, 1.0)  # relative losses cannot be > 1.0
                dq_loss_step = dq_loss_step + dq_plating

    dQ_loss_step = dq_loss_step * CAP_NOMINAL  # relative -> absolute losses in this timestep
    cap_aged_end = max(cap_aged_begin - dQ_loss_step, 0.0)  # capacity cannot be < 0 Ah
    aging_states = [q_loss_sei_total, q_loss_cyclic_total, q_loss_cyclic_low_total, q_loss_plating_total,
                    Q_chg_total, Q_dischg_total, E_chg_total, E_dischg_total]
    return cap_aged_end, aging_states


def get_r_cell_from_cap_aged(cap_aged):
    return R_CELL_0 + R_CELL_AGE * (1.0 - cap_aged / CAP_NOMINAL)


def check_plausibility():
    # ToDo: check plausibility of all dataframes? OCV, SoC, ... in valid range? are there steps?

    pass


# return open circuit voltage [2.5...4.2] based on cell SoC [0...1] - function primarily used internally
# soc_df and return value are pandas.Series
def get_ocv_from_soc_df(soc_df):
    # see "SoC OCV Curve at C_div_20 (0.15 A) for Python.xlsx"
    lin_a = 3.3
    lin_b = 0.9
    low_ths = 0.1326
    low_fac = 0.02
    low_exp = 28.0
    high_mid = 0.935
    high_delta = 0.065
    high_amp = 0.03

    v_corr_low = pd.Series(0, index=soc_df.index)
    v_corr_high = v_corr_low.copy()
    cond_low = (soc_df < low_ths)
    v_corr_low[cond_low] = low_fac * (1.0 - np.exp(low_exp * (low_ths - soc_df)))
    cond_high = (soc_df > (high_mid - high_delta))
    v_corr_high[cond_high] = high_amp * (((soc_df - high_mid) / high_delta)**2 - 1.0)
    return lin_a + lin_b * soc_df + v_corr_low + v_corr_high  # v_ocv


# return open circuit voltage [2.5...4.2] based on cell SoC [0...1] - approximation function primarily used internally.
# soc and return value are floats. Also gives reasonable values for [-0.02...1.05] -> [1.85..4.31 V].
def get_ocv_from_soc(soc):
    # see "SoC OCV Curve at C_div_20 (0.15 A) for Python.xlsx"
    lin_a = 3.3  # a
    lin_b = 0.9  # b
    low_ths = 0.1326
    v_ocv = lin_a + lin_b * soc
    if soc < low_ths:
        low_fac = 0.02  # d
        low_exp = 28.0  # e
        v_ocv = v_ocv + low_fac * (1.0 - np.exp(low_exp * (low_ths - soc)))
    else:
        high_mid = 0.935  # f
        high_delta = 0.065  # g
        if soc > (high_mid - high_delta):
            high_amp = 0.03  # h
            v_ocv = v_ocv + high_amp * (((soc - high_mid) / high_delta)**2 - 1.0)
    return v_ocv


# return cell SoC [0...1] based on open circuit voltage [2.5...4.2] - approximation function
# ocv and return value are floats
def get_soc_from_ocv(ocv):
    # see "SoC OCV Curve at C_div_20 (0.15 A) for Python.xlsx"
    lin_a = -(11.0/3.0)  # -3.666666667
    lin_b = (10.0/9.0)  # 1.111111111
    low_ths = 3.43
    if ocv < low_ths:
        d = 0.144444444
        e = 0.1493
        f = 0.45
        return d - e * (low_ths - ocv)**f
    else:
        high_ths = 4.083
        if ocv > high_ths:
            A = 7.100591716
            B = -12.37810651
            C = 9.477514793
            return (-B + math.sqrt(B * B - 4 * A * (C - ocv))) / (2 * A)
        else:
            return lin_a + lin_b * ocv


# return energy-related cell State of Energy (soe) [0..1] based on capacity-related cell SoC [0...1].
# soc and return value can be both floats or both pandas.Series. Also works beyond [0..1], although not very meaningful.
def get_soe_from_soc(soc):
    # see "SoC OCV Curve at C_div_20 (0.15 A) for Python.xlsx" - tested with different max. powers from 0.1 to 22.0 W
    a = 0.12
    b = 0.5
    c = -(a * b**2)
    soe = soc + a * (soc - b)**2 + c
    return soe


# return estimated anode potential [V] based on cell voltage [V] (this is not calibrated but more of a guess!)
# v_cell_df and return value are pandas.Series
def get_v_anode_from_v_cell(v_cell):
    V1 = 4.095
    V2 = 3.83
    V3 = 3.65
    V4 = 3.5
    C1 = 2.5
    M1 = 0.59047619
    C2 = 0.082
    C3 = 0.890555556
    M3 = 0.211111111
    C4 = 0.12
    C5 = 2.15
    M5 = 0.58
    if type(v_cell) is pd.Series:
        cond_a1 = (v_cell > V1)
        cond_a2 = ((v_cell > V2) & (v_cell <= V1))
        cond_a3 = ((v_cell > V3) & (v_cell <= V2))
        cond_a4 = ((v_cell > V4) & (v_cell <= V3))
        cond_a5 = (v_cell <= V4)
        v_anode = pd.Series(data=0.0, dtype=np.float64, index=v_cell.index)
        v_anode[cond_a1] = C1 - M1 * v_cell[cond_a1]
        v_anode[cond_a2] = C2
        v_anode[cond_a3] = C3 - M3 * v_cell[cond_a3]
        v_anode[cond_a4] = C4
        v_anode[cond_a5] = C5 - M5 * v_cell[cond_a5]  # estimation not good for SoC < 25%
    else:
        if v_cell > V2:
            if v_cell > V1:
                v_anode = C1 - M1 * v_cell
            else:
                v_anode = C2
        elif v_cell > V4:
            if v_cell > V3:
                v_anode = C3 - M3 * v_cell
            else:
                v_anode = C4
        else:
            v_anode = C5 - M5 * v_cell
    return v_anode


# append / merge / stitch together a list of data frames
def append_dataframes(df_array, df_new_array):
    for i in range(len(df_array)):
        df_array[i] = pd.concat([df_array[i], df_new_array[i]]).copy()
    return df_array


def interpolate_df(source_df: pd.Series, target_ixs):
    # # this could probably be optimized, but I don't have time now ...
    # source_df.name = "data"
    # source_df = source_df.astype(float)
    # nan_df = pd.Series(np.nan, index=target_ixs, name="data")
    # target_df = pd.merge(source_df, nan_df, how="outer", left_index=True, right_index=True)
    # target_df.drop(columns="data_y", inplace=True)
    # # target_df.rename(columns={"data_x": ""}, inplace=True)
    # target_df = target_df["data_x"]
    # # target_df = target_df.interpolate().ffill().bfill()  only works as intended when the index is spaced uniformly
    # target_df = target_df.interpolate(method="index").ffill().bfill()
    # target_df = target_df[target_df.index.isin(target_ixs)]

    all_ixs = source_df.index.union(target_ixs)  # FIXME check if it works as intended
    target_df = source_df.reindex(all_ixs).sort_index().interpolate(method="index").ffill().bfill()
    target_df = target_df[target_df.index.isin(target_ixs)]
    return target_df


def get_nearest_value_from_df(df, ix, fallback_value):
    # if ix in df.index:
    #     return df[ix]
    # return df.iloc[df.index.get_loc(ix, method='backfill')]
    if df.shape[0] > 0:
        match_ixs = df.index.get_indexer([ix], method='nearest')
        if match_ixs.shape[0] > 0:
            return df.iloc[match_ixs[0]]
    return fallback_value


# initialize variables to be used together with the battery aging model. ToDo: has to be called before using the model!
def init(capacity_initial=CAP_INITIAL,
         storage_time_days=STORAGE_TIME_DEFAULT,
         storage_soc=STORAGE_SOC_DEFAULT,
         storage_temperature=STORAGE_TEMPERATURE_DEFAULT
         ):
    soc_begin = storage_soc
    temp_cell_begin = storage_temperature  # assume the cell is and remains in steady state at this temperature
    cap_aged_begin = capacity_initial

    q_loss_sei_total = 0  # FIXME description
    q_loss_cyclic_total = 0
    q_loss_cyclic_low_total = 0
    q_loss_plating_total = 0
    Q_chg_total = 0
    Q_dischg_total = 0
    E_chg_total = 0
    E_dischg_total = 0
    aging_states = [q_loss_sei_total, q_loss_cyclic_total, q_loss_cyclic_low_total, q_loss_plating_total,
                    Q_chg_total, Q_dischg_total, E_chg_total, E_dischg_total]

    v_cell = get_ocv_from_soc(soc_begin)
    for i in range(storage_time_days):  # apply SEI aging to the cell for storage_time_days * 24 * 3600 seconds
        cap_aged_begin, aging_states = apply_aging(
            cap_aged_begin, aging_states, (24 * 60 * 60), v_cell, 0.0, temp_cell_begin)

    return cap_aged_begin, aging_states, temp_cell_begin, soc_begin


def init_empty_df():
    data_type = float
    v_cell_df = pd.Series(dtype=data_type)
    i_cell_df = pd.Series(dtype=data_type)
    p_cell_df = pd.Series(dtype=data_type)
    temp_cell_df = pd.Series(dtype=data_type)
    soc_df = pd.Series(dtype=data_type)
    return v_cell_df, i_cell_df, p_cell_df, temp_cell_df, soc_df
