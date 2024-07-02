# helper functions and definitions for the scenarios in use_case_model_EV_modular_v01.py
# feel free to implement additional charging strategies, lcations, or parameters

import datetime
import numpy as np
from enum import IntEnum

import bat_model_v01 as bat


# charging strategies
class CHG_STRAT(IntEnum):
    # no charging:
    NONE = 0,               # no charging (not possible / not used) -> leave this at 0 and make all others larger!
    # conventional charging:
    EARLY = 1,              # early ("stupid") charging -> charge as soon as arriving (high average SoC)
    EARLY_IF_LOW = 2,       # early charging, but only if battery below SOC_LOW threshold
    # scheduled charging
    LATE = 3,               # late ("scheduled") charging -> charge before departing (lower average SoC)
    LATE_IF_LOW = 4,        # late charging, but only if battery below SOC_LOW threshold
    # optimized ("smart") charging:
    V1G_OPT_EMISSION = 10,  # charging while specific emissions of electricity mix are low
    V1G_OPT_COST = 11,      # charging while cost of electricity is low
    V1G_OPT_REN = 12,       # charging while excess renewable energy is available
    # bidirectional charging (V2G):
    V2G_OPT_EMISSION = 20,  # emission-optimized V2G (dis)charging
    V2G_OPT_COST = 21,      # electricity-price-optimized V2G (dis)charging
    V2G_OPT_REN = 22,       # renewable-energy-optimized V2G (dis)charging
    V2G_OPT_FREQ = 23,      # frequency control
    V2G_OPT_PV = 24,        # optimize PV self-consumption (fallback: LATE_IF_LOW)
    # V2G_OPT_BAT = 25,       # battery-optimized V2G (dis)charging - ToDO: implement? or use late charging 40-60%?
    # ... (you may implement others)


# scenario list item names
ID = "ID"  # unique, numeric ID of the scenario
SIM_START = "SIM_START"
SIM_STOP = "SIM_STOP"
# SIM_YEARS = "SIM_YEARS"
HOME = "HOME"  # parameters for any day at home
WORK = "WORK"  # parameters for work days / at work
FREE = "FREE"  # parameters for free days (leisure / shopping activity)
TRIP = "TRIP"  # parameters for trip days (long-distance one-way trip)
LOCATION_ARRAY = [HOME, WORK, FREE, TRIP]  # all locations
CHG_STRATEGY = "CHG_STRATEGY"  # charging strategy, one of CHG_STRAT
CHG_P = "CHG_P"  # charging power -> in kW (EV level), e.g., 11 for 11 kW or 100 for 100 kW
CHG_V_LIM = "CHG_V_LIM"  # charging voltage limit -> in V (cell level), [bat.V_CELL_MIN, bat.V_CELL_MAX] = [2.5, 4.2]
CHG_I_CO = "CHG_I_CO"  # charging (or discharging) cut-off current -> in A (cell level), >0
CHG_SOC_LOW = "SOC_LOW"  # lower SoC limit (charge if below) -> [0, 1]
DEPARTURE = "DEPARTURE"  # departure time (range) -> float, int, or array(2) of float or int
DURATION = "DURATION"  # duration of stay -> float, int, or array(2) of float or int
SHIFT_BY_YEARS = "SHIFT_BY_YEARS"  # if > 0 (e.g., 12), the simulated year (e.g. 2023) is shifted by this number of
# years to the future (2023 + 12 = 2035): Renewable energy generation and demand data of 2035 is scaled according to
# projections of 2035. Emission and price data is estimated based on residual load


# validate the plausibility of the scenario definitions in scenario_list
def validate_scenario_list(scenario_list, logging=None):
    if type(scenario_list) is not list:
        return print_error(logging, "Scenario_list is not a list.")
    if len(scenario_list) <= 0:
        return print_error(logging, "Scenario_list is empty.")

    for scenario in scenario_list:
        if ID not in scenario:
            return print_error(logging, "A scenario in the scenario_list has no ID:\n   %s" % str(scenario))
        if SIM_START not in scenario:
            return print_error(logging, "A scenario in the scenario_list has no SIM_START:\n   %s" % str(scenario))
        if SIM_STOP not in scenario:
            return print_error(logging, "A scenario in the scenario_list has no SIM_STOP:\n   %s" % str(scenario))
        # if SIM_YEARS not in scenario:
        #     return print_error(logging, "A scenario in the scenario_list has no SIM_YEARS:\n   %s" % str(scenario))
        scenario_items_leftover = scenario.copy()
        sc_id = scenario_items_leftover.pop(ID)
        if type(sc_id) is not int:
            return print_error(logging, "The scenario ID should be an int, but it currently is %s (%s)):"
                               % (str(sc_id), str(type(sc_id))))
        sc_sim_start = scenario_items_leftover.pop(SIM_START)
        if type(sc_sim_start) is not datetime.date:
            return print_error(logging, "SIM_START of scenario %u should be a datetime.date, but it currently is "
                                        "%s (%s)):" % (sc_id, str(sc_sim_start), str(type(sc_sim_start))))
        sc_sim_stop = scenario_items_leftover.pop(SIM_STOP)
        if type(sc_sim_stop) is not datetime.date:
            return print_error(logging, "SIM_STOP of scenario %u should be a datetime.date, but it currently is "
                                        "%s (%s)):" % (sc_id, str(sc_sim_stop), str(type(sc_sim_stop))))
        # sc_sim_years = scenario_items_leftover.pop(SIM_YEARS)
        # if type(sc_sim_years) is not int:
        #     return print_error(logging, "SIM_YEARS of scenario %u should be an int, but it currently is %s (%s)):"
        #                        % (sc_id, str(sc_sim_years), str(type(sc_sim_years))))
        if SHIFT_BY_YEARS in scenario:
            shift_by_years = scenario_items_leftover.pop(SHIFT_BY_YEARS)
            if type(shift_by_years) is not int:
                return print_error(logging, "SHIFT_BY_YEARS of scenario %u should be an int, but it currently is "
                                            "%s (%s)):" % (sc_id, str(shift_by_years), str(type(shift_by_years))))
        n_loc = 0
        n_chg = 0
        for loc in LOCATION_ARRAY:
            if loc in scenario:
                valid, can_charge = validate_scenario_location(logging, sc_id, scenario_items_leftover.pop(loc), loc)
                if not valid:
                    return
                if can_charge:
                    n_chg = n_chg + 1
                n_loc = n_loc + 1
            else:
                if loc == HOME:
                    return print_error(logging, "Scenario %u has no %s definition (mandatory)" % (sc_id, loc))
                else:
                    print_info(logging, "Scenario %u has no %s definition" % (sc_id, loc))
        if n_loc == 0:
            return print_error(logging, "No locations provided for scenario %u" % sc_id)
        if n_chg == 0:
            return print_error(logging, "No charging opportunities provided for scenario %u" % sc_id)

        if len(scenario_items_leftover) > 0:
            print_info(logging, "Scenario %u has unknown extra items that will be ignored:\n   %s"
                       % (sc_id, str(scenario_items_leftover)))

    return True


# validate the plausibility of parameters/settings for a location of a scenario_list entry
def validate_scenario_location(logging, sc_id, scenario_loc, loc):
    scenario_loc_leftover = scenario_loc.copy()
    can_charge = False
    sc_chg_strat = CHG_STRAT.NONE
    if CHG_STRATEGY in scenario_loc:
        sc_chg_strat = scenario_loc_leftover.pop(CHG_STRATEGY)

    if (sc_chg_strat > CHG_STRAT.NONE) or (loc == TRIP):
        # charging strategy provided -> P, V, I should be given
        if (CHG_P in scenario_loc) and (CHG_V_LIM in scenario_loc) and (CHG_I_CO in scenario_loc):
            p_chg = scenario_loc_leftover.pop(CHG_P)
            v_chg = scenario_loc_leftover.pop(CHG_V_LIM)
            i_chg = scenario_loc_leftover.pop(CHG_I_CO)
            if p_chg <= 0.0:
                return print_error(logging, "Charging strategy provided for scenario %u %s, but charging power is "
                                            "invalid (should be positive)" % (sc_id, loc)), False
            if i_chg < 0.0:
                return print_error(logging, "Charging strategy provided for scenario %u %s, but charging current is "
                                            "invalid (should be positive or 0)" % (sc_id, loc)), False
            if v_chg <= bat.V_CELL_MIN:
                return print_error(logging, "Charging strategy provided for scenario %u %s, but charging voltage is "
                                            "invalid (should be in valid cell voltage range (%.3f, %.3f])"
                                   % (sc_id, loc, bat.V_CELL_MIN, bat.V_CELL_MAX)), False
            if v_chg > bat.V_CELL_MAX:
                print_warning(logging, "Charging strategy provided for scenario %u %s, but charging voltage is too high"
                                       " (should be in valid cell voltage range (%.3f, %.3f]) -> will be limited!"
                              % (sc_id, loc, bat.V_CELL_MIN, bat.V_CELL_MAX))
            p_chg_cell = p_chg * bat.wltp_profiles.P_EV_KW_TO_P_CELL_W
            i_cutoff_max = p_chg_cell / v_chg  # the current reached at the charging voltage
            if i_chg > i_cutoff_max:
                print_warning(logging, "Charging strategy provided for scenario %u %s, but charging cut-off current "
                                       "might be too high - this could prevent charging to start. "
                                       "Cut-off current should be 0 <= i_co < P_cell / V_CHG_LIM, i.e., < %.4f"
                              % (sc_id, loc, i_cutoff_max))
            can_charge = True
        else:
            if loc == TRIP:
                return print_error(logging, "No charging parameters (P/I/V) provided for scenario %u %s"
                                   % (sc_id, loc)), False
            else:
                return print_error(logging, "Charging strategy provided for scenario %u %s, but no charging parameters "
                                            "(P/I/V) given" % (sc_id, loc)), False

        if ((sc_chg_strat == CHG_STRAT.EARLY_IF_LOW) or (sc_chg_strat == CHG_STRAT.LATE_IF_LOW)
                or (sc_chg_strat == CHG_STRAT.V1G_OPT_EMISSION) or (sc_chg_strat == CHG_STRAT.V1G_OPT_COST)
                or (sc_chg_strat == CHG_STRAT.V1G_OPT_REN) or (sc_chg_strat == CHG_STRAT.V2G_OPT_EMISSION)
                or (sc_chg_strat == CHG_STRAT.V2G_OPT_COST) or (sc_chg_strat == CHG_STRAT.V2G_OPT_REN)
                or (sc_chg_strat == CHG_STRAT.V2G_OPT_PV)
                or (sc_chg_strat == CHG_STRAT.V2G_OPT_FREQ)):  # or (sc_chg_strat == CHG_STRAT.V2G_OPT_BAT)):
            if CHG_SOC_LOW in scenario_loc:
                soc_low = scenario_loc_leftover.pop(CHG_SOC_LOW)
                if (not np.issubdtype(type(soc_low), np.number)) or (soc_low < 0.0) or (soc_low > 1.0):
                    return print_error(logging, "Charging strategy %u provided for scenario %u %s, but SOC_LOW (%s) is "
                                                "invalid (should be in valid SoC range [0.0, 1.0]"
                                       % (sc_chg_strat, sc_id, loc, str(soc_low))), False
            else:
                return print_error(logging, "Charging strategy %u provided for scenario %u %s, but no SOC_LOW given"
                                   % (sc_chg_strat, sc_id, loc)), False

    if (loc == WORK) or (loc == FREE) or (loc == TRIP):
        if DEPARTURE in scenario_loc:
            departure = scenario_loc_leftover.pop(DEPARTURE)
            if np.issubdtype(type(departure), np.number):
                if (departure < 0.0) or (departure >= 24.0):
                    return print_error(logging, "Invalid departure time %s for scenario %u - should be in range [0, 24)"
                                       % (str(departure), sc_id)), False
            elif type(departure) is list:
                if len(departure) > 2:
                    print_warning(logging, "Departure range list length for scenario %u is > 2, ignoring all entries "
                                           "except first (earliest) and second (latest) departure time" % sc_id)
                if departure[1] <= departure[0]:
                    return print_error(logging, "Invalid departure time range for scenario %u - the first (earliest) "
                                                "should be <= the second (latest) departure time" % sc_id), False
                if (departure[0] < 0.0) or (departure[0] >= 24.0) or (departure[1] < 0.0) or (departure[1] >= 24.0):
                    return print_error(logging, "Invalid departure time(s) [%s, %s) for scenario %u - both should be in"
                                                " range [0, 24)" % (str(departure[0]), str(departure[1]), sc_id)), False
            else:
                return print_error(logging, "Invalid departure time for scenario %u - should be an int/float value or "
                                            "list with two entries, both entries in the range [0, 24)" % sc_id), False
        else:
            return print_error(logging, "No departure provided for %s in scenario %u" % (loc, sc_id)), False

        if (loc == WORK) or (loc == FREE):
            if DURATION in scenario_loc:
                duration = scenario_loc_leftover.pop(DURATION)
                if np.issubdtype(type(duration), np.number):
                    if (duration < 0.0) or (duration >= 24.0):
                        return print_error(logging,
                                           "Invalid duration (%s) for scenario %u %s - should be in range [0, 24)"
                                           % (str(duration), sc_id, loc)), False
                elif type(duration) is list:
                    if len(duration) > 2:
                        print_warning(logging,
                                      "Duration range list length for scenario %u %s is > 2, ignoring all entries "
                                      "except first (earliest) and second (latest) duration" % (sc_id, loc))
                    if duration[1] <= duration[0]:
                        return print_error(logging,
                                           "Invalid duration range for scenario %u %s - the first (earliest) "
                                           "should be <= the second (latest) duration" % (sc_id, loc)), False
                    if (duration[0] < 0.0) or (duration[0] >= 24.0) or (duration[1] < 0.0) or (duration[1] >= 24.0):
                        return print_error(logging, "Invalid duration(s) [%s, %s) for scenario %u %s - both should be "
                                                    "in range [0, 24)"
                                           % (str(duration[0]), str(duration[1]), sc_id, loc)), False
                else:
                    return print_error(logging,
                                       "Invalid duration for scenario %u %s - should be an int/float value or "
                                       "list with two entries, both entries in the range [0, 24)" % (sc_id, loc)), False
            else:
                return print_error(logging, "No duration provided for scenario %u %s" % (sc_id, loc)), False

    if len(scenario_loc_leftover) > 0:
        print_info(logging, "Scenario %u %s has unknown extra items that will be ignored:\n   %s"
                   % (sc_id, loc, str(scenario_loc_leftover)))

    return True, can_charge


# print error to a logging instance (or to console, if not present)
def print_error(logging, msg):
    msg = "Error: " + msg
    if logging is None:
        print(msg)
    else:
        logging.log.error(msg)
    return False


# print warning to a logging instance (or to console, if not present)
def print_warning(logging, msg):
    msg = "Warning: " + msg
    if logging is None:
        print(msg)
    else:
        logging.log.warning(msg)


# print information to a logging instance (or to console, if not present)
def print_info(logging, msg):
    msg = "Info: " + msg
    if logging is None:
        print(msg)
    else:
        logging.log.info(msg)


# get detailed information for a scenario (e.g., for plot subtitle or console/log)
def get_scenario_subtitle(scenario):
    # assume ID is not needed
    text = ""
    text = append_scenario_loc_subtitle(text, scenario, HOME)
    text = append_scenario_loc_subtitle(text, scenario, WORK)
    text = append_scenario_loc_subtitle(text, scenario, FREE)
    text = append_scenario_loc_subtitle(text, scenario, TRIP)
    return text


# get detailed information for a location of a scenario
def append_scenario_loc_subtitle(text, scenario, loc):
    text = text + loc + ": "
    if loc in scenario:
        sc_loc = scenario.get(loc)
        if CHG_STRATEGY in sc_loc:
            chg_strat = sc_loc.get(CHG_STRATEGY)
        else:
            chg_strat = CHG_STRAT.NONE
        text = text + "chg: %s " % chg_strat.name
        if chg_strat != CHG_STRAT.NONE:
            text = text + "("
            if CHG_P in sc_loc:
                p = sc_loc.get(CHG_P)
                text = text + "%.1fkW, " % p
            if CHG_V_LIM in sc_loc:
                v = sc_loc.get(CHG_V_LIM)
                text = text + "%.2fV, " % v
            if CHG_I_CO in sc_loc:
                i = sc_loc.get(CHG_I_CO)
                text = text + ">%.2fA, " % i
            if CHG_SOC_LOW in sc_loc:
                soc = sc_loc.get(CHG_SOC_LOW)
                text = text + "if <%.0f%%" % (soc * 100)
            text = text + ")"
        if DEPARTURE in sc_loc:
            dep = sc_loc.get(DEPARTURE)
            text = text + ", dep: %s" % str(dep)
        if DURATION in sc_loc:
            dur = sc_loc.get(DURATION)
            text = text + ", dur: %s" % str(dur)
    else:
        text = text + "N/A"

    text = text + "<br>"
    return text
