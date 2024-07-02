# helper functions to import and process input data (temperature, electricity data, ...)
# ToDo: YOU NEED TO DOWNLOAD THE DATA BEFORE USING THIS SCRIPT OR THE use_case_model_EV_modular SCRIPTS!
#   (They cannot be included in the script because they are not public domain but owned by other institutions/people.
#   However, they are available online for free.)
#
# ToDo: Please also have a look at the other ToDo's --> adjust paths...

import datetime
import math
# import pyarrow as pa
from pyarrow import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback


TIMEZONE_DEFAULT = 'Europe/Berlin'

input_data_dir = "D:\\bat\\analysis\\use_case_models\\input_data\\"  # ToDo: adjust to path of your data

# temperature data from 01.01.2010 to 13.03.2024 (or now)
#   if earlier/later data is needed, shift looked-up date by 12, 24, 36, ... years
temperature_data_file = input_data_dir + "weather_temperature\\data_OBS_DEU_PT10M_T2M_4177.csv"  # ToDo: adjust filename
temperature_date_column = "Zeitstempel"
temperature_value_column = "Wert"
temperature_data_columns = {temperature_date_column: "str", temperature_value_column: "float"}
temperature_data_sep = ","

# CO2 emission data from 01.01.2019 to 31.12.2023
#   if earlier/later data is needed, shift looked-up date by 4, 8, 12, ... years
emission_co2mon_data_file = input_data_dir + "co2-monitor.org\\EMF_2019-2023.csv"  # ToDo: adjust filename
emission_co2mon_date_column = "timestamp"  # in s (Unix timestamp)
# emission_co2mon_value_column = "scope_2"  # in g CO2_eq / kWh, combustion-related emissions due to electr. consumption
# emission_co2mon_value_column = "scope_3"  # in g CO2_eq / kWh, upstream emissions incl. grid losses
emission_co2mon_value_column = "scope_lc"  # in g CO2_eq / kWh, life cycle: combustion-related + upstream emissions
emission_co2mon_data_columns = {emission_co2mon_date_column: "float", emission_co2mon_value_column: "float"}
emission_co2mon_data_sep = ","

# CO2 emission data from 01.01.2012 to 18.03.2024
#   if earlier/later data is needed, shift looked-up date by 12, 24, ... years
# 1. to generate the file, use RegEx replacement (e.g., in Notepad++) before importing (because of double data column):
#    replace "\n[^,\r\n]+,([^,\r\n]+,[^,\r\n]+),[^,\r\n]+,[^,\r\n]+,[^,\r\n]+,[^,\r\n]+,[^,\r\n]+" with "\n$1"
# 2. in case there is still an error with missing values during the switch from summer to winter time, manually
#    duplicate the timestamps during the shifts from summer to winter time (20**-10-**T02:00:00,**.*)
emission_agora_data_file = input_data_dir + ("agora_co2_emissions\\"  # ToDo: adjust filename
                                             "co2_emissions_from_power_generation_2012-01-01_2024-03-18_combined.csv")
emission_agora_datetime_column = "date_id"
emission_agora_value_column = "Grid emission factor"  # in g CO2_eq / kWh
emission_agora_data_columns = {emission_agora_datetime_column: "str", emission_agora_value_column: "float"}
emission_agora_data_sep = ","

# frequency data from 22.06.2022 to 16.03.2024
#   if earlier/later data is needed, shift looked-up date by 1, 2, 3, ... times the duration of the data array
#   --> the then used frequency data will be from the same time of the day, but from a different day in the year
frequency_data_file = input_data_dir + "frequency\\Frequenz_combined_20220622_20240316.csv"  # ToDo: adjust filename
frequency_date_column = "DATE"  # date DD.MM.YYYY
frequency_time_column = "TIME"  # time HH:MM:SS
frequency_value_column = "FREQUENCY_[HZ]"  # in Hz
frequency_time_origin = 'Europe/Berlin'
frequency_data_columns = {frequency_date_column: "str", frequency_time_column: "str", frequency_value_column: "float"}
frequency_data_sep = ";"
frequency_decimal_sep = ","

# electricity price data from 01.01.2015 to 18.03.2024 (or now)
#   if earlier/later data is needed, shift looked-up date by 8, 16, 24, ... years
el_price_smard_data_file = (input_data_dir + "SMARD_electricity_price\\"  # ToDo: adjust filename
                                             "Großhandelspreise_201501010000_202403182359_Stunde_combined.csv")
el_price_smard_date_column = "Datum"  # date DD.MM.YYYY
el_price_smard_time_column = "Anfang"  # time HH:MM - electricity price valid in [this, this + 1h)
el_price_smard_value_column = "DE/(AT/)LU [€/MWh] Originalauflösungen"  # in €/MWh -> convert to ct/kWh
el_price_smard_conversion_divisor = 10.0  # divide €/MWh by 10 to get ct/kWh (1 €/MWh = 100 ct / 1000 kWh)
el_price_smard_time_origin = 'Europe/Berlin'
el_price_smard_data_columns = {el_price_smard_date_column: "str", el_price_smard_time_column: "str",
                               el_price_smard_value_column: "float"}
el_price_smard_data_sep = ";"

# electricity price data from 01.01.2012 to 17.03.2024 (or now)
#   if earlier/later data is needed, shift looked-up date by 12, 24, 36 ... years
# 1. to generate the file, use RegEx replacement (e.g., in Notepad++) before importing (because of double data column):
#    replace "\n[^,\r\n]+,([^,\r\n]+,[^,\r\n]+),[^,\r\n]+,[^,\r\n]+" with "\n$1"
# 2. in case there is still an error with missing values during the switch from summer to winter time, manually
#    duplicate the timestamps during the shifts from summer to winter time (20**-10-**T02:00:00,**.*)
el_price_agora_data_file = (input_data_dir + "agora_electricity_price\\power_price_and_renewable_generation_share_"
                                             "2012-01-01_2024-03-18_combined.csv")  # ToDo: adjust filename
el_price_agora_datetime_column = "date_id"  # date YYYY-MM-DDTHH-MM-SS - electricity price valid in [this, this + 1h)
el_price_agora_value_column = "Power price"  # in €/MWh -> convert to ct/kWh
el_price_agora_conversion_divisor = 10.0  # divide €/MWh by 10 to get ct/kWh (1 €/MWh = 100 ct / 1000 kWh)
el_price_agora_time_origin = 'Europe/Berlin'
el_price_agora_data_columns = {el_price_agora_datetime_column: "str", el_price_agora_value_column: "float"}
el_price_agora_data_sep = ","

# electricity generation and demand data from 01.01.2015 to including 17.03.2024 18:00 (1710694800, or now)
#   if earlier/later data is needed, shift looked-up date by 8, 16, 24, ... years. scale installation data according
#   to projections
EL_GEN_DEMAND_DROP_DATA_START = 1710630000  # drop data that is newer than this (Sun Mar 17 2024 00:00:00 GMT+0100)
GEN_PV = "PV"
GEN_WIND_ONSHORE = "Wind Onshore"
GEN_WIND_OFFSHORE = "Wind Offshore"
GEN_BIOMASS = "Biomass"
GEN_HYDRO = "Hydropower"
GEN_REN_TOTAL = "Total renewable Generation"
RESIDUAL_LOAD = "Residual load"
DEMAND = "Demand"
el_gen_demand_base_dir = input_data_dir + "SMARD_generation_and_demand\\"
el_gen_installed_data_file = "EE_installed_interpolated.csv"
el_demand_data_file = "410_electricity_demand.csv"
el_gen_renewable_rel_files = {
    GEN_PV: "4068_photovoltaic_generation_rel.csv",
    GEN_WIND_ONSHORE: "4067_wind_onshore_generation_rel.csv",
    GEN_WIND_OFFSHORE: "1225_wind_offshore_generation_rel.csv",
    GEN_BIOMASS: "4066_biomass_generation_rel.csv",
    GEN_HYDRO: "1226_hydro_generation_rel.csv",
    # ignore other renewables, they have a very small contribution, and we don't know how they will be scaled
}
el_gen_renewable_rel_columns = {
    GEN_PV: "photovoltaic_generation",
    GEN_WIND_ONSHORE: "wind_onshore_generation",
    GEN_WIND_OFFSHORE: "wind_offshore_generation",
    GEN_BIOMASS: "biomass_generation",
    GEN_HYDRO: "hydro_generation",
}
el_gen_renewable_installed_columns = {
    GEN_PV: "photovoltaic_installed",
    GEN_WIND_ONSHORE: "wind_onshore_installed",
    GEN_WIND_OFFSHORE: "wind_offshore_installed",
    GEN_BIOMASS: "biomass_installed",
    GEN_HYDRO: "hydropower_installed",
}
el_demand_column = "electricity_demand"
el_gen_demand_datetime_column = "unixtimestamp_s"
el_gen_demand_value_column = "Power price"  # in €/MWh -> convert to ct/kWh
el_gen_demand_abs_conversion_divisor = 250.0  # divide "MWh per quarter-hour" by 250 to get "GW" (for non-relative val.)
el_gen_demand_yearly_conversion_mul = 1000.0  # multiply "TWh" by 1000 to get "GWh" for yearly overall demand
el_gen_demand_data_sep = ";"
el_gen_demand_time_dtype = np.uint64
el_gen_demand_value_dtype = np.float64
el_demand_yearly = pd.Series({  # future: scenario B of NEP_2037_2045_V2023_2_Entwurf_Teil1_2
    2015: 500.22, 2016: 503.08, 2017: 505.67, 2018: 509.16, 2019: 497.29,  # demand according to SMARD
    2020: 485.29, 2021: 504.52, 2022: 482.64, 2023: 457.68,  # demand according to SMARD
    # assumption: linear interpolation between values, bfill before, ffill after (keep constant outside)
    2037: 929.8, 2045: 1053.4,  # using electricity demand, power-to-heat and 50% of power-to-hydrogen
    # https://www.netzentwicklungsplan.de/sites/default/files/2023-06/NEP_2037_2045_V2023_2_Entwurf_Teil1_2.pdf
})
# interpolate missing years
el_demand_yearly = el_demand_yearly.reindex(range(el_demand_yearly.index.min(),
                                                  el_demand_yearly.index.max() + 1)).interpolate()
el_gen_tz_origin = 'Europe/Berlin'
EL_GEN_DEM_COLS = list(el_gen_renewable_rel_columns.values()) + [el_demand_column]
EL_GEN_INST_COLS = list(el_gen_renewable_installed_columns.values())
EL_GEN_OUTPUT_COLS = list(el_gen_renewable_rel_columns.keys())
EL_GEN_DEM_OUTPUT_COLS = EL_GEN_OUTPUT_COLS + [GEN_REN_TOTAL, DEMAND, RESIDUAL_LOAD]


# household load data from 01.01.2010 to 31.12.2010
# https://solar.htw-berlin.de/elektrische-lastprofile-fuer-wohngebaeude/
# here: household 31 (17 can also be used)
# "If only one load profile is to be used for the calculation, but the results are still to be as representative as
#   possible, it is recommended to use:
#     - Load profile 31 with regard to a good match of the seasonal curve to the SLP
#     - Load profile 17 with regard to a good match of the diurnal profile to the SLP"
# -> load profile 17 has a section with low power demand (holiday?) --> use load profile 31
# -> we use a 30 s average version since the 1 s profile is quite time-consuming to calculate and the 30 s profile still
#    contains quite representative peaks
# -> if earlier/later data is needed, shift looked-up date by 7 * 52 days (to preserve weekdays)
# load_profile_data_file = input_data_dir + "htw_household_load_data\\htw_household_17_30s.csv"
load_profile_data_file = input_data_dir + "htw_household_load_data\\htw_household_31_30s.csv"
load_profile_timestamp_column = "unixtimestamp_s"  # in s (Unix timestamp)
load_profile_power_column = "power_W"  # in W
load_profile_data_columns = {load_profile_timestamp_column: "int", load_profile_power_column: "float"}
load_profile_data_sep = ","


# load temperature data from file and return it
def load_temperature_data(output_timezone=TIMEZONE_DEFAULT, as_unixtimestamp=True):
    # ToDo: if you receive a warning like:
    #   pyarrow.lib.ArrowInvalid: CSV parse error: Expected 6 columns, got 7:
    #   "OBS_DEU_PT10M_T2M","4177","2010-01-01T00:00:00","6.7","111","3",
    #   insert a comma at the end of the first row (header)
    temp_df = pd.read_csv(temperature_data_file, header=0, sep=temperature_data_sep,  # engine="pyarrow",
                          usecols=list(temperature_data_columns.keys()), dtype=temperature_data_columns,
                          parse_dates=[temperature_date_column])
    temp_df.set_index(temperature_date_column, drop=True, inplace=True)

    # resample with forward fill --> there is one sneaky 60 minute gap in the otherwise 10-Minute data
    temp_df = temp_df.resample('10Min').ffill()

    if as_unixtimestamp:
        # convert from datetime to unixtimestamp
        # temp_df.index = (temp_df.index - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta("1s")
        temp_df.index = (temp_df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")  # has no timezone yet!
    else:
        # data is in UTC and shall be converted to output_timezone
        temp_df.index = temp_df.index.tz_localize('UTC').tz_convert(output_timezone)
    temp_df.index.name = None
    temp_df = temp_df[temperature_value_column]
    # temp_df.rename(columns={temperature_value_column: ""}, inplace=True)
    return temp_df


# return temperature data for a time series. The data will be interpolated or down-/up-sampled. If data outside the
# available range is requested, available data from a time +/-12*n years that is available is used
# (available data is shifted to the requested time_series)
# ToDo: If your temperature data covers less than 12 years, adjust "shift_years". A multiple of 4 years is recommended
#       to avoid the data to be shifted over time due to leap years.
# (this function returns the data at a similar, available time, if the requested times in time_series are not available)
def get_temperature_data(data_df, time_series, is_unix_timestamp=True):
    return get_transformed_data(data_df, time_series, is_unix_timestamp, shift_years=12)


# load electricity emission data from file and return it.
def load_emission_data(data_source="Agora"):
    if data_source == "Agora":
        emission_df = pd.read_csv(emission_agora_data_file, header=0, sep=emission_agora_data_sep,
                                  usecols=list(emission_agora_data_columns.keys()), dtype=emission_agora_data_columns)
        emission_df["dt"] = pd.to_datetime(emission_df[emission_agora_datetime_column], format="%Y-%m-%dT%H:%M:%S")
        emission_df.set_index("dt", drop=True, inplace=True)
        emission_df = emission_df.tz_localize('Europe/Berlin', ambiguous='infer')

        # convert from datetime to unixtimestamp
        emission_df.index = (emission_df.index - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta("1s")

        emission_df.index.name = None
        emission_df = emission_df[emission_agora_value_column]
    elif data_source == "co2monitor":
        emission_df = pd.read_csv(emission_co2mon_data_file, header=0, sep=emission_co2mon_data_sep,
                                  usecols=list(emission_co2mon_data_columns.keys()), dtype=emission_co2mon_data_columns)
        emission_df.set_index(emission_co2mon_date_column, drop=True, inplace=True)
        # emission_df.index.name = None
        emission_df = emission_df[emission_co2mon_value_column]
    else:
        emission_df = None
    return emission_df


# return electricity emission data for a time series. The data will be interpolated or down-/up-sampled. If data outside
# the available range is requested, two methods exist (similar to the price data).
# A) Leave residual_load=None - Available data from a time +/-m*n years (m = 12 or 4 depending on the data source) that
#    is available is used (available data is shifted to the requested time_series) as a fallback.
# B) Provide a residual load time series for the time series that the data is requested for. For times with available
#    historic data, the historic data is used. For times outside the available range emissions are estimated based on
#    residual load.
# (you can also provide own future residual load, price, and emission time series estimates for the simulation time if
#  you adjust the code for loading and returning the data)
# (this function returns the data at a similar, available time, if the requested times in time_series are not available)
def get_emission_data(data_df, time_series, data_source="Agora", is_unix_timestamp=True,
                      residual_load=None):
    if data_source == "Agora":
        shift_years = 12
    elif data_source == "co2monitor":
        shift_years = 4
    else:
        return None
    if residual_load is None:  # simple approach -> if time outside the available data range, copy/repeat available data
        return get_transformed_data(data_df, time_series, is_unix_timestamp, shift_years=shift_years)
    else:
        if np.issubdtype(type(time_series), pd.Index):
            time_series = pd.Series(time_series, index=time_series)

        # detect data that's inside the available data and one that is outside
        cond_estimate = ((time_series < data_df.index[0]) | (time_series > data_df.index[-1]))

        # for available data:
        emission_df = get_transformed_data(data_df, time_series[~cond_estimate], is_unix_timestamp,
                                           shift_years=shift_years)

        # for data outside the input data range:
        # noinspection PyTypeChecker
        if any(cond_estimate):
            time_series_est = time_series[cond_estimate]
            est_ixs = pd.Index(time_series_est[~time_series_est.isin(residual_load.index)])
            all_ixs = est_ixs.union(residual_load.index)
            # residual_roi = residual_load.reindex(all_ixs).sort_index().interpolate(method="index")
            residual_roi = residual_load.reindex(all_ixs).sort_index().ffill().bfill()
            estimated_df = get_emission_estimate_based_on_residual_load(residual_roi)

            # merge
            emission_df = emission_df.reindex(time_series)
            try:
                emission_df[est_ixs] = estimated_df
            except ValueError:  # if residual_load already includes all necessary indexes?
                emission_df[all_ixs] = estimated_df
            if any(emission_df.isna()):
                # emission_df = emission_df.interpolate(method="index").ffill().bfill()
                emission_df = emission_df.ffill().bfill()
                print("debug")
        return emission_df


# load frequency data from file and return it
def load_freq_data(output_timezone=TIMEZONE_DEFAULT, as_unixtimestamp=True):
    # ---
    # too slow:
    # freq_df = pd.read_csv(frequency_data_file, header=0, sep=frequency_data_sep, decimal=frequency_decimal_sep,
    #                       usecols=list(frequency_data_columns.keys()), dtype=frequency_data_columns)
    # freq_df["datetime"] = freq_df[frequency_date_column] + " " + freq_df[frequency_time_column]
    # freq_df["dt"] = pd.to_datetime(freq_df["datetime"], format="%d.%m.%Y %H:%M:%S")
    # ---
    # # alternative:
    # convert_options = csv.ConvertOptions(decimal_point=frequency_decimal_sep,
    #                                      timestamp_parsers=["%d.%m.%Y", "%H:%M:%s"])
    # parse_options = csv.ParseOptions(delimiter=frequency_data_sep)
    # freq_pa = csv.read_csv(frequency_data_file, parse_options=parse_options, convert_options=convert_options)
    # freq_df = freq_pa.to_pandas()
    # freq_df["dt"] = freq_df.apply(lambda x: pd.datetime.combine(x[frequency_date_column], x[frequency_time_column]),1)
    # ---
    # faster ?
    convert_options = csv.ConvertOptions(decimal_point=frequency_decimal_sep, column_types=frequency_data_columns)
    parse_options = csv.ParseOptions(delimiter=frequency_data_sep)
    freq_pa = csv.read_csv(frequency_data_file, parse_options=parse_options, convert_options=convert_options)
    freq_df = freq_pa.to_pandas()
    freq_df["datetime"] = freq_df[frequency_date_column] + " " + freq_df[frequency_time_column]
    freq_df["dt"] = pd.to_datetime(freq_df["datetime"], format="%d.%m.%Y %H:%M:%S")
    # ---
    freq_df.set_index("dt", drop=True, inplace=True)
    freq_df = freq_df.tz_localize('Europe/Berlin', ambiguous='infer')
    if as_unixtimestamp:
        # convert from datetime to unixtimestamp
        freq_df.index = (freq_df.index - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta("1s")
    else:
        # data is in frequency_time_origin and shall be converted to output_timezone
        freq_df.index = freq_df.index.tz_localize(frequency_time_origin).tz_convert(output_timezone)
    freq_df.index.name = None
    freq_df = freq_df[frequency_value_column]
    return freq_df


# return frequency data for a time series. The data will be interpolated or down-/up-sampled. If data outside the
# available range is requested, data from another point in time is used (shifted by the complete data availability
# period, i.e., the input data is used kind of like a ring buffer. For example, if you provide just a week of frequency
# data, this week is just repeated over and over again.)
# (this function returns the data at a similar, available time, if the requested times in time_series are not available)
def get_freq_data(data_df, time_series, is_unix_timestamp=True):
    return get_transformed_data(data_df, time_series, is_unix_timestamp, shift_years=None)


# load electricity price data from file and return it
def load_electricity_price_data(data_source="Agora", output_timezone=TIMEZONE_DEFAULT, as_unixtimestamp=True):
    if data_source == "Agora":
        price_df = pd.read_csv(el_price_agora_data_file, header=0, sep=el_price_agora_data_sep,
                               usecols=list(el_price_agora_data_columns.keys()), dtype=el_price_agora_data_columns)
        price_df["dt"] = pd.to_datetime(price_df[el_price_agora_datetime_column], format="%Y-%m-%dT%H:%M:%S")
        price_df.set_index("dt", drop=True, inplace=True)
        price_df = price_df.tz_localize('Europe/Berlin', ambiguous='infer')
        if as_unixtimestamp:
            # convert from datetime to unixtimestamp
            price_df.index = (price_df.index - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta("1s")
        else:
            # data is in el_price_agora_time_origin and shall be converted to output_timezone
            price_df.index = price_df.index.tz_localize(el_price_agora_time_origin).tz_convert(output_timezone)
        price_df.index.name = None
        price_df = price_df[el_price_agora_value_column] / el_price_agora_conversion_divisor

    elif data_source == "SMARD":
        price_df = pd.read_csv(el_price_smard_data_file, header=0, sep=el_price_smard_data_sep,
                               usecols=list(el_price_smard_data_columns.keys()), dtype=el_price_smard_data_columns)
        price_df["datetime"] = price_df[el_price_smard_date_column] + " " + price_df[el_price_smard_time_column]
        price_df["dt"] = pd.to_datetime(price_df["datetime"], format="%d.%m.%Y %H:%M")
        price_df.set_index("dt", drop=True, inplace=True)
        price_df = price_df.tz_localize('Europe/Berlin', ambiguous='infer')
        if as_unixtimestamp:
            # convert from datetime to unixtimestamp
            price_df.index = (price_df.index - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta("1s")
        else:
            # data is in el_price_smard_time_origin and shall be converted to output_timezone
            price_df.index = price_df.index.tz_localize(el_price_smard_time_origin).tz_convert(output_timezone)
        price_df.index.name = None
        price_df = price_df[el_price_smard_value_column] / el_price_smard_conversion_divisor

        # the first 96 values (4 days) are nan -> simply copy the following 4 days (block-backfill)
        if price_df.isna().sum() >= 96:
            price_df[0:96] = price_df[96:192]
        if price_df.isna().sum() >= 0:
            # if there are more NaN's, fill with average electricity price
            price_df = price_df.fillna(price_df.mean())
    else:
        print("invalid electricity price data source")
        price_df = None

    return price_df


# return electricity price data for a time series. The data will be interpolated or down-/up-sampled. If data outside
# the available range is requested, two methods exist (similar to the emission data).
# A) Leave residual_load=None - Available data from a time +/-m*n years (m = 12 or 8 depending on the data source) that
#    is available is used (available data is shifted to the requested time_series) as a fallback.
# B) Provide a residual load time series for the time series that the data is requested for. For times with available
#    historic data, the historic data is used. For times outside the available range prices are estimated based on
#    residual load.
# (you can also provide own future residual load, price, and emission time series estimates for the simulation time if
#  you adjust the code for loading and returning the data)
# (this function returns the data at a similar, available time, if the requested times in time_series are not available)
def get_price_data(data_df, time_series, data_source="Agora", is_unix_timestamp=True,
                   residual_load=None):
    if data_source == "Agora":
        shift_years = 12
    elif data_source == "SMARD":
        shift_years = 8
    else:
        return None
    if residual_load is None:  # simple approach -> if time outside the available data range, copy/repeat available data
        return get_transformed_data(data_df, time_series, is_unix_timestamp, shift_years=shift_years)
    else:
        if np.issubdtype(type(time_series), pd.Index):
            time_series = pd.Series(time_series, index=time_series)

        # detect data that's inside the available data and one that is outside
        cond_estimate = ((time_series < data_df.index[0]) | (time_series > data_df.index[-1]))

        # for available data:
        price_df = get_transformed_data(data_df, time_series[~cond_estimate], is_unix_timestamp,
                                           shift_years=shift_years)

        # for data outside the input data range:
        # noinspection PyTypeChecker
        if any(cond_estimate):
            time_series_est = time_series[cond_estimate]
            est_ixs = pd.Index(time_series_est[~time_series_est.isin(residual_load.index)])
            all_ixs = est_ixs.union(residual_load.index)
            all_ixs = all_ixs[all_ixs.isin(time_series)]
            # residual_roi = residual_load.reindex(all_ixs).sort_index().interpolate(method="index")
            residual_roi = residual_load.reindex(all_ixs).sort_index().ffill().bfill()
            estimated_df = get_price_estimate_based_on_residual_load(residual_roi)

            # merge
            price_df = price_df.reindex(time_series)
            try:
                price_df[est_ixs] = estimated_df
            except ValueError:  # if residual_load already includes all necessary indexes
                price_df[all_ixs] = estimated_df
            if any(price_df.isna()):
                # price_df = price_df.interpolate(method="index").ffill().bfill()
                price_df = price_df.ffill().bfill()
                print("debug")
        return price_df


# load electricity generation and demand data from file, convert it into relative data (% of the installed generation
# capacity or annual electricity demand used) and return it
def load_el_gen_dem_data():
    # load relative generation power data
    gen_rel_df = pd.DataFrame(dtype=np.float64)
    for gen, file in el_gen_renewable_rel_files.items():
        value_col = el_gen_renewable_rel_columns.get(gen)
        dtype_dict = {el_gen_demand_datetime_column: el_gen_demand_time_dtype,
                      value_col: el_gen_demand_value_dtype}
        this_gen_rel_df = pd.read_csv(el_gen_demand_base_dir + file, header=0, sep=el_gen_demand_data_sep,
                                      dtype=dtype_dict)
        this_gen_rel_df.set_index(el_gen_demand_datetime_column, drop=True, inplace=True)
        this_gen_rel_df = this_gen_rel_df[value_col]
        if gen_rel_df.shape[0] == 0:
            gen_rel_df = this_gen_rel_df
        else:
            gen_rel_df = pd.concat([gen_rel_df, this_gen_rel_df], axis=1)
    gen_rel_df = gen_rel_df[gen_rel_df.isna().sum(axis=1) == 0]  # drop rows with any empty/NaN value
    gen_rel_df = gen_rel_df[gen_rel_df.index < EL_GEN_DEMAND_DROP_DATA_START]  # drop data with index >= DROP_DATA_START

    # load installed generation power data
    dtype_dict = {el_gen_demand_datetime_column: el_gen_demand_time_dtype}
    for col in el_gen_renewable_installed_columns.values():
        dtype_dict[col] = el_gen_demand_value_dtype
    gen_inst_df = pd.read_csv(el_gen_demand_base_dir + el_gen_installed_data_file, header=0,
                                 sep=el_gen_demand_data_sep, dtype=dtype_dict)
    gen_inst_df.set_index(el_gen_demand_datetime_column, drop=True, inplace=True)
    gen_inst_df = gen_inst_df[gen_inst_df.isna().sum(axis=1) == 0]  # drop rows with any empty/NaN value
    # gen_inst_df = gen_inst_df[gen_inst_df.index < DROP_DATA_START] **DON'T** drop data with index >= DROP_DATA_START!

    # load (absolute) demand data and convert it to relative demand (GW_momentary / GWh_year)
    dtype_dict = {el_gen_demand_datetime_column: el_gen_demand_time_dtype, el_demand_column: el_gen_demand_value_dtype}
    demand_df = pd.read_csv(el_gen_demand_base_dir + el_demand_data_file, header=0, sep=el_gen_demand_data_sep,
                               dtype=dtype_dict)
    demand_df.set_index(el_gen_demand_datetime_column, drop=True, inplace=True)
    demand_df = demand_df[~demand_df[el_demand_column].isna()]  # drop empty/NaN rows
    demand_df = demand_df[demand_df.index < EL_GEN_DEMAND_DROP_DATA_START]  # drop data with index >= DROP_DATA_START
    demand_df = demand_df / el_gen_demand_abs_conversion_divisor  # MWh per time interval -> GW

    min_year = pd.to_datetime(demand_df.index.min(), unit='s').tz_localize('UTC').tz_convert(el_gen_tz_origin).year
    max_year = pd.to_datetime(demand_df.index.max(), unit='s').tz_localize('UTC').tz_convert(el_gen_tz_origin).year
    t_u0 = pd.Timestamp("1970-01-01", tz='UTC')
    for y in range(min_year, max_year + 1):
        t_min = (pd.Timestamp(year=y, month=1, day=1, tz=el_gen_tz_origin) - t_u0) // pd.Timedelta("1s")
        t_max = (pd.Timestamp(year=y+1, month=1, day=1, tz=el_gen_tz_origin) - t_u0) // pd.Timedelta("1s")
        dem_year_GWh = get_demand_from_year(y) * el_gen_demand_yearly_conversion_mul
        cond = ((demand_df.index >= t_min) & (demand_df.index < t_max))
        demand_df.loc[cond, el_demand_column] = demand_df.loc[cond, el_demand_column] / dem_year_GWh

    # merge all columns
    # el_gen_dem_df = pd.concat([gen_rel_df, gen_inst_df, demand_df], axis=1)
    el_gen_dem_df = gen_rel_df.merge(gen_inst_df, left_index=True, right_index=True, how='outer')
    el_gen_dem_df = el_gen_dem_df.merge(demand_df, left_index=True, right_index=True, how='outer')
    el_gen_dem_df.sort_index(inplace=True)
    return el_gen_dem_df


# return electricity generation and demand data for a time series. The data will be interpolated or down-/up-sampled.
# This function returns projected data if the requested times in time_series are not available.
# scale_shift_years (>0) can be used to scale the renewable electricity share to future installation targets, i.e.,
#   data & dates of the year 2023 can be used but scaled to REN and demand targets of 2033 if scale_shift_years=10.
#   This preserves the weekdays of 2023, but simulates the REN production and electricity demand magnitude of 2033.
#   The advantage compared to simulating dates of 2033 and using data of 2023 is that the demand profile (which strongly
#   depends on weekdays) is not distorted. Instead, with scale_shift_years > 0, the magnitude of generation and demand
#   is adjusted to future installation and demand targets/projections.
def get_el_gen_dem_data(data_df, time_series, scale_shift_years=0, timezone=TIMEZONE_DEFAULT):
    # if type(time_series) is pd.Index:
    if np.issubdtype(type(time_series), pd.Index):
        time_series = np.array(time_series)
    # make sure we have relative generation data to work with
    gen_dem_data = data_df[EL_GEN_DEM_COLS][data_df.index < EL_GEN_DEMAND_DROP_DATA_START]
    gen_dem_data = get_transformed_data(gen_dem_data, time_series, True, shift_years=8)

    # make sure we have demand data to work with - scale to future if requested
    gen_dem_data[DEMAND] = get_transformed_demand_data(gen_dem_data[el_demand_column], time_series, scale_shift_years,
                                                       timezone=timezone)

    # scale generation to future if requested
    inst_data = data_df[EL_GEN_INST_COLS]
    inst_time_min = inst_data.index[0]
    inst_time_max = inst_data.index[-1]
    # get time series that will be used for installation data. consider leap years, but actually not very important
    #   because installation data changes very gradually
    time_series_use_inst = time_series + scale_shift_years * (365.25 * 24.0 * 60.0 * 60.0)
    cond_after = (time_series_use_inst > inst_time_max)
    time_series_use_inst[cond_after] = inst_time_max
    cond_before = (time_series_use_inst < inst_time_min)
    time_series_use_inst[cond_before] = inst_time_min

    new_ixs = pd.Index(time_series_use_inst)
    all_ixs = new_ixs.union(inst_data.index)
    # inst_data_roi = inst_data.reindex(all_ixs).sort_index().interpolate(method="index").ffill().bfill()
    inst_data_roi = inst_data.reindex(all_ixs).sort_index().ffill().bfill()
    inst_data_roi = inst_data_roi.loc[new_ixs, :]

    # inst_data_roi = inst_data_roi.loc[time_series_use_inst, :]

    # overwrite index -> for the use-case, it looks like the data was from the requested time
    try:
        inst_data_roi.index = time_series
    except ValueError:
        print("debug here")

    # create dataframe with PV, Wind, ..., Total renewable, Demand, Residual load
    data_df_roi = pd.DataFrame(dtype=np.float64, columns=EL_GEN_DEM_OUTPUT_COLS, index=time_series)
    try:
        for gen, rel in el_gen_renewable_rel_columns.items():
            inst = el_gen_renewable_installed_columns.get(gen)
            data_df_roi[gen] = gen_dem_data[rel] * inst_data_roi[inst]
    except ValueError:
        print("debug here")

    data_df_roi[GEN_REN_TOTAL] = data_df_roi[EL_GEN_OUTPUT_COLS].sum(axis=1)
    data_df_roi[DEMAND] = gen_dem_data[DEMAND]

    # Residual load: > 0 if we need more generation power, < 0 if we have excess renewables
    data_df_roi[RESIDUAL_LOAD] = data_df_roi[DEMAND] - data_df_roi[GEN_REN_TOTAL]

    return data_df_roi


# only return relative PV data from electricity generation and demand data
def get_el_gen_pv_data(data_df, time_series):
    if np.issubdtype(type(time_series), pd.Index):
        time_series = np.array(time_series)
    # make sure we have relative generation data to work with
    pv_col = el_gen_renewable_rel_columns.get(GEN_PV)
    gen_dem_data = data_df[[pv_col]][data_df.index < EL_GEN_DEMAND_DROP_DATA_START]
    gen_dem_data = get_transformed_data(gen_dem_data, time_series, True, shift_years=8)
    return gen_dem_data[pv_col]


# return historic or projected/estimated future yearly electricity demand --> see el_demand_yearly
def get_demand_from_year(year):
    if year in el_demand_yearly.index:
        return el_demand_yearly[year]
    elif year > el_demand_yearly.index.max():
        return el_demand_yearly[el_demand_yearly.index.max()]
    elif year < el_demand_yearly.index.min():
        return el_demand_yearly[el_demand_yearly.index.min()]
    # fallback, shouldn't happen since we filled el_demand_yearly before
    dem = el_demand_yearly.reindex(range(el_demand_yearly.index.min(), el_demand_yearly.index.max() + 1)).interpolate()
    if year in dem.index:
        return dem[year]
    return dem.mean()  # maybe the used entered the wrong data type?


# transform data, i.e., shift the available data to a desired time series
# interpolation, up-/down-sampling, shifting data (with or without preserving week days*) is supported.
# *this is useful since the electricity demand of a Saturday and Sunday is much different from that of a weekday
def get_transformed_data(data_df: pd.Series, time_series: pd.Series, is_unix_timestamp: bool,
                         shift_years=None, preserve_weekday=False, interpolate=False):
    time_min = data_df.index[0]
    time_max = data_df.index[-1]

    if np.issubdtype(type(time_series), pd.Index):
        time_series = pd.Series(time_series, index=time_series)
    time_series_use = time_series.copy()

    if shift_years is not None:
        if preserve_weekday:
            # Method 1: closest period in data with same weekdays. will not work if we only have one year of data!
            # shift_days = shift_years * 365.25  # always consider leap years
            # shift_days = round(shift_days / 7, 0) * 7  # round to

            # Method 2: 52 weeks. disadvantage: this will shift seasonality over the years!
            shift_days = shift_years * 364
        else:
            if math.remainder(shift_years, 4) == 0:  # shift_years is 4, 8, 12, 16, 20, ...
                shift_days = shift_years * 365.25  # consider leap years
            else:
                shift_days = shift_years * 365  # ignore leap years
        if is_unix_timestamp:
            time_shift = shift_days * 24.0 * 60.0 * 60.0
        else:
            time_shift = datetime.timedelta(days=shift_days)
    else:
        # shift by length of available data (has the same effect as repeating the data at the ends)
        time_shift = time_max - time_min + (data_df.index[-1] - data_df.index[-2])  # time_max - time_min + dt_step

    # back-shift:
    while True:
        cond_after = (time_series_use > time_max)
        if cond_after.sum() > 0:  # type: ignore
            time_series_use[cond_after] = time_series_use[cond_after] - time_shift
        else:
            break  # all in valid range

    # forward-shift:
    while True:
        cond_before = (time_series_use < time_min)
        if cond_before.sum() > 0:  # type: ignore
            time_series_use[cond_before] = time_series_use[cond_before] + time_shift
        else:
            break  # all in valid range

    try:
        # data_df_roi = pd.Series(np.nan, dtype=np.float64, index=time_series_use)
        # ixs = data_df.index[data_df.index.isin(time_series_use)]
        # data_df_roi[ixs] = data_df[ixs]
        if type(data_df) is pd.Series:
            # try:
            data_df_roi = data_df[time_series_use]
            # except Exception:
            #     print("Python Error in solar_charging:\n%s" % traceback.format_exc())
            #     print("debug")
        else:
            data_df_roi = data_df.loc[time_series_use, :]
    except KeyError:
        # try to use filling/interpolation
        new_ixs = pd.Index(time_series_use[~time_series_use.isin(data_df.index)])
        all_ixs = new_ixs.union(data_df.index)
        if interpolate:
            data_df = data_df.reindex(all_ixs).sort_index().interpolate(method="index")
        else:
            # noinspection PyBroadException
            try:
                data_df = data_df.reindex(all_ixs).sort_index().ffill().bfill()
            except Exception:  # prevent program termination -> we want to continue with the other scenarios regardless
                print("Python Error in solar_charging:\n%s" % traceback.format_exc())
                print("debug here")  # this might occur when something with the date conversion failed
                # --> e.g., the imported timestamps are UTC+1, and not in the expected timezone Europe/Berlin,
                # considering daylight saving time --> FIX this in data which is imported!

        if type(data_df) is pd.Series:
            data_df_roi = data_df[time_series_use]
        else:
            data_df_roi = data_df.loc[time_series_use, :]
        # try:  # is this really needed? -> probably not
        #     data_df = data_df.reindex(all_ixs).sort_index().interpolate(method="index")
        #     if type(data_df) is pd.Series:
        #         data_df_roi = data_df[time_series_use]
        #     else:
        #         data_df_roi = data_df.loc[time_series_use, :]
        # except KeyError:
        #     print("debug here")
        # print("debug here")

    # overwrite index -> for the use-case, it looks like the data was from the requested time
    try:
        data_df_roi.index = time_series
    except ValueError:
        print("debug")

    return data_df_roi


# transform electricity demand data, i.e., shift the available data to a desired time series
def get_transformed_demand_data(demand_data_ser, time_series, scale_shift_years, timezone=TIMEZONE_DEFAULT):
    min_year = pd.to_datetime(time_series.min(), unit='s').tz_localize('UTC').tz_convert(timezone).year
    max_year = pd.to_datetime(time_series.max(), unit='s').tz_localize('UTC').tz_convert(timezone).year
    t_u0 = pd.Timestamp("1970-01-01", tz='UTC')
    demand_data_transformed = demand_data_ser.copy()
    for y in range(min_year, max_year + 1):
        t_min = (pd.Timestamp(year=y, month=1, day=1, tz=el_gen_tz_origin) - t_u0) // pd.Timedelta("1s")
        t_max = (pd.Timestamp(year=y+1, month=1, day=1, tz=el_gen_tz_origin) - t_u0) // pd.Timedelta("1s")
        dem_future_GWh = get_demand_from_year(y + scale_shift_years) * el_gen_demand_yearly_conversion_mul
        cond = ((demand_data_transformed.index >= t_min) & (demand_data_transformed.index < t_max))
        demand_data_transformed[cond] = demand_data_transformed[cond] * dem_future_GWh
    return demand_data_transformed


# get an estimation of the electricity emissions based on the residual load
# Please feel free to improve this! For example, one could use average emissions based on a merit order list, or let
# emissions decrease over time, considering that hydrogen produced from renewable energy sources is used when the
# residual load is high.
def get_emission_estimate_based_on_residual_load(residual_load_df):  # residual_load_df in GW
    # Using data of all years: Emissions [gCO2eq/kWh] = 6.9587 * Residual load [GW] + 189.0052
    # Using data of the last 5 years: Emissions [gCO2eq/kWh] = 7.2344 * Residual load [GW] + 171.0925
    lin_a = 7.2344
    lin_b = 171.0925
    emission_min = 50.0  # in gCO2eq/kWh, minimum emissions always active - see Abbildung 8 in
    # https://www.umweltbundesamt.de/sites/default/files/medien/11850/publikationen/20231219_49_2023_cc_emissionsbilanz_erneuerbarer_energien_2022_bf.pdf
    # -> PV: 57 gCO2eq/kWh, Wind onshore: 18 gCO2eq/kWh, Wind offshore: 10 gCO2eq/kWh
    emission_df = lin_a * residual_load_df + lin_b
    emission_df[emission_df < emission_min] = emission_min
    return emission_df  # in gCO2eq/kWh


# get an estimation of the electricity prices based on the residual load
# Please feel free to improve this! For example, one could use prices based on a merit order list, or let
# prices vary over time (randomly or based on future electricity price estimates).
def get_price_estimate_based_on_residual_load(residual_load_df):  # residual_load_df in GW
    # Plotting data of all and last 5 years -> see test(), I determined the following, heavily simplified estimation
    # it is not good, but okay for our purposes. My impression is that it is rather pessimistic, i.e., I think the
    # price differences will often be more extreme and the avg. electricity price in general might be slightly lower.
    # The price scenario is based on the assumption that in short to medium term, the marginal price is determined by
    # gas power plants (LNG) and in the long term by hydrogen.
    # 1 ct/kWh = 10 €/MWh
    # 100 GW residual load: 27 ct/kWh (higher price is possible)
    #  40 GW residual load: 12 ct/kWh
    #   0 GW residual load:  2 ct/kWh
    # -20 GW residual load:  0 ct/kWh
    # -40 GW residual load: -2 ct/kWh (which is the minimum possible price)
    # I could imagine that in the future with a lot of flexibility, the price for residual load << 0 GW will be even
    # flatter? Of course the price also depends on the period of the residual load, e.g., if the residual load was
    # >> 0 GW for three days, prices will be higher than if the residual load just entered > 0 GW after a sustained
    # period of << 0 GW. We don't want to model this but focus on battery aging.
    # Feel free to implement a better function here :)
    c = 2         # price at 0 GW residual load: 2 ct/kWh
    m_pos = 0.25  # 0.25 ct/GW -> slope is 25 ct/100 GW for positive residual load
    m_neg = 0.1   # 0.1 ct/GW -> slope is 10 ct/100 GW for negative residual load (excess energy)
    p_min = -2    # minimum price: -2 ct/kWh
    price_df = m_pos * residual_load_df + c
    price_df[residual_load_df < 0.0] = m_neg * residual_load_df + c
    price_df[price_df < p_min] = p_min
    return price_df  # in ct/kWh


#
# load (household) load profile data from file and return it
def load_load_profile_data():  # output_timezone=TIMEZONE_DEFAULT
    temp_df = pd.read_csv(load_profile_data_file, header=0, sep=load_profile_data_sep,  # engine="pyarrow",
                          usecols=list(load_profile_data_columns.keys()), dtype=load_profile_data_columns)
    temp_df.set_index(load_profile_timestamp_column, drop=True, inplace=True)

    # # data is in UTC and shall be converted to output_timezone
    # temp_df.index = temp_df.index.tz_localize('UTC').tz_convert(output_timezone)
    temp_df.index = temp_df.index
    temp_df.index.name = None
    temp_df = temp_df[load_profile_power_column] / 1000.0
    # temp_df.rename(columns={load_profile_power_column: ""}, inplace=True)
    return temp_df  # power in kW


# return (household) load profile data for a time series. The data will be interpolated or down-/up-sampled. If data
# outside the available range is requested, available data from a time +/-n years that is available is used
# (available data is shifted to the requested time_series, weekdays are preserved by default)
# ToDo: If your hpushold data covers more than 1 year, adjust "shift_years". If data is available, a multiple of 4 years
#       is recommended to avoid the data to be shifted over time due to leap years.
# (this function returns the data at a similar, available time, if the requested times in time_series are not available)
def get_load_profile_data(data_df, time_series, is_unix_timestamp=True):
    return get_transformed_data(data_df, time_series, is_unix_timestamp, shift_years=1, preserve_weekday=True)


# for debugging only: visualization and several tests with the input data
def test():
    tz = TIMEZONE_DEFAULT

    el_gen_dem_data_df = load_el_gen_dem_data()
    # # ----------------------------------------------------------------------------------------------------------------
    # tsg = np.arange(1262300400, 2429910000, 3600)  # 01.01.2010 - 01.01.2047 -- 1 h resolution
    # el_gen_dem_plot = get_el_gen_dem_data(el_gen_dem_data_df, tsg, 0)
    # print("debug here")
    #
    # # plot results
    # el_gen_dem_plot.index = pd.to_datetime(el_gen_dem_plot.index, unit='s').tz_localize('UTC').tz_convert(tz)
    # plt.plot(el_gen_dem_plot.index, el_gen_dem_plot[GEN_REN_TOTAL], c="green")
    # plt.plot(el_gen_dem_plot.index, el_gen_dem_plot[DEMAND], c="black")
    # plt.plot(el_gen_dem_plot.index, el_gen_dem_plot[RESIDUAL_LOAD], c="red", linestyle='dashed')
    # plt.grid(True)
    # plt.show(block=True)
    # plt.pause(.1)
    # # ----------------------------------------------------------------------------------------------------------------

    temp_data_df = load_temperature_data()
    emission_co2mon_df = load_emission_data(data_source="co2monitor")
    emission_agora_df = load_emission_data(data_source="Agora")
    # freq_data_df = load_freq_data()
    price_smard_df = load_electricity_price_data(data_source="SMARD")
    price_agora_df = load_electricity_price_data(data_source="Agora")

    year_colors = {2015: np.array([255, 0, 0]) / 255.0,    # red
                   2016: np.array([255, 115, 0]) / 255.0,  # orange
                   2017: np.array([255, 255, 0]) / 255.0,  # yellow
                   2018: np.array([185, 255, 0]) / 255.0,  # light green
                   2019: np.array([23, 255, 0]) / 255.0,   # green
                   2020: np.array([0, 255, 255]) / 255.0,  # cyan
                   2021: np.array([0, 185, 255]) / 255.0,  # light blue
                   2022: np.array([0, 23, 255]) / 255.0,   # dark blue
                   2023: np.array([162, 0, 255]) / 255.0,  # purple
                   2024: np.array([255, 0, 255]) / 255.0}  # pink
    # ------------------------------------------------------------------------------------------------------------------
    # ixs = np.arange(1325362400, 1325366000, 300)
    # p_df = get_price_data(price_agora_df, ixs)

    # ------------------------------------------------------------------------------------------------------------------
    # # plot emissions vs. residual load (colored / grouped by year)
    # ixs_co2 = emission_agora_df.index
    # ixs_dem = el_gen_dem_data_df[el_gen_dem_data_df.index < EL_GEN_DEMAND_DROP_DATA_START].index
    # ixs = ixs_co2[ixs_co2.isin(ixs_dem)]
    # emission_roi = emission_agora_df[ixs]
    # # demand_roi = get_transformed_demand_data(el_gen_dem_data_df[el_demand_column][ixs], ixs, 0)
    # el_gen_dem_roi = get_el_gen_dem_data(el_gen_dem_data_df, ixs, scale_shift_years=0)
    # residual_roi = el_gen_dem_roi[RESIDUAL_LOAD]
    # min_year = pd.to_datetime(ixs.min(), unit='s').tz_localize('UTC').tz_convert(tz).year
    # max_year = pd.to_datetime(ixs.max(), unit='s').tz_localize('UTC').tz_convert(tz).year
    # t_u0 = pd.Timestamp("1970-01-01", tz='UTC')
    # x = np.arange(-20, 100, 1)
    # for y in range(min_year, max_year + 1):
    #     t_min = (pd.Timestamp(year=y, month=1, day=1, tz=el_gen_tz_origin) - t_u0) // pd.Timedelta("1s")
    #     t_max = (pd.Timestamp(year=y + 1, month=1, day=1, tz=el_gen_tz_origin) - t_u0) // pd.Timedelta("1s")
    #     cond = ((residual_roi.index >= t_min) & (residual_roi.index < t_max))
    #
    #     a, b = np.polyfit(residual_roi[cond].values, emission_roi[cond].values, 1)
    #     color = year_colors.get(y)
    #     plt.scatter(residual_roi[cond].values, emission_roi[cond].values,
    #                 marker=".", color=color, alpha=0.1, edgecolor="none")
    #     plt.plot(x, a * x + b, color=color, alpha=0.8, label=("%u" % y))
    #
    # # all years
    # a, b = np.polyfit(residual_roi.values, emission_roi.values, 1)
    # plt.plot(x, a * x + b, color="gray", alpha=0.8, label="avg. of all")
    # print("Using data of all years: Emissions [gCO2eq/kWh] = %.4f * Residual load [GW] + %.4f" % (a, b))
    #
    # # last 5 years
    # t_max = residual_roi.index[-1]
    # t_min = t_max - (4 * 365 + 366) * 24 * 3600
    # cond = ((residual_roi.index >= t_min) & (residual_roi.index < t_max))
    # a, b = np.polyfit(residual_roi[cond].values, emission_roi[cond].values, 1)
    # plt.plot(x, a * x + b, color="black", alpha=0.8, label="avg. of last 5 years")
    # print("Using data of the last 5 years: Emissions [gCO2eq/kWh] = %.4f * Residual load [GW] + %.4f" % (a, b))
    #
    # plt.legend(loc="upper left")
    # plt.xlabel("Residual load [GW]")
    # plt.ylabel("Specific emissions [gCO2eq/kWh]")
    # plt.grid(True)
    # plt.show(block=True)
    # plt.pause(.1)

    # ------------------------------------------------------------------------------------------------------------------
    # plot prices vs. residual load (colored / grouped by year)
    ixs_price = price_agora_df.index
    ixs_dem = el_gen_dem_data_df[el_gen_dem_data_df.index < EL_GEN_DEMAND_DROP_DATA_START].index
    ixs = ixs_price[ixs_price.isin(ixs_dem)]
    price_roi = price_agora_df[ixs]
    el_gen_dem_roi = get_el_gen_dem_data(el_gen_dem_data_df, ixs, scale_shift_years=0)
    residual_roi = el_gen_dem_roi[RESIDUAL_LOAD]
    min_year = pd.to_datetime(ixs.min(), unit='s').tz_localize('UTC').tz_convert(tz).year
    max_year = pd.to_datetime(ixs.max(), unit='s').tz_localize('UTC').tz_convert(tz).year
    t_u0 = pd.Timestamp("1970-01-01", tz='UTC')
    x = np.arange(-20, 101, 1)
    x_pos = np.arange(0, 101, 1)
    x_neg = np.arange(-20, 1, 1)
    for y in range(min_year, max_year + 1):
        t_min = (pd.Timestamp(year=y, month=1, day=1, tz=el_gen_tz_origin) - t_u0) // pd.Timedelta("1s")
        t_max = (pd.Timestamp(year=y + 1, month=1, day=1, tz=el_gen_tz_origin) - t_u0) // pd.Timedelta("1s")
        cond = ((residual_roi.index >= t_min) & (residual_roi.index < t_max))

        color = year_colors.get(y)
        plt.scatter(residual_roi[cond].values, price_roi[cond].values,
                    marker=".", color=color, alpha=0.1, edgecolor="none")

        a, b = np.polyfit(residual_roi[cond].values, price_roi[cond].values, 1)
        plt.plot(x, a * x + b, color=color, alpha=0.8, label=("%u" % y))

        # a, b, c = np.polyfit(residual_roi[cond].values, price_roi[cond].values, 2)
        # plt.plot(x, a * x**2 + b * x + c, color=color, alpha=0.8, label=("%u" % y))

        # cond_pos = cond & (residual_roi >= 0)
        # cond_neg = cond & (residual_roi <= 0)
        # a_pos, b_pos = np.polyfit(residual_roi[cond_pos].values, price_roi[cond_pos].values, 1)
        # plt.plot(x_pos, a_pos * x_pos + b_pos, color=color, alpha=0.8, label=("%u" % y))
        # if any(cond_neg):
        #     a_neg, b_neg = np.polyfit(residual_roi[cond_neg].values, price_roi[cond_neg].values, 1)
        #     plt.plot(x_neg, a_neg * x_neg + b_neg, color=color, alpha=0.8)

    # all years

    a, b = np.polyfit(residual_roi.values, price_roi.values, 1)
    plt.plot(x, a * x + b, color="gray", alpha=0.8, label="avg. of all")
    print("Using data of all years: Price [ct/kWh] = %.4f * Residual load [GW] + %.4f" % (a, b))

    # a, b, c = np.polyfit(residual_roi.values, price_roi.values, 2)
    # plt.plot(x, a * x**2 + b * x + c, color="gray", alpha=0.8, label="avg. of all")
    # print("Using data of all years: Price [ct/kWh] = %.4f * r^2 + %.4f * r + %.4f, r = Residual load [GW]"
    #       % (a, b, c))

    # cond_pos = (residual_roi >= 0)
    # cond_neg = (residual_roi <= 0)
    # a_pos, b_pos = np.polyfit(residual_roi[cond_pos].values, price_roi[cond_pos].values, 1)
    # plt.plot(x_pos, a_pos * x_pos + b_pos, color="gray", alpha=0.8, label="avg. of all")
    # a_neg, b_neg = np.polyfit(residual_roi[cond_neg].values, price_roi[cond_neg].values, 1)
    # plt.plot(x_neg, a_neg * x_neg + b_neg, color="gray", alpha=0.8)
    # print("Using data of all years:\n"
    #       "   Price [ct/kWh] = %.4f * Residual load [GW] + %.4f  if Residual load >= 0\n"
    #       "   Price [ct/kWh] = %.4f * Residual load [GW] + %.4f  if Residual load <= 0"
    #       % (a_pos, b_pos, a_neg, b_neg))

    # last 5 years
    t_max = residual_roi.index[-1]
    t_min = t_max - (4 * 365 + 366) * 24 * 3600
    cond = ((residual_roi.index >= t_min) & (residual_roi.index < t_max))

    a, b = np.polyfit(residual_roi[cond].values, price_roi[cond].values, 1)
    plt.plot(x, a * x + b, color="black", alpha=0.8, label="avg. of last 5 years")
    print("Using data of the last 5 years: Price [gCO2eq/kWh] = %.4f * Residual load [GW] + %.4f" % (a, b))

    # a, b, c = np.polyfit(residual_roi[cond].values, price_roi[cond].values, 2)
    # plt.plot(x, a * x**2 + b * x + c, color="black", alpha=0.8, label="avg. of last 5 years")
    # print("Using data of the last 5 years: Price [gCO2eq/kWh] = %.4f * r^2 + %.4f * r + %.4f, r = Residual load [GW]"
    #       % (a, b, c))

    # cond_pos = cond & (residual_roi >= 0)
    # cond_neg = cond & (residual_roi <= 0)
    # a_pos, b_pos = np.polyfit(residual_roi[cond_pos].values, price_roi[cond_pos].values, 1)
    # plt.plot(x_pos, a_pos * x_pos + b_pos, color="black", alpha=0.8, label="avg. of last 5 years")
    # a_neg, b_neg = np.polyfit(residual_roi[cond_neg].values, price_roi[cond_neg].values, 1)
    # plt.plot(x_neg, a_neg * x_neg + b_neg, color="black", alpha=0.8)
    # print("Using data of the last 5 years:\n"
    #       "   Price [ct/kWh] = %.4f * Residual load [GW] + %.4f  if Residual load >= 0\n"
    #       "   Price [ct/kWh] = %.4f * Residual load [GW] + %.4f  if Residual load <= 0"
    #       % (a_pos, b_pos, a_neg, b_neg))

    plt.legend(loc="upper left")
    plt.xlabel("Residual load [GW]")
    plt.ylabel("Electricity price [ct/kWh]")
    plt.grid(True)
    plt.show(block=True)
    plt.pause(.1)
    # ------------------------------------------------------------------------------------------------------------------

    print("debug here")

    # # downsample frequency data for plot
    # freq_data_plot = freq_data_df[::5]  # use every 5th value

    # extend time to see if get_transformed_data(...) works
    ts = np.arange(1420066800, 2082754800, 3600)  # 01.01.2015 - 31.12.2035 -- 1 h resolution
    tsf = np.arange(1577833200, 1735686000, 5)  # 01.01.2020 - 31.12.2024 -- 5 sec resolution
    temp_data_plot = get_temperature_data(temp_data_df, ts)
    emission_co2mon_plot = get_emission_data(emission_co2mon_df, ts, data_source="co2monitor")
    emission_agora_plot = get_emission_data(emission_agora_df, ts, data_source="Agora")
    # freq_data_plot = get_freq_data(freq_data_df, tsf)
    price_smard_plot = get_price_data(price_smard_df, ts, data_source="SMARD")
    price_agora_plot = get_price_data(price_agora_df, ts, data_source="Agora")

    # convert unix timestamps to datetimes for nicer plotting
    temp_data_plot.index = pd.to_datetime(temp_data_plot.index, unit='s').tz_localize('UTC').tz_convert(tz)
    emission_co2mon_plot.index = pd.to_datetime(emission_co2mon_plot.index, unit='s').tz_localize('UTC').tz_convert(tz)
    emission_agora_plot.index = pd.to_datetime(emission_agora_plot.index, unit='s').tz_localize('UTC').tz_convert(tz)
    # freq_data_plot.index = pd.to_datetime(freq_data_plot.index, unit='s').tz_localize('UTC').tz_convert(tz)
    price_smard_plot.index = pd.to_datetime(price_smard_plot.index, unit='s').tz_localize('UTC').tz_convert(tz)
    price_agora_plot.index = pd.to_datetime(price_agora_plot.index, unit='s').tz_localize('UTC').tz_convert(tz)

    # plot results
    fig = plt.figure()
    plt_temp = fig.add_subplot(5, 1, 1)
    plt_emission = fig.add_subplot(5, 1, 2, sharex=plt_temp)
    plt_freq = fig.add_subplot(5, 1, 3, sharex=plt_temp)
    plt_price = fig.add_subplot(5, 1, 4, sharex=plt_temp)

    plt_temp.plot(temp_data_plot.index, temp_data_plot, c="red")
    plt_temp.set_ylabel("Temperature [°C]")
    plt_temp.grid(True)

    plt_emission.plot(emission_co2mon_plot.index, emission_co2mon_plot, c="orange")
    plt_emission.plot(emission_agora_plot.index, emission_agora_plot, c="green", linestyle='dashed')
    plt_emission.set_ylabel("emission intensity [gCO2eq/kWh]")
    plt_emission.grid(True)

    # plt_freq.plot(freq_data_plot.index, freq_data_plot, c="blue")
    # plt_freq.set_ylabel("Frequency [Hz]")
    # plt_freq.grid(True)

    plt_price.plot(price_smard_plot.index, price_smard_plot, c="blue")
    plt_price.plot(price_agora_plot.index, price_agora_plot, c="red", linestyle='dashed')
    plt_price.set_ylabel("Electricity price [ct/kWh]")
    plt_price.grid(True)

    plt_price.set_xlabel("Time")
    plt.tight_layout()
    plt.show(block=True)
    plt.pause(.1)

    print("debug here")


# for debugging only
if __name__ == "__main__":
    test()
