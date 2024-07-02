# Battery aging model with Vehicle-to-Grid (V2G) / Smart Charging (V1G) use case models
Vehicle-to-Grid (V2G) and Smart Charging (V1G) use case models with a battery degradation model derived from the battery aging data published in:  
- **Dataset:** "Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell [dataset]"  
  https://publikationen.bibliothek.kit.edu/1000168959  
  DOI: [10.35097/1947](https://doi.org/10.35097/1947)
- **Description:** "Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell"  
  [Paper in review]
  

This model is described in Chapter 7 of the dissertation:  
Matthias Luh: *"Robust electricity grids through intelligent,  highly efficient bidirectional charging systems  for electric vehicles"*

## File Overview and explanation

- **Use case simulations using the battery degradation model:**
  - **use_case_model_EV_modular_v01.py:** *"modular and flexibly extendable modeling framework [...] to simulate different smart and bidirectional EV charging use cases and analyze the resulting
battery aging"* [Chapter 7.3.2 of the dissertation]  
    &rarr; this is probably what you came for, if you want to simulate battery degradation in EVs
  - **use_case_model_EV_modular_v01_fast.py:** faster version of *use_case_model_EV_modular_v01.py*, but does not return log data (voltage/current/power/temperature profile of the cell, power/price/emissions/residual load/... of the grid)  
    Suggestion: Start with *use_case_model_EV_modular_v01.py* and only use *use_case_model_EV_modular_v01_fast.py* when you know what you do.
  - **use_case_model_001_test.py:** a simple charge-discharge test using the aging model
  - **use_case_model_002_energy_test.py:** a simple repeated charge-discharge test using different power values with the aging model
  - **use_case_model_003_cc_cv_test.py:** various CC-CV charging/discharging and ambient temperature variation tests
  - **use_case_model_004_cycling_experiment_test.py:** preliminary tests for a simulation of the battery aging experiment conducted for the dissertation
  - **use_case_model_005_cycling_experiment.py:** simulation of the battery aging experiment conducted for the dissertation
  - **use_case_model_006_driving_test.py:** simple simulation of an EV driving
- **Battery degradation (capacity fade) model:** 
  - **bat_model_v01.py:** Battery degradation model (capacity fade) derived from experimental cell aging test data (see dissertation, Chapter 7)  
    &rarr; this is probably what you came for, if you want to simulate battery degradation in your own application
  - **bat_model_v01_fast.py:** faster model, but does not return log data (voltage, current, temperature, ...)  
    Suggestion: Start with *bat_model_v01.py* and only use *bat_model_v01_fast.py* when you know what you do.
- **Additional scripts and files:**
  - **plot_results_use_case_model_EV_modular_v01.py:** plot the capacity fade over time for all use case simulations
  - **result_plot.py:** helper functions used to plot results
  - **driving_profile_helper.py:** helper functions to generate the scenario's driving day types used in use_case_model_EV_modular scripts
  - **input_data_helper.py:** helper functions to import and process input data (temperature, electricity data, ...)
      &rarr; see *"Required input data"* below!
  - **scenario_helper.py:** helper functions and definitions for the scenarios in use_case_model_EV_modular_v01.py
  - **wltp_profiles.py:** cell power profiles derived based on the WLTP speed profile (WLTC Class 3b)
  - **logger.py:** used to log (debug) information, warnings, and errors to the console and a log text file 
  - **requirements.txt:** Required libraries (and version with which they were successfully tested)

## Required input data

If you use the *use_case_model_EV_modular* scripts, you need to download data that is used as an input.
The data is NOT included in this repository because of the license (data is used, but not owned, so you need to download it yourself).


