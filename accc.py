#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:35:40 2024

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""
import numpy as np
import fair_tools
import xarray as xr

def logistic_growth(cdr_initial, relative_peak_rate, growth_rate, cdr_inflection_year, cumulative_cdr_target):
    """
    Logistic growth function adjusted to meet a cumulative CDR target, with relative rates.

    Parameters:
    cdr_initial : float
        Initial annual CDR rate (Gt CO₂/year, negative for sinks).
    relative_peak_rate : float
        Relative rate multiplier at the inflection point (e.g., 10 means 10x `cdr_initial`).
    growth_rate : float
        Growth rate, controls the steepness of the curve.
    cdr_inflection_year : float
        Year where growth is fastest (inflection point).
    cumulative_cdr_target : float
        Total cumulative CDR by the end year (Gt CO₂, negative for sinks).

    Returns:
    xr.DataArray
        Annual CDR rates adjusted to meet the cumulative target.
    """
    cdr_start_year = 2024.5
    cdr_end_year = 2100.5
    years = np.arange(cdr_start_year, cdr_end_year + 1)

    # Define max rate as a multiple of the initial rate
    cdr_peak_rate = relative_peak_rate * cdr_initial

    # Logistic growth equation
    logistic = cdr_peak_rate / (1 + (cdr_peak_rate - cdr_initial) / cdr_initial * np.exp(-growth_rate * (years - cdr_inflection_year)))

    # Adjust to meet cumulative target
    cumulative_cdr = np.cumsum(logistic) * (years[1] - years[0])  # Integration over years
    scale_factor = cumulative_cdr_target / cumulative_cdr[-1]  # Scaling factor
    logistic_scaled = logistic * scale_factor  # Scale the curve

    # Convert to xarray DataArray
    logistic_scaled = xr.DataArray(
        logistic_scaled,
        coords=[years],
        dims=["timepoints"]
    )
    return logistic_scaled





def create_scenario():

    f_accc=fair_tools.createConstrainedRuns(year_end=2101, scenarios=['ssp126'])
    scenario_mapping = {'ssp126': 'accc'}
    
    emissions_accc=dict() # A dict to hold gross and net emissions
    
    # Update scenario names in the Fair instance
    f_accc = fair_tools.update_scenario_names(f_accc, scenario_mapping)
    
    # Net emissions from GCP and add gross removal estimated by State of the CDR report
    cdr_2024=-2.2
    
    # FFI emissions from global carbon project
    emi_2024={'CO2 FFI': 37.4,
              'CO2 AFOLU': 4.2-cdr_2024 }
    
    # Set parameters
    start_year = 2024.5
    annual_decrease_rate = 0.055
    cdr_years = f_accc.emissions.timepoints.loc[start_year:2101]  # Use consistent timepoints from f_accc
    nyears = len(cdr_years)  # Number of years for ACCC scenario
    
    # Initialize emi_gross_accc as an xarray DataArray with the same dimensions as f_accc.emissions
    emi_gross_accc = xr.DataArray(
        np.zeros((nyears, len(emi_2024.keys()))),
        coords={
            "timepoints": cdr_years,
            "specie": list(emi_2024.keys())
        },
        dims=["timepoints", "specie"]
    )
    
    # Insert annual reductions into FFI and AFOLU emissions for each specie
    for specie in emi_2024.keys():
        # Calculate the emissions decrease over the years as an array
        emi_gross_values = emi_2024[specie] * (1 - annual_decrease_rate) ** np.arange(nyears)
        
        # Store the emissions in emi_gross_accc for the corresponding specie
        emi_gross_accc.loc[dict(specie=specie)] = emi_gross_values[:]
    
        # Assign the calculated values back to f_accc.emissions for the given specie and timepoints
        f_accc.emissions.loc[dict(specie=specie, timepoints=slice(2024, 2101))] = emi_gross_accc.sel(specie=specie)
    
    
        emissions_accc[specie+' gross']=emi_gross_accc.loc[dict(specie=specie)]
    
    
    # Make ACCC scenario for land-based CDR
    
    # Set parameters
    cdr_2024 = -2.2  # initial CDR value at 2024.5 (negative for sink)
    cdr_2050 = -10   # CDR value at 2050
    cdr_start_year = 2024.5
    cdr_increase_end_year = 2050
    cdr_decrease_rate = 0.01  # 1% decrease per year
    
    # Calculate the number of years for exponential increase phase
    years_increase_phase = cdr_increase_end_year - cdr_start_year
    
    # Calculate the required exponential growth rate to reach exactly -10 by 2050
    growth_rate = np.log(cdr_2050 / cdr_2024) / years_increase_phase
    
    # Use timepoints directly from f_accc.emissions to ensure consistency
    cdr_years = f_accc.emissions.timepoints.loc[2024:2101]  # Range from 2024 to 2101
    nyears = len(cdr_years)  # Ensure correct length for target slice
    
    # Create the CDR array as an xarray DataArray with metadata for years
    cdr_values = xr.DataArray(
        np.zeros(nyears), 
        coords=[cdr_years], 
        dims=["timepoints"]
    )
    
    # Exponential increase until 2049
    increase_years = cdr_years[cdr_years < cdr_increase_end_year]  # Up to 2049.5
    cdr_values.loc[dict(timepoints=increase_years)] = (
        cdr_2024 * np.exp(growth_rate * (increase_years - cdr_start_year))
    )
    
    # Set the value at 2050 explicitly
    cdr_values.loc[dict(timepoints=2050.5)] = cdr_2050
    
    # 1% annual decrease after 2050
    decrease_years = cdr_years[(cdr_years > cdr_increase_end_year)]
    cdr_values.loc[dict(timepoints=decrease_years)] = (
        cdr_2050 * (1 - cdr_decrease_rate) ** np.arange(1, len(decrease_years) + 1)
    )
    
    
    emissions_accc['CDR land-based']=cdr_values
    
    # Update f_accc.emissions by adding cdr_values without in-place operations
    f_accc.emissions.loc[dict(specie="CO2 AFOLU", timepoints=slice(2024, 2101))] = (
        f_accc.emissions.sel(specie="CO2 AFOLU", timepoints=slice(2024, 2101)) + cdr_values
    )
    
    
    

    # Parameters
    cdr_initial = -1.35e-3  # Initial annual CDR rate (Gt CO₂/year, negative for sinks)
    relative_peak_rate = 15.0  # Peak rate is 15x the initial rate
    growth_rate = 0.15         # Logistic growth rate
    cdr_inflection_year = 2030.5  # Year where growth is fastest
    cumulative_cdr_target = -600  # Total cumulative CDR by 2100 (Gt CO₂, negative for sinks)

    # Generate S-curve
    cdr_other_values = logistic_growth(cdr_initial, relative_peak_rate, growth_rate, cdr_inflection_year, cumulative_cdr_target)
    
    emissions_accc['CDR novel']=cdr_other_values
    
    # Add the CDR values to existing CO2 FFI emissions in f_accc
    f_accc.emissions.loc[dict(specie='CO2 FFI', timepoints=slice(2024, 2101))] += cdr_other_values
    
    emissions_accc['CO2 net']=f_accc.emissions.loc[dict(specie='CO2 FFI')]+f_accc.emissions.loc[dict(specie='CO2 AFOLU')]

    return f_accc, emissions_accc