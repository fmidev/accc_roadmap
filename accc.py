#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:35:40 2024

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""
import numpy as np
import fair_tools
import xarray as xr
from scipy.optimize import minimize_scalar

def logistic_growth(cdr_initial, cdr_inflection_year, cumulative_cdr_target):
    """
    Logistic CDR curve with enforced initial value and realistic smooth shape.
    Optimizes peak rate L to meet cumulative CDR target.
    
    Parameters:
    cdr_initial : float
        Initial annual CDR (GtCO₂/year), positive for removals.
    cdr_inflection_year : float
        Inflection year (midpoint of S-curve).
    cumulative_cdr_target : float
        Cumulative target CDR by 2100 (GtCO₂), positive.

    Returns:
    xr.DataArray
        Annual CDR values over time.
    """
    # Time axis
    t_start = 2024.5
    t_end = 2100.5
    years = np.arange(t_start, t_end + 1)
    dt = years[1] - years[0]
    t0 = cdr_inflection_year

    def make_logistic(L):
        # Compute k so that f(t_start) = cdr_initial
        denom = L / cdr_initial - 1
        if denom <= 0:
            return np.full_like(years, np.nan)
        k = -np.log(denom) / (t_start - t0)
        f = L / (1 + np.exp(-k * (years - t0)))
        return f

    def objective(L):
        curve = make_logistic(L)
        if np.any(np.isnan(curve)):
            return 1e20  # invalid curve
        cumulative = np.sum(curve) * dt
        return (cumulative - cumulative_cdr_target) ** 2

    # Bounds for L (peak value); should be much higher than cdr_initial
    result = minimize_scalar(objective, bounds=(cdr_initial * 2, 20), method='bounded')

    if not result.success:
        raise RuntimeError('Peak rate optimization failed.')

    best_L = result.x
    final_curve = make_logistic(best_L)

    return xr.DataArray(final_curve, coords=[years], dims=['timepoints'])





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
    
    # Set parameters for the reductuins of grpss emission so that year-2030 net
    # CO2 emissions are 43% below year-2019 emissions (based on the 43%
    # reduction of GHG emissions in the C1 category of IPCC AR6 WG3 scenarios)
    # Net CO2 emissions in 2019 were 40.58 GtCO2yr-1 according to GCP
    # (https://www.icos-cp.eu/science-and-impact/global-carbon-budget/2024).
    # A 43% reduction from this results in 23.13 GtCO2yr-1 net emissions in 2030.
    # The parameters were optimized with trial and error.
    start_year = 2024.5
    annual_decrease_rate = {'CO2 FFI': 0.0979,'CO2 AFOLU': 0.0571} 
    gross_emi_target = {'CO2 FFI': 1.5,'CO2 AFOLU': 1.} 
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
        initial_emission = emi_2024[specie]

        # Formula for asymptotic approach to the target_min
        emi_gross_values = gross_emi_target[specie] + (initial_emission - gross_emi_target[specie]) * np.exp(-annual_decrease_rate[specie] * np.arange(nyears))
        
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
    
    
    

    # Parameters for novel CDR
    cdr_initial = 0.00135
    cdr_inflection_year = 2050.5
    cumulative_cdr_target = 600

    # Generate S-curve
    cdr_novel_values = -logistic_growth(
       cdr_initial,
       cdr_inflection_year,
       cumulative_cdr_target
    )


    
    emissions_accc['CDR novel']=cdr_novel_values
    
    # Add the CDR values to existing CO2 FFI emissions in f_accc
    f_accc.emissions.loc[dict(specie='CO2 FFI', timepoints=slice(2024, 2101))] += cdr_novel_values
    
    emissions_accc['CO2 net']=f_accc.emissions.loc[dict(specie='CO2 FFI')]+f_accc.emissions.loc[dict(specie='CO2 AFOLU')]
    # Select only scenario timeoints for emissions_accc
    emissions_accc['CO2 net']=emissions_accc['CO2 net'].sel(timepoints=slice(2024,2101))

    return f_accc, emissions_accc