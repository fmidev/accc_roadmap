#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 13:01:06 2022

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""
import numpy as np
import matplotlib.pyplot as pl
import fair_tools
import accc
from pathlib import Path
import pandas as pd
import xarray as xr

fig_path = Path('figures')
output_data_path = Path('output')

opacity=0.5 # alpha parameter for the plots

include_ssps=True

if include_ssps:
    # Create SSP runs for reference
    f_ssps=fair_tools.createConstrainedRuns(year_end=2101, scenarios=['ssp119','ssp126'])
    f_ssps.run()

    #Rebase temperatures to be relative to 1850-1900
    f_ssps=fair_tools.rebase_temperature(f_ssps)

    # Calculate natural sinks
    natural_sinks_annual_ssps, natural_sinks_cumulative_ssps=fair_tools.calculate_natural_sinks(f_ssps)

# Create ACCC scenario
f_accc, emissions_accc=accc.create_scenario()

# Run FaiR
f_accc.run()

#Rebase temperatures to be relative to 1850-1900
f_accc=fair_tools.rebase_temperature(f_accc)

# Calculate natural sinks
natural_sinks_annual_accc, natural_sinks_cumulative_accc=fair_tools.calculate_natural_sinks(f_accc)

#Calculate 95% quantiles for natural sinks
natural_sinks_annual_accc_95ci_high=natural_sinks_annual_accc.quantile(0.975, dim='config').sel(scenario='accc')
natural_sinks_annual_accc_95ci_low=natural_sinks_annual_accc.quantile(0.025, dim='config').sel(scenario='accc')

natural_sinks_cumulative_accc_95ci_high=natural_sinks_cumulative_accc.quantile(0.975, dim='config').sel(scenario='accc')
natural_sinks_cumulative_accc_95ci_low=natural_sinks_cumulative_accc.quantile(0.025, dim='config').sel(scenario='accc')



# %%  Figure on sat and CO2
fig, ax= pl.subplots(1,2)
ax=ax.flatten()
if include_ssps:
    for scenario in f_ssps.scenarios:
        f_ssps.concentration.sel(specie='CO2', scenario=scenario).mean(dim='config').plot(x='timebounds', ax=ax[0], label=scenario)
for scenario in f_accc.scenarios:
    f_accc.concentration.sel(specie='CO2', scenario=scenario).mean(dim='config').plot(x='timebounds', ax=ax[0], label=scenario)
ax[0].legend()
ax[0].set_title('CO$_2$ concentration')
ax[0].set_ylabel('ppm')
ax[0].set_xlim([2020,2100])
ax[0].set_ylim([350,475])
ax[0].set_yticks(np.arange(350, 476, 25))

if include_ssps:
    for scenario in f_ssps.scenarios:
        f_ssps.temperature.sel(layer=0, scenario=scenario).mean(dim='config').plot(x='timebounds', ax=ax[1], label=scenario)
for scenario in f_accc.scenarios:
    f_accc.temperature.sel(layer=0, scenario=scenario).mean(dim='config').plot(x='timebounds', ax=ax[1], label=scenario)
ax[1].legend()
ax[1].set_title('Surface temperature\nrelative to 1850-1900')
ax[1].set_ylabel('°C')
ax[1].set_xlim([2024,2100])
ax[1].set_ylim([1,2])

fig.tight_layout()
fig.savefig(fig_path / 'CO2_and_surface_temperature.png', dpi=150, bbox_inches='tight')


# %%  Figure on annual emissions and sinks
fig2, ax2= pl.subplots(1,1)

emissions_accc['CO2 FFI gross'].plot(ax=ax2,label='Gross FFI CO$_2$ emissions')
emissions_accc['CO2 AFOLU gross'].plot(ax=ax2,label='Gross AFOLU CO$_2$ emissions')


emissions_accc['CDR land-based'].plot(ax=ax2,label='Land-based CDR')
emissions_accc['CDR novel'].plot(ax=ax2,label='Novel CDR')
f_accc.emissions.sel(specie='CO2',config=1234).plot(ax=ax2, label='Net CO$_2$ emissions')
natural_sinks_annual_accc.mean(dim='config').plot(ax=ax2, label='Natural sinks')
ax2.fill_between(natural_sinks_annual_accc_95ci_low.timebounds, natural_sinks_annual_accc_95ci_low, natural_sinks_annual_accc_95ci_high, alpha=opacity)

ax2.set_xlim([2024,2100])
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax2.set_ylabel('Gt CO$_2$ yr$^-1$')
ax2.set_xlabel('')
ax2.set_title('Emissions and sinks')
ax2.grid(which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
fig2.savefig(fig_path / 'emissions_and_sinks.png', dpi=150, bbox_inches='tight')

# %%  Figure on cumulative emissions and sinks
fig3, ax3= pl.subplots(1,1)

emissions_accc['CO2 FFI gross'].cumsum().plot(ax=ax3,label='Gross FFI CO$_2$ emissions')
emissions_accc['CO2 AFOLU gross'].cumsum().plot(ax=ax3,label='Gross AFOLU CO$_2$ emissions')


emissions_accc['CDR land-based'].cumsum().plot(ax=ax3,label='Land-based CDR')
emissions_accc['CDR novel'].cumsum().plot(ax=ax3,label='Novel CDR')
f_accc.emissions.sel(specie='CO2',config=1234, timepoints=slice(2024,2102)).cumsum().plot(ax=ax3, label='Net CO$_2$ emissions')
natural_sinks_cumulative_accc.sel(timebounds=slice(2024,2102)).mean(dim='config').plot(ax=ax3, label='Natural sinks')
ax3.fill_between(natural_sinks_cumulative_accc_95ci_low.timebounds, natural_sinks_cumulative_accc_95ci_low, natural_sinks_cumulative_accc_95ci_high, alpha=opacity)

ax3.set_xlim([2024,2100])
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax3.set_ylabel('Gt CO$_2$')
ax3.set_xlabel('')
ax3.set_title('Cumulative emissions and sinks')
ax3.grid(which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
fig3.savefig(fig_path / 'emissions_and_sinks_cumulative.png', dpi=150, bbox_inches='tight')

# %%  Figure on annual emissions and sinks aggregated
fig4, ax4= pl.subplots(1,1)

(emissions_accc['CO2 FFI gross']+emissions_accc['CO2 AFOLU gross']).plot(ax=ax4,label='Gross CO$_2$ emissions')
(emissions_accc['CDR land-based']+emissions_accc['CDR novel']).plot(ax=ax4,label='CDR')

f_accc.emissions.sel(specie='CO2',config=1234).plot(ax=ax4, label='Net CO$_2$ emissions')
natural_sinks_annual_accc.mean(dim='config').plot(ax=ax4, label='Natural sinks')
ax4.fill_between(natural_sinks_annual_accc_95ci_low.timebounds, natural_sinks_annual_accc_95ci_low, natural_sinks_annual_accc_95ci_high, alpha=opacity, color='red')

ax4.set_xlim([2024,2100])
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax4.set_ylabel('Gt CO$_2$ yr$^-1$')
ax4.set_xlabel('')
ax4.set_title('Emissions and sinks')
ax4.grid(which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
fig4.savefig(fig_path / 'aggregatged_emissions_and_sinks.png', dpi=150, bbox_inches='tight')

# %%  Save data into Excel files

# Assuming `temp_data` is your DataArray from the previous operation
sat_accc_data = f_accc.temperature.sel(layer=0, scenario=scenario).loc[dict(timebounds=slice(2024, 2101))]

# Calculate the mean of neighboring time points
sat_annual_mean = (sat_accc_data[:-1,:].values + sat_accc_data[1:,:].values) / 2

# Assign new timebounds to the result
sat_accc = xr.DataArray(sat_annual_mean, coords={'year': np.arange(2024,2101), 'config':sat_accc_data.config}, dims=['year', 'config'])
    

# Assuming `temp_data` is your DataArray from the previous operation
co2_accc_data = f_accc.concentration.sel(specie='CO2', scenario=scenario).loc[dict(timebounds=slice(2024, 2101))]

# Calculate the mean of neighboring time points
co2_annual_mean = (co2_accc_data[:-1,:].values + co2_accc_data[1:,:].values) / 2

# Assign new timebounds to the result
co2_accc = xr.DataArray(co2_annual_mean, coords={'year': np.arange(2024,2101), 'config':co2_accc_data.config}, dims=['year', 'config'])


# Process data into Pandas Dataframe
output_data=pd.DataFrame(index=np.arange(2024,2101))
# Name the index as 'year'
output_data.index.name = 'year'

for specie, data in emissions_accc.items():
    # Select the relevant slice
    if 'timepoints' in data.coords:
        sliced_data = data.loc[dict(timepoints=slice(2024, 2101))]
        
        # If the data has a 'config' dimension, select the first config (index 0)
        if 'config' in sliced_data.dims:
            sliced_data = sliced_data.sel(config=1234)
        
        # Add the result to the DataFrame
        output_data[specie] = sliced_data.values
    
output_data['sat mean']=sat_accc.mean(dim='config')
output_data['sat 95%_CI_low']=sat_accc.quantile(0.025, dim='config')
output_data['sat 95%_CI_high']=sat_accc.quantile(0.975, dim='config')

output_data['CO2 mean']=co2_accc.mean(dim='config')
output_data['CO2 95%_CI_low']=co2_accc.quantile(0.025, dim='config')
output_data['CO2 95%_CI_high']=co2_accc.quantile(0.975, dim='config')

excel_filename = output_data_path / 'accc_roadmap.xlsx'

# Create an Excel writer using the context manager
with pd.ExcelWriter(excel_filename) as writer:
    
    output_data[emissions_accc.keys()].to_excel(writer, sheet_name='Emissions and sinks', index=True)

    # Save temperature-related data in a single sheet
    temperature_data = output_data[['sat mean', 'sat 95%_CI_low', 'sat 95%_CI_high']]
    temperature_data.to_excel(writer, sheet_name='Temperature', index=True)

    # Save CO2 concentration-related data in a single sheet
    co2_data = output_data[['CO2 mean', 'CO2 95%_CI_low', 'CO2 95%_CI_high']]
    co2_data.to_excel(writer, sheet_name='CO2 concentration', index=True)


# Print important scenario metrics
print('CO2 concentration in 2100 = ', output_data['CO2 mean'].loc[2100].round(2))


