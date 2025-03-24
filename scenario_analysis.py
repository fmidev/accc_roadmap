#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 13:01:06 2022

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""
import matplotlib.pyplot as pl
import fair_tools
import accc
from pathlib import Path

figpath = Path('figures')

opacity=0.5 # alpha parameter for the plots

# Create SSP runs for reference
f_ssps=fair_tools.createConstrainedRuns(year_end=2101, scenarios=['ssp119','ssp126'])


# Create ACCC scenario
f_accc, emissions_accc=accc.create_scenario()

# Run FaiR
f_ssps.run()
f_accc.run()

#Rebase temperatures to be relative to 1850-1900
f_ssps=fair_tools.rebase_temperature(f_ssps)
f_accc=fair_tools.rebase_temperature(f_accc)

# Calculate natural sinks
natural_sinks_annual_ssps, natural_sinks_cumulative_ssps=fair_tools.calculate_natural_sinks(f_ssps)
natural_sinks_annual_accc, natural_sinks_cumulative_accc=fair_tools.calculate_natural_sinks(f_accc)

#Calculate 95% quantiles for natural sinks
natural_sinks_annual_accc_95ci_high=natural_sinks_annual_accc.quantile(0.975, dim='config').sel(scenario='accc')
natural_sinks_annual_accc_95ci_low=natural_sinks_annual_accc.quantile(0.025, dim='config').sel(scenario='accc')

natural_sinks_cumulative_accc_95ci_high=natural_sinks_cumulative_accc.quantile(0.975, dim='config').sel(scenario='accc')
natural_sinks_cumulative_accc_95ci_low=natural_sinks_cumulative_accc.quantile(0.025, dim='config').sel(scenario='accc')


fig, ax= pl.subplots(1,2)
ax=ax.flatten()
for scenario in f_ssps.scenarios:
    f_ssps.concentration.sel(specie='CO2', scenario=scenario).mean(dim='config').plot(x='timebounds', ax=ax[0], label=scenario)
for scenario in f_accc.scenarios:
    f_accc.concentration.sel(specie='CO2', scenario=scenario).mean(dim='config').plot(x='timebounds', ax=ax[0], label=scenario)
ax[0].legend()
ax[0].set_title('CO$_2$ concentration')
ax[0].set_ylabel('ppm')
ax[0].set_xlim([2020,2100])
ax[0].set_ylim([350,475])


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
fig.savefig(figpath / 'CO2_and_surface_temperature.png', dpi=150, bbox_inches='tight')

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
fig2.savefig(figpath / 'emissions_and_sinks.png', dpi=150, bbox_inches='tight')

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
fig3.savefig(figpath / 'emissions_and_sinks_cumulative.png', dpi=150, bbox_inches='tight')






