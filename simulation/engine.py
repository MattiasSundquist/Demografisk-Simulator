import streamlit as st
import pandas as pd
from .population import Population
from utils.summary import create_yearly_summary, create_forecast_summary

class SimulationEngine:
    """Orkestrerar hela simuleringsprocessen från start till slut."""
    def __init__(self, population: Population, scb_data: pd.DataFrame, params: dict, forecast_settings: dict, scale_factor: int):
        self.population = population
        self.scb_data = scb_data
        self.params = params
        self.forecast_settings = forecast_settings
        self.scale_factor = scale_factor
        self.summary_log = []

    def run_simulation(self):
        """Kör hela simuleringsloopen från startår till slutår och returnerar resultaten."""
        start_year = self.params['simulation']['start_year']
        end_year_historical = self.params['simulation']['end_year']
        end_year_forecast = self.forecast_settings['forecast_end_year']
        total_years = end_year_forecast - start_year + 1

        progress_bar = st.progress(0, "Startar simulering...")

        for i, year in enumerate(range(start_year, end_year_forecast + 1)):
            scb_data_year = self.scb_data.loc[year]

            # 1. Åldrande och Assimilering
            self.population.perform_aging_and_assimilation(year, self.params)

            if year <= end_year_historical:
                # Kör historiska processer
                self.population.perform_deaths(scb_data_year, self.scale_factor)
                num_babies = self.population.perform_births(scb_data_year, self.scale_factor, self.params)
                self.population.perform_migration(scb_data_year, self.scale_factor, self.params)
                summary = create_yearly_summary(year, self.population.agents, scb_data_year, num_babies)
            else:
                # Kör prognos-processer
                self.population.perform_deaths_forecast(year, self.scale_factor, self.params, self.forecast_settings)
                num_babies = self.population.perform_births_forecast(year, self.scale_factor, self.params, self.forecast_settings)
                self.population.perform_migration_forecast(year, self.scale_factor, self.params, self.forecast_settings)
                summary = create_forecast_summary(year, self.population.agents, num_babies)
            
            self.summary_log.append(summary)
            progress_bar.progress((i + 1) / total_years, f"Simulerar år {year+1}...")
        
        progress_bar.empty()
        
        summary_df = pd.DataFrame(self.summary_log).set_index('Year')
        event_df = pd.DataFrame(self.population.event_log)
        
        return summary_df, self.population.agents, event_df