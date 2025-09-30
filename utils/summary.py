import numpy as np

# Denna konstant behövs här för att definiera grupperna korrekt
ASSIMILATION_THRESHOLD = 0.999

def create_yearly_summary(year, population_df, scb_data_year, num_new_babies):
    """Skapar en sammanfattning för ett historiskt år med jämförelse mot SCB-data."""
    final_live_pop = population_df[population_df['is_alive']]
    first_generation_mask = final_live_pop['is_immigrant'] == True
    swedish_background_mask = final_live_pop['swedish_share'] >= ASSIMILATION_THRESHOLD
    later_generations_mask = (~first_generation_mask) & (~swedish_background_mask)
    
    return {
        'Year': year, 
        'Simulated_Population': len(final_live_pop), 
        'Actual_Population_SCB': scb_data_year["Folkmängd 31 december Population on 31 December"], 
        'Swedish_Background': swedish_background_mask.sum(), 
        'Later_Generations': later_generations_mask.sum(), 
        'First_Generation': first_generation_mask.sum(), 
        'Simulated_Births': num_new_babies, 
        'Actual_Births_SCB': scb_data_year["Födda Births"]
    }

def create_forecast_summary(year, population_df, num_new_babies):
    """Skapar en sammanfattning för ett prognosår utan SCB-data."""
    final_live_pop = population_df[population_df['is_alive']]
    first_generation_mask = final_live_pop['is_immigrant'] == True
    swedish_background_mask = final_live_pop['swedish_share'] >= ASSIMILATION_THRESHOLD
    later_generations_mask = (~first_generation_mask) & (~swedish_background_mask)
    
    return {
        'Year': year, 
        'Simulated_Population': len(final_live_pop), 
        'Actual_Population_SCB': np.nan, 
        'Swedish_Background': swedish_background_mask.sum(), 
        'Later_Generations': later_generations_mask.sum(), 
        'First_Generation': first_generation_mask.sum(), 
        'Simulated_Births': num_new_babies, 
        'Actual_Births_SCB': np.nan
    }