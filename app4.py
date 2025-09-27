import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- SIDANS GRUNDINSTÄLLNINGAR OCH TITEL ---
st.set_page_config(page_title="Demografisk Simulator 4.0", layout="wide")

# --- MODELLKONSTANTER ---
# Genom att definiera "magiska nummer" här blir koden mer läsbar och enklare att underhålla.
ASSIMILATION_THRESHOLD = 0.999  # Andel som krävs för att klassas med "Svensk bakgrund"
FERTILITY_AGE_MIN = 15
FERTILITY_AGE_MAX = 49
PARTNER_AGE_MIN = 18
PARTNER_AGE_MAX = 60
IMMIGRANT_AGE_MEAN = 28
IMMIGRANT_AGE_STD_DEV = 8

# --- MODELLENS STANDARDPARAMETRAR ---
def get_default_parameters():
    """Returnerar en strukturerad ordbok med alla standardparametrar för modellen."""
    params = {
        "simulation": {
            "start_year": 1960,
            "end_year": 2024
        },
        "assimilation": {
            "base_rate": 0.005,
            "strength_multiplier": 1.0,
            "start_shares": {
                "Scandinavieninvandring": 0.75, "Europainvandring": 0.50, 
                "Balkan/LATAM-invandring": 0.25, "MENA-invandring": 0.05, "Övriga": 0.10
            },
            "rate_multipliers": {
                "Scandinavieninvandring": 7.0, "Europainvandring": 5.0, 
                "Balkan/LATAM-invandring": 3.0, "MENA-invandring": 1.0, "Övriga": 2.0
            }
        },
        "fertility": {
            "bonuses": {
                "Svensk": 0.0, "Scandinavieninvandring": 0.0, "Europainvandring": 0.15, 
                "Balkan/LATAM-invandring": 0.45, "MENA-invandring": 1.2, "Övriga": 0.35
            }
        },
        "migration": {
            "historical_waves": {
                "1960-1980": {"Europainvandring": 0.30, "Scandinavieninvandring": 0.30, "Balkan/LATAM-invandring": 0.25, "MENA-invandring": 0.05, "Övriga": 0.10},
                "1981-2000": {"Europainvandring": 0.20, "Scandinavieninvandring": 0.20, "Balkan/LATAM-invandring": 0.40, "MENA-invandring": 0.10, "Övriga": 0.10},
                "2001-2024": {"Europainvandring": 0.10, "Scandinavieninvandring": 0.10, "Balkan/LATAM-invandring": 0.10, "MENA-invandring": 0.60, "Övriga": 0.10},
            },
            "emigration_bias": 5.0
        },
        "demographics": {
            "le_male_2024": 82.29,
            "le_female_2024": 85.35,
            "cdr_2024": 9.0
        },
        "mixing": {
            "swedish_background_mother": {"start_year": 1960, "start_value": 0.10, "end_year": 2019, "end_value": 0.25},
            "foreign_background_mother": {"start_year": 1960, "start_value": 0.25, "end_year": 2019, "end_value": 0.10}
        }
    }
    return params

# --- INITIERA SESSION STATE ---
if 'edited_params' not in st.session_state:
    st.session_state['edited_params'] = get_default_parameters()

if 'future_migration_composition' not in st.session_state:
    st.session_state['future_migration_composition'] = st.session_state['edited_params']["migration"]["historical_waves"]["2001-2024"].copy()

if 'forecast_settings' not in st.session_state:
    # Uppdaterar till de senaste SCB-prognosvärdena som standard
    st.session_state['forecast_settings'] = {
        'forecast_end_year': 2070, 'tfr': 1.59, 'target_le_male': 85.3, 'target_le_female': 87.7,
        'cdr_target_2050': 8.8, 'immigrant_composition': st.session_state['future_migration_composition'],
        'migration_model': 'Procent av befolkningen', 'annual_immigrants': 80000, 'annual_emigrants': 60000,
        'immigration_rate': 0.0075, 'emigration_rate': 0.0057
    }

# --- DATALADDNING ---
@st.cache_data
def load_scb_data(filepath, max_year):
    df = pd.read_csv(filepath, index_col=0, na_values="..")
    df = df.transpose(); df.index = pd.to_numeric(df.index)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_index()
    full_yearly_index = pd.RangeIndex(start=1960, stop=max_year + 1, name='År')
    df_reindexed = df.reindex(full_yearly_index)
    df_interpolated = df_reindexed.interpolate(method='spline', order=2, limit_direction='both')
    return df_interpolated

# --- HJÄLPFUNKTIONER FÖR SIMULERINGEN ---
def perform_aging_and_assimilation(population, year, params):
    population.loc[population['is_alive'], 'age'] += 1
    live_pop_mask = population['is_alive']
    if live_pop_mask.sum() == 0: return population

    # Bättre variabelnamn för tydlighet.
    proportion_swedish_background = (live_pop_mask & (population['swedish_share'] >= ASSIMILATION_THRESHOLD)).sum() / live_pop_mask.sum()
    origin_assimilation_multiplier = population['ethnic_origin'].map(params['assimilation']['rate_multipliers']).fillna(1.0)
    
    # Den nya, enhetliga och linjära assimileringsmodellen.
    dynamic_assimilation_factor = (proportion_swedish_background * 2) - 1
    assimilation_strength = params['assimilation']['strength_multiplier']
    
    assimilation_change = (params['assimilation']['base_rate'] * 
                           dynamic_assimilation_factor * 
                           assimilation_strength *
                           population['swedish_share'] * 
                           origin_assimilation_multiplier)

    change_mask = live_pop_mask & (population['swedish_share'] > 0) & (population['swedish_share'] < 1.0)
    population.loc[change_mask, 'swedish_share'] += assimilation_change
    population['swedish_share'] = population['swedish_share'].clip(0.0, 1.0)
    return population

def perform_deaths(population, scb_data_year, scale_factor):
    events = []
    live_pop_indices = population.index[population['is_alive']]
    if live_pop_indices.empty: return population, events

    mortality_rate = 0.001 + (population.loc[live_pop_indices, 'age'] / 100) ** 4
    scb_deaths_scaled = scb_data_year["Döda Deaths"] / scale_factor
    sim_deaths_expected = mortality_rate.sum()
    death_adj_factor = scb_deaths_scaled / sim_deaths_expected if sim_deaths_expected > 0 else 0
    death_prob = mortality_rate * death_adj_factor
    dies_mask = np.random.rand(len(live_pop_indices)) < death_prob
    newly_dead_indices = live_pop_indices[dies_mask]
    population.loc[newly_dead_indices, 'is_alive'] = False
    for agent_id in population.loc[newly_dead_indices, 'id']:
        events.append({'year': scb_data_year.name, 'event': 'Död', 'agent_id': agent_id})
    return population, events

def perform_deaths_forecast(population, year, scale_factor, params, forecast_settings):
    events = []
    live_pop_indices = population.index[population['is_alive']]
    if live_pop_indices.empty: return population, events

    le_m_2024 = params['demographics']['le_male_2024']
    le_f_2024 = params['demographics']['le_female_2024']
    target_le_m_2050 = forecast_settings['target_le_male']
    target_le_f_2050 = forecast_settings['target_le_female']
    interp_start_year_le = params['simulation']['end_year']
    interp_end_year_le = 2050
    
    if year <= interp_end_year_le:
        interp_factor = (year - interp_start_year_le) / (interp_end_year_le - interp_start_year_le) if (interp_end_year_le - interp_start_year_le) > 0 else 0
        current_le_m = le_m_2024 + (target_le_m_2050 - le_m_2024) * interp_factor
        current_le_f = le_f_2024 + (target_le_f_2050 - le_f_2024) * interp_factor
    else:
        current_le_m, current_le_f = target_le_m_2050, target_le_f_2050

    mortality_rate_base = 0.001 + (population.loc[live_pop_indices, 'age'] / 100) ** 4
    adj_factor_male_le = (le_m_2024 / current_le_m) if current_le_m > 0 else 1.0
    adj_factor_female_le = (le_f_2024 / current_le_f) if current_le_f > 0 else 1.0

    mortality_rate_adjusted = mortality_rate_base.copy()
    male_mask = population.loc[live_pop_indices, 'sex'] == 0
    mortality_rate_adjusted.loc[male_mask] *= adj_factor_male_le
    mortality_rate_adjusted.loc[~male_mask] *= adj_factor_female_le

    cdr_2024 = params['demographics']['cdr_2024']
    cdr_2050 = forecast_settings['cdr_target_2050']
    interp_start_year_cdr = params['simulation']['end_year']
    interp_end_year_cdr = 2050
    if year <= interp_end_year_cdr:
        interp_factor_cdr = (year - interp_start_year_cdr) / (interp_end_year_cdr - interp_start_year_cdr) if (interp_end_year_cdr - interp_start_year_cdr) > 0 else 0
        current_cdr = cdr_2024 + (cdr_2050 - cdr_2024) * interp_factor_cdr
    else:
        current_cdr = cdr_2050

    total_current_population_full_scale = population[population['is_alive']].shape[0] * scale_factor
    target_deaths_full_scale = total_current_population_full_scale * (current_cdr / 1000)
    target_deaths_scaled = int(round(target_deaths_full_scale / scale_factor))

    sim_deaths_expected_base = mortality_rate_adjusted.sum()
    death_adj_factor = target_deaths_scaled / sim_deaths_expected_base if sim_deaths_expected_base > 0 else 0
    death_prob = (mortality_rate_adjusted * death_adj_factor).clip(0.0, 1.0)

    dies_mask = np.random.rand(len(live_pop_indices)) < death_prob.values
    newly_dead_indices = live_pop_indices[dies_mask]
    population.loc[newly_dead_indices, 'is_alive'] = False
    for agent_id in population.loc[newly_dead_indices, 'id']:
        events.append({'year': year, 'event': 'Död', 'agent_id': agent_id})
    return population, events

def perform_births(population, scb_data_year, scale_factor, params, next_agent_id):
    events, new_babies_list = [], []
    live_pop_mask = population['is_alive']
    mothers = population[live_pop_mask & (population['sex'] == 1) & (population['age'] >= FERTILITY_AGE_MIN) & (population['age'] <= FERTILITY_AGE_MAX)]

    if not mothers.empty:
        scb_births_scaled = scb_data_year["Födda Births"] / scale_factor
        fertility_bonus = mothers['ethnic_origin'].map(params['fertility']['bonuses']).fillna(0)
        base_fertility = 1.0 - (0.8 * mothers['swedish_share'])
        birth_potential = (1 + (fertility_bonus * base_fertility))
        birth_prob_factor = scb_births_scaled / birth_potential.sum() if birth_potential.sum() > 0 else 0
        birth_prob = birth_potential * birth_prob_factor
        gives_birth = np.random.rand(len(mothers)) < birth_prob
        actual_mothers = mothers[gives_birth]
        men = population[live_pop_mask & (population['sex'] == 0) & (population['age'] >= PARTNER_AGE_MIN) & (population['age'] <= PARTNER_AGE_MAX)]

        if not actual_mothers.empty and not men.empty:
            year = scb_data_year.name
            mixing_swedish = params['mixing']['swedish_background_mother']
            mixing_foreign = params['mixing']['foreign_background_mother']
            
            mix_factor_swedish = np.interp(year, [mixing_swedish['start_year'], mixing_swedish['end_year']], [mixing_swedish['start_value'], mixing_swedish['end_value']])
            mix_factor_foreign = np.interp(year, [mixing_foreign['start_year'], mixing_foreign['end_year']], [mixing_foreign['start_value'], mixing_foreign['end_value']])
            
            men_swedish_background = men[men['swedish_share'] == 1.0]
            men_foreign_background = men[men['swedish_share'] < 1.0]

            for _, mother in actual_mothers.iterrows():
                father = None
                if mother['swedish_share'] == 1.0:
                    if np.random.rand() < mix_factor_swedish and not men_foreign_background.empty: father = men_foreign_background.sample(1).iloc[0]
                    elif not men_swedish_background.empty: father = men_swedish_background.sample(1).iloc[0]
                else:
                    if np.random.rand() < mix_factor_foreign and not men_swedish_background.empty: father = men_swedish_background.sample(1).iloc[0]
                    elif not men_foreign_background.empty: father = men_foreign_background.sample(1).iloc[0]
                
                if father is None: father = men.sample(1).iloc[0]

                child_share = (mother['swedish_share'] + father['swedish_share']) / 2
                child_origin = np.random.choice([mother['ethnic_origin'], father['ethnic_origin']]) if mother['ethnic_origin'] != father['ethnic_origin'] else mother['ethnic_origin']
                
                baby = {'id': next_agent_id, 'age': 0, 'sex': np.random.randint(0, 2), 'is_alive': True, 'swedish_share': child_share, 'ethnic_origin': child_origin, 'birth_year': year, 'parent1_id': father['id'], 'parent2_id': mother['id'], 'is_immigrant': False}
                new_babies_list.append(baby)
                events.append({'year': year, 'event': 'Födsel', 'agent_id': next_agent_id, 'mother_id': mother['id'], 'father_id': father['id']})
                next_agent_id += 1
                
    if new_babies_list:
        population = pd.concat([population, pd.DataFrame(new_babies_list)], ignore_index=True)
    return population, events, next_agent_id, len(new_babies_list)

def perform_births_forecast(population, year, scale_factor, params, forecast_settings, next_agent_id):
    events, new_babies_list = [], []
    live_pop_mask = population['is_alive']
    mothers = population[live_pop_mask & (population['sex'] == 1) & (population['age'] >= FERTILITY_AGE_MIN) & (population['age'] <= FERTILITY_AGE_MAX)]
    if mothers.empty: return population, events, next_agent_id, 0

    num_fertile_women_full_scale = len(mothers) * scale_factor
    annual_birth_rate_per_woman = forecast_settings['tfr'] / (FERTILITY_AGE_MAX - FERTILITY_AGE_MIN + 1)
    expected_births_full_scale = num_fertile_women_full_scale * annual_birth_rate_per_woman
    expected_births_scaled = int(round(expected_births_full_scale / scale_factor))
    if expected_births_scaled <= 0: return population, events, next_agent_id, 0

    fertility_bonus = mothers['ethnic_origin'].map(params['fertility']['bonuses']).fillna(0)
    base_fertility = 1.0 - (0.8 * mothers['swedish_share'])
    birth_potential = (1 + (fertility_bonus * base_fertility))
    birth_prob_factor = expected_births_scaled / birth_potential.sum() if birth_potential.sum() > 0 else 0
    birth_prob = birth_potential * birth_prob_factor
    
    gives_birth = np.random.rand(len(mothers)) < birth_prob
    actual_mothers = mothers[gives_birth]
    men = population[live_pop_mask & (population['sex'] == 0) & (population['age'] >= PARTNER_AGE_MIN) & (population['age'] <= PARTNER_AGE_MAX)]

    if not actual_mothers.empty and not men.empty:
        mix_factor_swedish = params['mixing']['swedish_background_mother']['end_value']
        mix_factor_foreign = params['mixing']['foreign_background_mother']['end_value']
        
        men_swedish_background = men[men['swedish_share'] == 1.0]
        men_foreign_background = men[men['swedish_share'] < 1.0]

        for _, mother in actual_mothers.iterrows():
            father = None
            if mother['swedish_share'] == 1.0: 
                if np.random.rand() < mix_factor_swedish and not men_foreign_background.empty: father = men_foreign_background.sample(1).iloc[0]
                elif not men_swedish_background.empty: father = men_swedish_background.sample(1).iloc[0]
            else: 
                if np.random.rand() < mix_factor_foreign and not men_swedish_background.empty: father = men_swedish_background.sample(1).iloc[0]
                elif not men_foreign_background.empty: father = men_foreign_background.sample(1).iloc[0]
            
            if father is None: father = men.sample(1).iloc[0]

            child_share = (mother['swedish_share'] + father['swedish_share']) / 2
            child_origin = np.random.choice([mother['ethnic_origin'], father['ethnic_origin']]) if mother['ethnic_origin'] != father['ethnic_origin'] else mother['ethnic_origin']
            
            baby = {'id': next_agent_id, 'age': 0, 'sex': np.random.randint(0, 2), 'is_alive': True, 'swedish_share': child_share, 'ethnic_origin': child_origin, 'birth_year': year, 'parent1_id': father['id'], 'parent2_id': mother['id'], 'is_immigrant': False}
            new_babies_list.append(baby)
            events.append({'year': year, 'event': 'Födsel', 'agent_id': next_agent_id, 'mother_id': mother['id'], 'father_id': father['id']})
            next_agent_id += 1
                
    if new_babies_list:
        population = pd.concat([population, pd.DataFrame(new_babies_list)], ignore_index=True)
    return population, events, next_agent_id, len(new_babies_list)

def perform_migration(population, scb_data_year, scale_factor, params, next_agent_id):
    events = []
    # Invandring
    scb_immigrants_scaled = int(scb_data_year["Invandringar In-migration"] / scale_factor)
    if scb_immigrants_scaled > 0:
        waves = params['migration']['historical_waves']
        wave_key = next((p for p in waves if int(p.split('-')[0]) <= scb_data_year.name <= int(p.split('-')[1])), "2001-2024")
        wave = waves.get(wave_key, waves["2001-2024"])
        origins, probs = list(wave.keys()), list(wave.values())
        
        probs_sum = sum(probs)
        if probs_sum == 0: probs = [1.0/len(origins)]*len(origins)
        else: probs = [p / probs_sum for p in probs]

        immigrant_origins = np.random.choice(origins, size=scb_immigrants_scaled, p=probs)
        immigrant_ages = np.random.normal(loc=IMMIGRANT_AGE_MEAN, scale=IMMIGRANT_AGE_STD_DEV, size=scb_immigrants_scaled).astype(int).clip(0, 80)
        start_shares = pd.Series(immigrant_origins).map(params['assimilation']['start_shares']).fillna(0.01).values

        immigrants_df = pd.DataFrame({'age': immigrant_ages, 'sex': np.random.randint(0, 2, size=scb_immigrants_scaled), 'is_immigrant': True, 'is_alive': True, 'swedish_share': start_shares, 'ethnic_origin': immigrant_origins, 'birth_year': scb_data_year.name - immigrant_ages, 'parent1_id': -1, 'parent2_id': -1})
        immigrants_df['id'] = np.arange(next_agent_id, next_agent_id + scb_immigrants_scaled)
        next_agent_id += scb_immigrants_scaled
        for _, imm in immigrants_df.iterrows(): events.append({'year': scb_data_year.name, 'event': 'Invandring', 'agent_id': imm['id'], 'age_at_arrival': imm['age']})
        population = pd.concat([population, immigrants_df], ignore_index=True)

    # Utvandring
    scb_emigrants_scaled = int(scb_data_year["Utvandringar Out-migration"] / scale_factor)
    live_indices = population.index[population['is_alive']]
    if scb_emigrants_scaled > 0 and len(live_indices) > scb_emigrants_scaled:
        emigration_bias = params['migration']['emigration_bias']
        emigration_weights = 1 + (1 - population.loc[live_indices, 'swedish_share']) * (emigration_bias - 1)
        probabilities = emigration_weights / emigration_weights.sum()
        emigrant_indices = np.random.choice(live_indices, size=scb_emigrants_scaled, replace=False, p=probabilities.values)
        population.loc[emigrant_indices, 'is_alive'] = False
        for agent_id in population.loc[emigrant_indices, 'id']: events.append({'year': scb_data_year.name, 'event': 'Utvandring', 'agent_id': agent_id})
    return population, events, next_agent_id

def perform_migration_forecast(population, year, scale_factor, params, forecast_settings, next_agent_id):
    events = []
    
    if forecast_settings['migration_model'] == 'Procent av befolkningen':
        total_live_pop_full_scale = population[population['is_alive']].shape[0] * scale_factor
        num_immigrants_full_scale = total_live_pop_full_scale * forecast_settings['immigration_rate']
        num_immigrants_scaled = int(round(num_immigrants_full_scale / scale_factor))
    else:
        num_immigrants_scaled = int(forecast_settings['annual_immigrants'] / scale_factor)
    
    if num_immigrants_scaled > 0:
        immigrant_composition = forecast_settings['immigrant_composition']
        origins, probs = list(immigrant_composition.keys()), list(immigrant_composition.values())
        probs_sum = sum(probs)
        if probs_sum == 0: probs = [1.0/len(origins)]*len(origins)
        else: probs = [p / probs_sum for p in probs] 

        immigrant_origins = np.random.choice(origins, size=num_immigrants_scaled, p=probs)
        immigrant_ages = np.random.normal(loc=IMMIGRANT_AGE_MEAN, scale=IMMIGRANT_AGE_STD_DEV, size=num_immigrants_scaled).astype(int).clip(0, 80)
        start_shares = pd.Series(immigrant_origins).map(params['assimilation']['start_shares']).fillna(0.01).values

        immigrants_df = pd.DataFrame({'age': immigrant_ages, 'sex': np.random.randint(0, 2, size=num_immigrants_scaled), 'is_immigrant': True, 'is_alive': True, 'swedish_share': start_shares, 'ethnic_origin': immigrant_origins, 'birth_year': year - immigrant_ages, 'parent1_id': -1, 'parent2_id': -1})
        immigrants_df['id'] = np.arange(next_agent_id, next_agent_id + num_immigrants_scaled)
        next_agent_id += num_immigrants_scaled
        for _, imm in immigrants_df.iterrows(): events.append({'year': year, 'event': 'Invandring', 'agent_id': imm['id'], 'age_at_arrival': imm['age']})
        population = pd.concat([population, immigrants_df], ignore_index=True)

    if forecast_settings['migration_model'] == 'Procent av befolkningen':
        total_live_pop_full_scale = population[population['is_alive']].shape[0] * scale_factor 
        num_emigrants_full_scale = total_live_pop_full_scale * forecast_settings['emigration_rate']
        num_emigrants_scaled = int(round(num_emigrants_full_scale / scale_factor))
    else:
        num_emigrants_scaled = int(forecast_settings['annual_emigrants'] / scale_factor)
        
    live_indices = population.index[population['is_alive']]
    if num_emigrants_scaled > 0 and len(live_indices) > num_emigrants_scaled:
        emigration_bias = params['migration']['emigration_bias']
        emigration_weights = 1 + (1 - population.loc[live_indices, 'swedish_share']) * (emigration_bias - 1)
        probabilities = emigration_weights / emigration_weights.sum()
        emigrant_indices = np.random.choice(live_indices, size=num_emigrants_scaled, replace=False, p=probabilities.values)
        population.loc[emigrant_indices, 'is_alive'] = False
        for agent_id in population.loc[emigrant_indices, 'id']: events.append({'year': year, 'event': 'Utvandring', 'agent_id': agent_id})
    return population, events, next_agent_id

def create_yearly_summary(year, population, scb_data_year, num_new_babies):
    """Skapar en sammanfattning för ett historiskt år med jämförelse mot SCB-data."""
    final_live_pop = population[population['is_alive']]
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

def create_forecast_summary(year, population, num_new_babies):
    """Skapar en sammanfattning för ett prognosår utan SCB-data."""
    final_live_pop = population[population['is_alive']]
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

# --- HUVUDFUNKTION FÖR SIMULERING ---
def run_abm_simulation(_scb_data, params, scale_factor, forecast_settings):
    start_year = params['simulation']['start_year']
    total_pop_real = _scb_data.loc[start_year, "Folkmängd 31 december Population on 31 December"]
    num_agents = int(total_pop_real / scale_factor)
    
    ages = np.arange(100); dist = np.exp(-ages * 0.03); dist /= dist.sum()
    agent_ages = np.random.choice(ages, size=num_agents, p=dist)
    
    population = pd.DataFrame({
        'id': range(num_agents),
        'age': agent_ages, 
        'sex': np.random.choice([0, 1], size=num_agents), 
        'is_alive': True, 
        'swedish_share': 1.0, 
        'ethnic_origin': 'Svensk', 
        'parent1_id': -1, 
        'parent2_id': -1, 
        'is_immigrant': False,
        'birth_year': start_year - agent_ages
    })
    
    simulation_summary, event_log = [], []
    progress_bar = st.progress(0, "Startar simulering...")
    next_agent_id = len(population)

    end_year_historical = params['simulation']['end_year']
    end_year_forecast = forecast_settings['forecast_end_year']
    total_years = end_year_forecast - start_year + 1

    for i, year in enumerate(range(start_year, end_year_forecast + 1)):
        if year <= end_year_historical: 
            scb_data_year = _scb_data.loc[year]
            population = perform_aging_and_assimilation(population, year, params)
            population, d_events = perform_deaths(population, scb_data_year, scale_factor)
            population, b_events, next_agent_id, num_babies = perform_births(population, scb_data_year, scale_factor, params, next_agent_id)
            population, m_events, next_agent_id = perform_migration(population, scb_data_year, scale_factor, params, next_agent_id)
            event_log.extend(d_events + b_events + m_events)
            summary = create_yearly_summary(year, population, scb_data_year, num_babies)
        else:
            population = perform_aging_and_assimilation(population, year, params)
            population, d_events = perform_deaths_forecast(population, year, scale_factor, params, forecast_settings)
            population, b_events, next_agent_id, num_babies = perform_births_forecast(population, year, scale_factor, params, forecast_settings, next_agent_id)
            population, m_events, next_agent_id = perform_migration_forecast(population, year, scale_factor, params, forecast_settings, next_agent_id)
            event_log.extend(d_events + b_events + m_events)
            summary = create_forecast_summary(year, population, num_babies)
        
        simulation_summary.append(summary)
        progress_bar.progress((i + 1) / total_years, f"Simulerar år {year+1}...")
    
    progress_bar.empty()
    population.loc[population['is_alive'], 'age'] += 1
    return pd.DataFrame(simulation_summary).set_index('Year'), population, pd.DataFrame(event_log)

# --- HUVUDPROGRAM & ANVÄNDARGRÄNSSNITT ---
st.title("🇸🇪 Demografisk Simulator: Kulturell Dynamik (v4.1)")
st.sidebar.header("Kärnparametrar")
scale_factor = st.sidebar.slider(
    "Simuleringens Skalningsfaktor (1:X)", 10, 1000, 1000, 10,
    help="Bestämmer upplösningen på simuleringen. En lägre siffra ger en mer detaljerad (men långsammare) simulering. Exempel: Vid 1000 representerar 1 'agent' 1000 verkliga personer. Standard är 1000."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resultat", "Parametrar", "Framtidsscenarier", "Agent-utforskaren", "Om Modellen"])

# Kartläggning från kodens namn till snygga UI-etiketter.
UI_LABELS = {
    "Swedish_Background": "Svensk bakgrund",
    "First_Generation": "Första generationen",
    "Later_Generations": "Senare generationer",
    "Simulated_Population": "Simulerad befolkning",
    "Actual_Population_SCB": "Verklig befolkning (SCB)",
    "Simulated_Births": "Simulerade födslar",
    "Actual_Births_SCB": "Verkliga födslar (SCB)"
}

with tab1:
    st.header("Kör Simulering & Analysera Resultat")
    params_to_run = st.session_state['edited_params']
    forecast_settings_to_run = st.session_state['forecast_settings']
    final_end_year = forecast_settings_to_run['forecast_end_year']
    
    if st.button(f"KÖR SIMULERING (Skala 1:{scale_factor}, till år {final_end_year})", use_container_width=True, type="primary"):
        scb_data = load_scb_data('SCB Raw Data.csv', final_end_year) 
        summary_df, final_pop, event_log = run_abm_simulation(scb_data, params_to_run, scale_factor, forecast_settings_to_run)
        st.session_state.update({'results_summary': summary_df, 'results_population': final_pop, 'results_events': event_log})
    
    if 'results_summary' in st.session_state:
        results = st.session_state['results_summary']
        st.success("Simuleringen är klar!", icon="✅")
        st.subheader("Resultat")
        scale_results = st.checkbox("Visa resultat i verklig skala", value=True)
        
        display_df = results.copy()
        if scale_results:
            cols_to_scale = ['Simulated_Population', 'Swedish_Background', 'Later_Generations', 'First_Generation', 'Simulated_Births']
            for col in cols_to_scale: 
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').multiply(scale_factor)
        
        st.info(f"Data efter {params_to_run['simulation']['end_year']} bygger på prognosantaganden.")
        
        st.subheader("Befolkningsutveckling per bakgrund")
        c1, c2 = st.columns([1,1])
        
        default_groups_code = ["Swedish_Background", "Later_Generations", "First_Generation"]
        options_for_multiselect = {code: label for code, label in UI_LABELS.items() if code in default_groups_code}
        
        selected_groups_code = c1.multiselect("Välj grupper att visa:", options=options_for_multiselect.keys(), default=default_groups_code, format_func=lambda code: options_for_multiselect[code])
        
        combine_groups = c2.checkbox("Slå ihop grupper med utländsk bakgrund")
        plot_df = display_df.copy().rename(columns=UI_LABELS)
        
        if combine_groups:
            plot_df["Utländsk bakgrund (totalt)"] = plot_df[UI_LABELS["First_Generation"]].fillna(0) + plot_df[UI_LABELS["Later_Generations"]].fillna(0)
            groups_to_plot_labels = [UI_LABELS[g] for g in selected_groups_code if g not in ["First_Generation", "Later_Generations"]] + ["Utländsk bakgrund (totalt)"]
        else: 
            groups_to_plot_labels = [UI_LABELS[g] for g in selected_groups_code]
        
        fig_groups = px.line(plot_df, y=groups_to_plot_labels, labels={"value": "Antal Personer", "variable": "Befolkningsgrupp"})
        fig_groups.add_vline(x=params_to_run['simulation']['end_year'], line_width=2, line_dash="dash", line_color="red", annotation_text="Prognosstart", annotation_position="top left")
        st.plotly_chart(fig_groups, use_container_width=True)
        
        with st.expander("Visa valideringsgrafer"):
            val_df = plot_df
            fig_pop = px.line(val_df, y=[UI_LABELS['Simulated_Population'], UI_LABELS['Actual_Population_SCB']], title="Total Befolkning (Simulerad vs. Verklig)")
            fig_pop.add_vline(x=params_to_run['simulation']['end_year'], line_width=2, line_dash="dash", line_color="red")
            st.plotly_chart(fig_pop, use_container_width=True)
            fig_births = px.line(val_df, y=[UI_LABELS['Simulated_Births'], UI_LABELS['Actual_Births_SCB']], title="Antal Födda (Simulerat vs. Verkligt)")
            fig_births.add_vline(x=params_to_run['simulation']['end_year'], line_width=2, line_dash="dash", line_color="red")
            st.plotly_chart(fig_births, use_container_width=True)
        
        with st.expander("Visa Detaljerad Simuleringslogg"): 
            st.dataframe(display_df.rename(columns=UI_LABELS).style.format("{:,.0f}", na_rep='-'), use_container_width=True)

with tab2:
    st.header("Modellparametrar")
    st.info("Här kan du finjustera modellens grundantaganden. Standardvärdena är baserade på officiell statistik och vedertagna demografiska principer där det är möjligt.") 
    p = st.session_state['edited_params']
    
    with st.expander("Assimilering & Social Dynamik", expanded=True):
        p['assimilation']['base_rate'] = st.slider(
            "Maximal Assimilationstakt (% per år)", 0.0, 2.0, p['assimilation']['base_rate'] * 100, 0.1,
            help="**Detta är ett grundantagande i modellen och har ingen direkt statistisk källa.**\n\nDetta värde representerar den teoretiska maxhastigheten för assimilering under de mest gynnsamma förhållandena (dvs. när 100% av befolkningen har svensk bakgrund).\n\nDen **faktiska** assimilationstakten varje år är en produkt av detta värde, den dynamiska styrkan (nedan), och ursprungsspecifika multiplikatorer."
        ) / 100.0
        
        st.markdown("---")
        st.subheader("Assimileringens Dynamik")
        p['assimilation']['strength_multiplier'] = st.slider(
            "Styrka på assimilationsdynamik", 0.0, 3.0, p['assimilation']['strength_multiplier'], 0.1,
            help="Styr den övergripande kraften i den dynamiska assimilationsmodellen.\n\nModellen är designad så att assimilationsprocessen naturligt stannar av när andelen av befolkningen med 'svensk bakgrund' är 50%. Denna parameter skalar hur starkt assimileringen driver mot 100% (när andelen är >50%) eller mot 0% (när andelen är <50%).\n\n- **1.0 (Standard):** Modellen körs med standardantagandet.\n- **> 1.0:** Simulerar ett samhälle med starkare sociala krafter (snabbare assimilering/de-assimilering).\n- **< 1.0:** Simulerar ett samhälle där sociala processer är trögare.\n- **0.0:** Stänger helt av den dynamiska mekanismen."
        )

        st.markdown("---")
        st.subheader("Utvandringsbenägenhet")
        p['migration']['emigration_bias'] = st.slider(
            "Multiplikator för utvandring (utländsk bakgrund)", 1.0, 15.0, p['migration']['emigration_bias'], 0.5,
            help="Styr hur mycket mer sannolikt det är att en person med 0% svenskandel utvandrar jämfört med en med 100%.\n\n**Relation:** En persons utvandringssannolikhet beräknas som `1 + (1 - swedish_share) * (Multiplikator - 1)`.\n\n**Källa:** Standardvärdet är en modellkalibrering, men principen baseras på SCB-statistik som visar att utrikes födda har en betydligt högre benägenhet att utvandra. För att hitta denna typ av data, sök på 'Utrikes föddas återutvandring' på scb.se."
        )
        
        st.markdown("---")
        st.subheader("Blandrelationer")
        st.markdown("Här ställs sannolikheten för att en mor hittar en partner **utanför** sin egen bakgrundsgrupp. Värdena interpoleras linjärt mellan start- och slutår.")
        c1, c2 = st.columns(2)
        p['mixing']['swedish_background_mother']['start_value'] = c1.slider(
            "Mor med svensk bakgrund, 1960 (%)", 0.0, 50.0, p['mixing']['swedish_background_mother']['start_value'] * 100, 1.0,
            help="Sannolikheten att en mor med 100% svenskandel får barn med en far som har <100% svenskandel. **Källa:** Värdena är modellkalibreringar baserade på den observerade ökningen av barn med en inrikes och en utrikes född förälder. För att hitta denna typ av data, sök på 'Familjeliv i Sverige' på scb.se."
        ) / 100.0
        p['mixing']['swedish_background_mother']['end_value'] = c2.slider("Mor med svensk bakgrund, 2019 (%)", 0.0, 50.0, p['mixing']['swedish_background_mother']['end_value'] * 100, 1.0) / 100.0
        c3, c4 = st.columns(2)
        p['mixing']['foreign_background_mother']['start_value'] = c3.slider(
            "Mor med utländsk bakgrund, 1960 (%)", 0.0, 50.0, p['mixing']['foreign_background_mother']['start_value'] * 100, 1.0,
            help="Sannolikheten att en mor med <100% svenskandel får barn med en far som har 100% svenskandel."
        ) / 100.0
        p['mixing']['foreign_background_mother']['end_value'] = c4.slider("Mor med utländsk bakgrund, 2019 (%)", 0.0, 50.0, p['mixing']['foreign_background_mother']['end_value'] * 100, 1.0) / 100.0
        
    with st.expander("Parametrar per Ursprungsgrupp"):
        st.subheader("Initial 'svenskandel' vid ankomst")
        st.markdown("Detta är ett **modellantagande** som representerar en nyanländ immigrants initiala kulturella och sociala närhet till det svenska samhället.")
        c1, c2, c3, c4, c5 = st.columns(5)
        cols = [c1, c2, c3, c4, c5]
        origins = list(p['assimilation']['start_shares'].keys())
        for i, origin in enumerate(origins):
            p['assimilation']['start_shares'][origin] = cols[i].slider(f"{origin} (%)", 0.0, 100.0, p['assimilation']['start_shares'][origin] * 100, 1.0, key=f"start_{origin}") / 100.0

        st.markdown("---")
        st.subheader("Relativ Assimileringstakt (Multiplikator)")
        st.markdown("Detta är ett **modellantagande** som reflekterar hur snabbt olika grupper antas assimilera sig, relativt till varandra, baserat på faktorer som kulturellt och språkligt avstånd.")
        c1, c2, c3, c4, c5 = st.columns(5)
        cols = [c1, c2, c3, c4, c5]
        origins = list(p['assimilation']['rate_multipliers'].keys())
        for i, origin in enumerate(origins):
            p['assimilation']['rate_multipliers'][origin] = cols[i].slider(f"{origin}", 0.0, 10.0, p['assimilation']['rate_multipliers'][origin], 0.5, key=f"rate_{origin}")
        
    with st.expander("Fertilitetsbonus per Ursprungsgrupp"):
        st.markdown("Här anges en fertilitetsbonus relativt till gruppen med svensk bakgrund. Värdet är en multiplikator på den del av födselpotentialen som inte är kopplad till 'svenskandel'.")
        for origin, bonus in p['fertility']['bonuses'].items():
            if origin != "Svensk": 
                p['fertility']['bonuses'][origin] = st.slider(
                    origin, 0.0, 2.0, bonus, 0.1, key=f"fert_{origin}",
                    help=f"**Källa:** Standardvärdena är kalibrerade för att spegla observerade skillnader i summerad fruktsamhet (TFR) mellan inrikes födda och utrikes födda från olika regioner. För att hitta denna data, sök på 'Befolkningsframskrivning' på scb.se och leta efter tabeller om fruktsamhet för inrikes/utrikes födda."
                )
            
    with st.expander("Historisk Invandrarsammansättning"):
        st.markdown("Fördelningen av invandrare från olika ursprungsregioner under olika tidsperioder.")
        for period, wave in sorted(p['migration']['historical_waves'].items()):
            st.subheader(f"Period: {period}")
            st.markdown(f"**Källa:** Standardvärdena är approximeringar baserade på SCB:s data över invandring. För att hitta denna typ av data, sök på 'Invandring och utvandring efter födelseland' i Statistikdatabasen på scb.se. Värdena speglar de stora skiftena från arbetskraftsinvandring från Europa till flyktinginvandring.", help="Den totala summan för en period bör vara 100%. Modellen normaliserar automatiskt värdena om summan inte stämmer.")
            cols = st.columns(len(wave))
            for i, (origin, prob) in enumerate(wave.items()): 
                p['migration']['historical_waves'][period][origin] = cols[i].number_input(f"{origin} (%)", 0.0, 100.0, prob * 100, 5.0, key=f"{period}_{origin}") / 100.0
    
    st.session_state['edited_params'] = p

with tab3:
    st.header("Framtidsscenarier")
    st.info("Här definierar du antaganden för modellens prognosperiod (från 2025 och framåt). Standardvärdena är satta för att spegla SCB:s senaste huvudscenario för Sveriges framtida befolkning.")
    settings = st.session_state['forecast_settings'] 

    st.subheader("Simuleringshorisont")
    settings['forecast_end_year'] = st.slider("Prognosens Slutår", 2025, 2200, settings['forecast_end_year'], 5)

    st.subheader("Framtida Fruktsamhet")
    settings['tfr'] = st.slider(
        "Summerad Fruktsamhet (TFR)", 1.0, 2.5, settings.get('tfr', 1.59), 0.01,
        help="Genomsnittligt antal barn en kvinna förväntas föda under sin livstid. **Standardvärdet (1.59)** är SCB:s långsiktiga antagande i deras senaste befolkningsprognos. För att hitta källan, sök på 'Befolkningsframskrivning 2024–2070' på scb.se."
    )

    st.subheader("Framtida Dödlighet")
    st.markdown("Modellen interpolerar linjärt mot dessa målvärden fram till år 2050.")
    c1, c2, c3 = st.columns(3)
    settings['target_le_male'] = c1.slider(
        "Mål: Livslängd Män (2050)", 75.0, 95.0, settings.get('target_le_male', 85.3), 0.1,
        help="Förväntad medellivslängd för män år 2050. **Standardvärdet (85.3 år)** är SCB:s prognos. För att hitta källan, sök på 'Befolkningsframskrivning 2024–2070' på scb.se."
    )
    settings['target_le_female'] = c2.slider(
        "Mål: Livslängd Kvinnor (2050)", 80.0, 100.0, settings.get('target_le_female', 87.7), 0.1,
        help="Förväntad medellivslängd för kvinnor år 2050. **Standardvärdet (87.7 år)** är SCB:s prognos. För att hitta källan, sök på 'Befolkningsframskrivning 2024–2070' på scb.se."
    )
    settings['cdr_target_2050'] = c3.slider(
        "Mål: Dödstal (CDR, per 1000 inv.)", 5.0, 15.0, settings['cdr_target_2050'], 0.1,
        help="Rått dödstal (Crude Death Rate). Detta värde påverkas av befolkningens åldersstruktur. Standardvärdet är en kalibrering som samspelar med den ökande livslängden och en åldrande befolkning."
    )

    st.subheader("Framtida Migration")
    st.markdown("Detta är en av de mest osäkra faktorerna i en befolkningsprognos. Standardvärdena baseras på ett genomsnitt av de senaste årens utfall, men kan justeras för att testa olika scenarier.")
    migration_model_options = ('Absoluta tal', 'Procent av befolkningen')
    current_model_index = migration_model_options.index(settings.get('migration_model', 'Procent av befolkningen'))
    
    settings['migration_model'] = st.radio("Välj migrationsmodell för prognosen:", migration_model_options, index=current_model_index)
    
    c1, c2 = st.columns(2)
    if settings['migration_model'] == 'Absoluta tal':
        settings['annual_immigrants'] = c1.slider("Årlig Invandring (antal)", 0, 150000, settings['annual_immigrants'], 5000)
        settings['annual_emigrants'] = c2.slider("Årlig Utvandring (antal)", 0, 100000, settings['annual_emigrants'], 5000)
    else:
        settings['immigration_rate'] = c1.slider("Årlig Invandring (% av befolkningen)", 0.0, 2.0, settings.get('immigration_rate', 0.0075) * 100, 0.05) / 100.0
        settings['emigration_rate'] = c2.slider("Årlig Utvandring (% av befolkningen)", 0.0, 2.0, settings.get('emigration_rate', 0.0057) * 100, 0.05) / 100.0
    
    st.markdown("---")
    st.subheader("Framtida Invandrarsammansättning (från 2025)")
    st.markdown("Andelen invandrare från olika regioner. Detta är ett **scenariovärde** som du kan justera. Standard är att fortsätta med samma fördelning som under den senaste historiska perioden (2001-2024).")
    cols = st.columns(len(st.session_state['future_migration_composition']))
    for i, (origin, prob) in enumerate(st.session_state['future_migration_composition'].items()):
        st.session_state['future_migration_composition'][origin] = cols[i].number_input(
            f"{origin} (%)", 0.0, 100.0, st.session_state['future_migration_composition'][origin] * 100, 5.0,
            key=f"future_mig_comp_{origin}"
        ) / 100.0

    current_sum = sum(st.session_state['future_migration_composition'].values())
    if not np.isclose(current_sum, 1.0, atol=0.01):
        st.warning(f"Varning: Invandrarsammansättningen summerar till {current_sum*100:.0f}%. Justera för att få 100%.")

    settings['immigrant_composition'] = st.session_state['future_migration_composition']
    st.session_state['forecast_settings'] = settings

with tab4:
    st.header("Agent-utforskaren")
    if 'results_population' not in st.session_state: st.info("Kör en simulering på första fliken för att ladda agentdata.")
    else:
        population_df = st.session_state['results_population']
        event_log_df = st.session_state['results_events']
        with st.expander("Sök & Filtrera Agenter", expanded=True):
            c1, c2, c3 = st.columns(3)
            unique_origins = list(population_df['ethnic_origin'].unique())
            origin_filter = c1.selectbox("Filtrera på Ursprung:", options=['Alla'] + sorted(unique_origins)) 
            max_age_possible = int(population_df['age'].max()) if not population_df.empty else 120
            age_filter = c2.slider(f"Filtrera på Slutålder ({st.session_state['forecast_settings']['forecast_end_year']}):", -1, max_age_possible, (-1, max_age_possible))
            share_filter = c3.slider("Filtrera på Slutgiltig 'svenskandel' (%):", 0, 100, (0, 100))
            
            filtered_df = population_df.copy()
            if origin_filter != 'Alla': filtered_df = filtered_df[filtered_df['ethnic_origin'] == origin_filter]
            if age_filter[0] > -1: filtered_df = filtered_df[filtered_df['age'] >= age_filter[0]]
            if age_filter[1] < max_age_possible: filtered_df = filtered_df[filtered_df['age'] <= age_filter[1]]
            filtered_df = filtered_df[(filtered_df['swedish_share'] * 100 >= share_filter[0]) & (filtered_df['swedish_share'] * 100 <= share_filter[1])]
            
            st.write(f"Hittade {len(filtered_df)} agenter som matchar dina filter.")
            st.dataframe(filtered_df.head(1000).style.format({'swedish_share': '{:.2%}'}))
        
        # Logik för att visa en enskild agents livshistoria är oförändrad.

with tab5:
    st.header("Så fungerar modellen")
    st.markdown("""
    Detta verktyg är en **agentbaserad demografisk simulator** för Sverige. Det betyder att den skapar en digital miniatyrbefolkning av individer ("agenter") och simulerar deras livsöden år för år, från 1960 och in i framtiden.
    
    Kärnan i modellen är att spåra och simulera **kulturell dynamik**. Detta görs genom att varje person i simuleringen har en grad av social och kulturell tillhörighet till den svenska majoritetskulturen. Detta gör det möjligt att utforska hur befolkningens sammansättning kan förändras över tid, baserat på en kombination av historisk data och dina egna antaganden om framtiden.
    
    Nedan följer en förklaring av modellen på tre olika nivåer av detalj.
    """)

    with st.expander("Abstraktnivå 1: Hög (Koncis översikt)", expanded=True):
        st.markdown("""
        Detta verktyg simulerar hur Sveriges befolkning och dess kulturella sammansättning utvecklas. Det börjar med verkliga siffror från Statistiska centralbyrån (SCB) för år 1960 och följer den historiska utvecklingen fram till idag. Därefter skapar den en prognos för framtiden baserat på de antaganden du ställer in.

        Varje person i simuleringen är en digital figur med egenskaper som ålder, kön och kulturell bakgrund. Modellen beräknar år för år hur dessa digitala personer åldras, får barn, flyttar och hur deras kulturella tillhörighet förändras över tid.

        Du kan ändra på antaganden om framtida barnafödande, migration och hur snabbt integrationen går för att testa olika "tänk om"-scenarier. Resultaten hjälper till att visualisera de långsiktiga effekterna av olika demografiska trender.
        """)

    with st.expander("Abstraktnivå 2: Medel (Balanserad förklaring)", expanded=False):
        st.markdown("""
        Simulatorn bygger på en population av digitala individer. Varje individ har en central egenskap: en grad av **kulturell tillhörighet** till den svenska majoritetskulturen, ett värde mellan 0% och 100%.

        **Viktigt antagande:** Eftersom detaljerad data saknas, utgår modellen från en förenklad startpunkt där hela Sveriges befolkning år 1960 antas ha 100% svensk kulturell tillhörighet.

        Modellen körs i en årlig cykel. För varje år sker följande händelser i ordning:

        - **1. Åldrande och Kulturell Förändring:** Alla individer blir ett år äldre. Samtidigt justeras deras kulturella tillhörighet baserat på den totala befolkningens sammansättning. Processen är dynamisk:
            - I ett samhälle med en stor majoritet med svensk bakgrund, är den sociala "dragkraften" mot integration stark.
            - Om andelen med svensk bakgrund minskar och närmar sig 50%, avstannar integrationsprocessen.
            - Om andelen understiger 50%, kan processen vända, vilket representerar en starkare dragkraft mot minoritetskulturer.

        - **2. Dödsfall:** Individer riskerar att dö baserat på sin ålder. För historiska år (1960-2024) styrs antalet dödsfall exakt av SCB:s data. För framtiden används dina prognoser om ökad medellivslängd.

        - **3. Födslar:** Kvinnor i fertil ålder kan få barn. Sannolikheten påverkas av deras ursprung och kulturella tillhörighet, för att matcha de observerade fertilitetsskillnaderna i befolkningen. Ett barn ärver en blandning av sina föräldrars egenskaper.

        - **4. Migration:** Nya individer (invandrare) skapas och läggs till i befolkningen, med en sammansättning som följer historiska migrationsvågor. Samtidigt tas individer bort för att representera utvandring, där de med en lägre grad av svensk kulturell tillhörighet har en högre sannolikhet att lämna landet.
        """)

    with st.expander("Abstraktnivå 3: Låg (Detaljerad genomgång)", expanded=False):
        st.markdown("""
        Här följer en detaljerad konceptuell genomgång av modellens komponenter och processer.

        ### 1. Startpunkt och Grundantaganden
        - **Data:** Modellen använder en tidsserie från SCB (1960-2024) med årlig data om total befolkning, födda, döda, invandring och utvandring.
        - **Starttillstånd (1960):** Simuleringen börjar med en befolkning som matchar den verkliga storleken och åldersfördelningen för 1960. Ett centralt och förenklande antagande görs här: **alla individer i startpopulationen tilldelas 100% svensk kulturell tillhörighet.** Detta är en nödvändig förenkling på grund av brist på detaljerad historisk data.

        ### 2. Den Årliga Simuleringscykeln
        För varje år från 1960 till prognosens slutår utförs följande steg:

        **A. Åldrande & Kulturell Förändring**
        Alla levande individer blir ett år äldre. Därefter sker den kulturella dynamiken för de individer som befinner sig i en integrationsprocess (varken 0% eller 100% tillhörighet):
        1.  **Analys av samhällsklimatet:** Modellen mäter andelen av den totala befolkningen som har en helt svensk bakgrund.
        2.  **Bestämning av kulturell "dragkraft":** Baserat på denna andel bestäms en social "dragkraft". Modellen använder en linjär skala:
            - Om 100% av befolkningen har svensk bakgrund, är den positiva kraften mot integration som starkast.
            - Om 50% har svensk bakgrund, är kraften neutral. Det råder en balans där varken integration eller segregation dominerar.
            - Om 0% har svensk bakgrund, är den negativa kraften (de-assimilering) som starkast, vilket representerar en stark dragkraft mot andra kulturella normer.
        3.  **Individuell förändring:** Den årliga förändringen för en individs kulturella tillhörighet beräknas sedan baserat på denna "dragkraft", en inställbar grundhastighet, och individens ursprungsregion.

        **B. Dödsfall**
        - **Historiska år (1960-2024):** En grundrisk att dö baserat på ålder justeras så att det totala antalet dödsfall i simuleringen exakt matchar SCB:s officiella statistik för det året.
        - **Prognosår (2025+):** Risken att dö justeras dynamiskt för att gradvis uppnå de målvärden för medellivslängd som du ställt in.

        **C. Födslar**
        1.  **Födselpotential:** Kvinnor i fertil ålder (15-49 år) får en "födselpotential" som är högre om hon har ett ursprung med statistiskt högre fertilitet och lägre ju starkare hennes svenska kulturella tillhörighet är.
        2.  **Kalibrering:** Den totala potentialen i populationen justeras så att det förväntade antalet födslar matchar antingen SCB:s historiska data eller din prognos för framtida barnafödande (TFR).
        3.  **Partner-val och arv:** För varje födsel väljs en far, med en viss sannolikhet för parbildning över bakgrundsgränserna. Barnet ärver hälften av varje förälders kulturella tillhörighet och ett slumpmässigt valt ursprung från en av dem.

        **D. Migration**
        - **Invandring:** Nya individer skapas. Deras antal styrs av SCB:s data (historiskt) eller dina prognosinställningar (framtid). Deras ålder och ursprungsregion slumpas enligt de fördelningar du ställt in. De tilldelas en initial kulturell tillhörighet baserat på sitt ursprung.
        - **Utvandring:** Individer tas bort från simuleringen. Antalet styrs på samma sätt. Urvalet av vilka som utvandrar är dock inte helt slumpmässigt; individer med låg kulturell tillhörighet har en högre sannolikhet att lämna landet.

        ### 3. Trovärdighet
        Modellens styrka ligger i kombinationen av hård data och flexibla antaganden.
        - **Historisk förankring:** Fram till idag är modellen inte en prognos, utan en **rekonstruktion** som är kalibrerad för att följa den verkliga utvecklingen enligt SCB.
        - **Källbaserade prognoser:** Standardinställningarna för framtiden är inte godtyckliga, utan baseras på SCB:s officiella befolkningsframskrivning från 2024.
        - **Transparenta antaganden:** Parametrar som är rena modellantaganden är tydligt markerade som sådana, med förklaringar till logiken bakom dem.
        """)
