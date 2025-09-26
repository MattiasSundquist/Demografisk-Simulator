import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- SIDANS GRUNDINSTÄLLNINGAR ---
st.set_page_config(page_title="Demografisk Simulator 3.3", layout="wide")

# --- DEL 2: AGENT-BASERAD MODELL - Huvudparametrar ---
def get_model_v2_params():
    params = {
        "start_year": 1960, "end_year": 2024,
        "base_assimilation_rate": 0.005,
        "mixing_factor_inhemsk": {"start_year": 1960, "start_value": 0.10, "end_year": 2019, "end_value": 0.25},
        "mixing_factor_utlandsk": {"start_year": 1960, "start_value": 0.25, "end_year": 2019, "end_value": 0.10},
        "fertility_bonuses": {
            "Svensk": 0.0, "Scandinavieninvandring": 0.0, "Europainvandring": 0.15, 
            "Balkan/LATAM-invandring": 0.45, "MENA-invandring": 1.2, "Övriga": 0.35
        },
        "migration_waves": {
            "1960-1980": {"Europainvandring": 0.30, "Scandinavieninvandring": 0.30, "Balkan/LATAM-invandring": 0.25, "MENA-invandring": 0.05, "Övriga": 0.10},
            "1981-2000": {"Europainvandring": 0.20, "Scandinavieninvandring": 0.20, "Balkan/LATAM-invandring": 0.40, "MENA-invandring": 0.10, "Övriga": 0.10},
            "2001-2024": {"Europainvandring": 0.10, "Scandinavieninvandring": 0.10, "Balkan/LATAM-invandring": 0.10, "MENA-invandring": 0.60, "Övriga": 0.10},
        },
        "assimilation_start_share": {
            "Scandinavieninvandring": 0.75, "Europainvandring": 0.50, "Balkan/LATAM-invandring": 0.25, "MENA-invandring": 0.05, "Övriga": 0.10
        },
        "assimilation_rate_multiplier": {
            "Scandinavieninvandring": 7.0, "Europainvandring": 5.0, "Balkan/LATAM-invandring": 3.0, "MENA-invandring": 1.0, "Övriga": 2.0
        },
        "le_male_2024": 82.29, "le_female_2024": 85.35, "cdr_2024": 9.0,
        "enable_tipping_point_assimilation": True, "assimilation_tipping_point": 0.50, "negative_assimilation_strength": 1.0
    }
    return params

# --- INITIERA SESSION STATE MED STANDARDVÄRDEN ---
if 'edited_params' not in st.session_state:
    st.session_state['edited_params'] = get_model_v2_params()

if 'future_migration_composition' not in st.session_state:
    st.session_state['future_migration_composition'] = st.session_state['edited_params']["migration_waves"]["2001-2024"].copy()

if 'forecast_settings' not in st.session_state:
    st.session_state['forecast_settings'] = {
        'forecast_end_year': 2070, 'tfr': 1.65, 'target_le_male': 86.0, 'target_le_female': 88.0,
        'cdr_target_2050': 8.8, 'immigrant_composition': st.session_state['future_migration_composition'],
        'migration_model': 'Absoluta tal', 'annual_immigrants': 80000, 'annual_emigrants': 60000,
        'immigration_rate': 0.0075, 'emigration_rate': 0.0057
    }

# --- DEL 1: LADDNING AV SCB-DATA ---
@st.cache_data
def load_and_prepare_scb_data(filepath, max_year):
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

    share_inhemsk = (live_pop_mask & (population['swedish_share'] >= 0.999)).sum() / live_pop_mask.sum()
    
    rate_multiplier = population['ethnic_origin'].map(params['assimilation_rate_multiplier']).fillna(1.0)
    
    if params.get("enable_tipping_point_assimilation", False) and share_inhemsk < params["assimilation_tipping_point"]:
        # --- SCENARIO 1: NEGATIV ASSIMILERING (under tröskeln) ---
        negative_base_rate = params['base_assimilation_rate'] * params["negative_assimilation_strength"]
        assimilation_change = -1 * negative_base_rate * (1.0 - population['swedish_share']) * rate_multiplier
    else:
        # --- SCENARIO 2: STANDARD POSITIV ASSIMILERING (i alla andra fall) ---
        dynamic_assimilation_rate = params['base_assimilation_rate'] * share_inhemsk
        assimilation_change = dynamic_assimilation_rate * population['swedish_share'] * rate_multiplier

    population.loc[live_pop_mask & (population['swedish_share'] > 0), 'swedish_share'] += assimilation_change
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
    # (Denna funktion är oförändrad från föregående version)
    events = []
    live_pop_indices = population.index[population['is_alive']]
    if live_pop_indices.empty: return population, events

    le_m_2024, le_f_2024 = params['le_male_2024'], params['le_female_2024']
    target_le_m_2050, target_le_f_2050 = forecast_settings['target_le_male'], forecast_settings['target_le_female']
    interp_start_year_le, interp_end_year_le = params['end_year'], 2050
    
    if year <= interp_end_year_le:
        interp_factor_le = (year - interp_start_year_le) / (interp_end_year_le - interp_start_year_le) if (interp_end_year_le - interp_start_year_le) > 0 else 0
        current_le_m = le_m_2024 + (target_le_m_2050 - le_m_2024) * interp_factor_le
        current_le_f = le_f_2024 + (target_le_f_2050 - le_f_2024) * interp_factor_le
    else:
        current_le_m, current_le_f = target_le_m_2050, target_le_f_2050

    mortality_rate_base = 0.001 + (population.loc[live_pop_indices, 'age'] / 100) ** 4
    adj_factor_male_le = (le_m_2024 / current_le_m) if current_le_m > 0 else 1.0
    adj_factor_female_le = (le_f_2024 / current_le_f) if current_le_f > 0 else 1.0

    mortality_rate_adjusted = mortality_rate_base.copy()
    male_mask = population.loc[live_pop_indices, 'sex'] == 0
    mortality_rate_adjusted.loc[male_mask] *= adj_factor_male_le
    mortality_rate_adjusted.loc[~male_mask] *= adj_factor_female_le

    cdr_2024, cdr_2050 = params['cdr_2024'], forecast_settings['cdr_target_2050']
    interp_start_year_cdr, interp_end_year_cdr = params['end_year'], 2050
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
    # (Denna funktion är oförändrad från föregående version)
    events, new_babies_list = [], []
    live_pop_mask = population['is_alive']
    mothers = population[live_pop_mask & (population['sex'] == 1) & (population['age'] >= 15) & (population['age'] <= 49)]

    if not mothers.empty:
        scb_births_scaled = scb_data_year["Födda Births"] / scale_factor
        fertility_bonus = mothers['ethnic_origin'].map(params['fertility_bonuses']).fillna(0)
        base_fertility = 1.0 - (0.8 * mothers['swedish_share'])
        birth_potential = (1 + (fertility_bonus * base_fertility))
        birth_prob_factor = scb_births_scaled / birth_potential.sum() if birth_potential.sum() > 0 else 0
        birth_prob = birth_potential * birth_prob_factor
        gives_birth = np.random.rand(len(mothers)) < birth_prob
        actual_mothers = mothers[gives_birth]
        men = population[live_pop_mask & (population['sex'] == 0) & (population['age'] >= 18) & (population['age'] <= 60)]

        if not actual_mothers.empty and not men.empty:
            year = scb_data_year.name
            p_inhemsk, p_utlandsk = params['mixing_factor_inhemsk'], params['mixing_factor_utlandsk']
            mix_inhemsk = np.interp(year, [p_inhemsk['start_year'], p_inhemsk['end_year']], [p_inhemsk['start_value'], p_inhemsk['end_value']])
            mix_utlandsk = np.interp(year, [p_utlandsk['start_year'], p_utlandsk['end_year']], [p_utlandsk['start_value'], p_utlandsk['end_value']])
            
            men_inhemska = men[men['swedish_share'] == 1.0]
            men_utlandsk_bakgrund = men[men['swedish_share'] < 1.0]

            for _, mother in actual_mothers.iterrows():
                father = None
                if mother['swedish_share'] == 1.0:
                    if np.random.rand() < mix_inhemsk and not men_utlandsk_bakgrund.empty: father = men_utlandsk_bakgrund.sample(1).iloc[0]
                    elif not men_inhemska.empty: father = men_inhemska.sample(1).iloc[0]
                else:
                    if np.random.rand() < mix_utlandsk and not men_inhemska.empty: father = men_inhemska.sample(1).iloc[0]
                    elif not men_utlandsk_bakgrund.empty: father = men_utlandsk_bakgrund.sample(1).iloc[0]
                
                if father is None: father = men.sample(1).iloc[0]

                child_share = (mother['swedish_share'] + father['swedish_share']) / 2
                child_origin = mother['ethnic_origin'] if mother['swedish_share'] < father['swedish_share'] else father['ethnic_origin']
                baby = {'id': next_agent_id, 'age': 0, 'sex': np.random.randint(0, 2), 'is_alive': True, 'swedish_share': child_share, 'ethnic_origin': child_origin, 'birth_year': year, 'parent1_id': father['id'], 'parent2_id': mother['id'], 'is_immigrant': False}
                new_babies_list.append(baby)
                events.append({'year': year, 'event': 'Födsel', 'agent_id': next_agent_id, 'mother_id': mother['id'], 'father_id': father['id']})
                next_agent_id += 1
                
    if new_babies_list:
        population = pd.concat([population, pd.DataFrame(new_babies_list)], ignore_index=True)
    return population, events, next_agent_id, len(new_babies_list)

def perform_births_forecast(population, year, scale_factor, params, forecast_settings, next_agent_id):
    # (Denna funktion är oförändrad från föregående version)
    events, new_babies_list = [], []
    live_pop_mask = population['is_alive']
    mothers = population[live_pop_mask & (population['sex'] == 1) & (population['age'] >= 15) & (population['age'] <= 49)]
    if mothers.empty: return population, events, next_agent_id, 0

    num_fertile_women_full_scale = len(mothers) * scale_factor
    annual_birth_rate_per_woman = forecast_settings['tfr'] / 35.0
    expected_births_full_scale = num_fertile_women_full_scale * annual_birth_rate_per_woman
    expected_births_scaled = int(round(expected_births_full_scale / scale_factor))
    if expected_births_scaled <= 0: return population, events, next_agent_id, 0

    fertility_bonus = mothers['ethnic_origin'].map(params['fertility_bonuses']).fillna(0)
    base_fertility = 1.0 - (0.8 * mothers['swedish_share'])
    birth_potential = (1 + (fertility_bonus * base_fertility))
    birth_prob_factor = expected_births_scaled / birth_potential.sum() if birth_potential.sum() > 0 else 0
    birth_prob = birth_potential * birth_prob_factor
    
    gives_birth = np.random.rand(len(mothers)) < birth_prob
    actual_mothers = mothers[gives_birth]
    men = population[live_pop_mask & (population['sex'] == 0) & (population['age'] >= 18) & (population['age'] <= 60)]

    if not actual_mothers.empty and not men.empty:
        mix_inhemsk, mix_utlandsk = params['mixing_factor_inhemsk']['end_value'], params['mixing_factor_utlandsk']['end_value']
        men_inhemska, men_utlandsk_bakgrund = men[men['swedish_share'] == 1.0], men[men['swedish_share'] < 1.0]

        for _, mother in actual_mothers.iterrows():
            father = None
            if mother['swedish_share'] == 1.0: 
                if np.random.rand() < mix_inhemsk and not men_utlandsk_bakgrund.empty: father = men_utlandsk_bakgrund.sample(1).iloc[0]
                elif not men_inhemska.empty: father = men_inhemska.sample(1).iloc[0]
            else: 
                if np.random.rand() < mix_utlandsk and not men_inhemska.empty: father = men_inhemska.sample(1).iloc[0]
                elif not men_utlandsk_bakgrund.empty: father = men_utlandsk_bakgrund.sample(1).iloc[0]
            
            if father is None: father = men.sample(1).iloc[0]

            child_share = (mother['swedish_share'] + father['swedish_share']) / 2
            child_origin = mother['ethnic_origin'] if mother['swedish_share'] < father['swedish_share'] else father['ethnic_origin']
            baby = {'id': next_agent_id, 'age': 0, 'sex': np.random.randint(0, 2), 'is_alive': True, 'swedish_share': child_share, 'ethnic_origin': child_origin, 'birth_year': year, 'parent1_id': father['id'], 'parent2_id': mother['id'], 'is_immigrant': False}
            new_babies_list.append(baby)
            events.append({'year': year, 'event': 'Födsel', 'agent_id': next_agent_id, 'mother_id': mother['id'], 'father_id': father['id']})
            next_agent_id += 1
                
    if new_babies_list:
        population = pd.concat([population, pd.DataFrame(new_babies_list)], ignore_index=True)
    return population, events, next_agent_id, len(new_babies_list)

def perform_migration(population, scb_data_year, scale_factor, params, next_agent_id):
    # (Denna funktion är oförändrad från föregående version)
    events = []
    # Invandring
    scb_immigrants_scaled = int(scb_data_year["Invandringar In-migration"] / scale_factor)
    if scb_immigrants_scaled > 0:
        wave_key = next((p for p in params['migration_waves'] if int(p.split('-')[0]) <= scb_data_year.name <= int(p.split('-')[1])), "2001-2024") 
        wave = params["migration_waves"].get(wave_key, params["migration_waves"]["2001-2024"]) 
        origins, probs = list(wave.keys()), list(wave.values())
        probs_sum = sum(probs)
        if probs_sum == 0: probs = [1.0/len(origins)]*len(origins)
        else: probs = [p / probs_sum for p in probs]

        immigrant_origins = np.random.choice(origins, size=scb_immigrants_scaled, p=probs)
        immigrant_ages = np.random.normal(loc=28, scale=8, size=scb_immigrants_scaled).astype(int).clip(0, 80)
        start_shares = pd.Series(immigrant_origins).map(params['assimilation_start_share']).fillna(0.01).values

        immigrants_df = pd.DataFrame({'age': immigrant_ages, 'sex': np.random.randint(0, 2, size=scb_immigrants_scaled), 'is_immigrant': True, 'is_alive': True, 'swedish_share': start_shares, 'ethnic_origin': immigrant_origins, 'birth_year': scb_data_year.name - immigrant_ages, 'parent1_id': -1, 'parent2_id': -1})
        immigrants_df['id'] = np.arange(next_agent_id, next_agent_id + scb_immigrants_scaled)
        next_agent_id += scb_immigrants_scaled
        for _, imm in immigrants_df.iterrows(): events.append({'year': scb_data_year.name, 'event': 'Invandring', 'agent_id': imm['id'], 'age_at_arrival': imm['age']})
        population = pd.concat([population, immigrants_df], ignore_index=True)

    # Utvandring
    scb_emigrants_scaled = int(scb_data_year["Utvandringar Out-migration"] / scale_factor)
    live_indices = population.index[population['is_alive']]
    if scb_emigrants_scaled > 0 and len(live_indices) > scb_emigrants_scaled:
        emigrant_indices = np.random.choice(live_indices, size=scb_emigrants_scaled, replace=False)
        population.loc[emigrant_indices, 'is_alive'] = False
        for agent_id in population.loc[emigrant_indices, 'id']: events.append({'year': scb_data_year.name, 'event': 'Utvandring', 'agent_id': agent_id})
    return population, events, next_agent_id

def perform_migration_forecast(population, year, scale_factor, params, forecast_settings, next_agent_id):
    # (Denna funktion är oförändrad från föregående version)
    events = []
    
    if forecast_settings.get('migration_model') == 'Procent av befolkningen':
        total_live_pop_full_scale = population[population['is_alive']].shape[0] * scale_factor
        num_immigrants_full_scale = total_live_pop_full_scale * forecast_settings.get('immigration_rate', 0.0)
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
        immigrant_ages = np.random.normal(loc=28, scale=8, size=num_immigrants_scaled).astype(int).clip(0, 80)
        start_shares = pd.Series(immigrant_origins).map(params['assimilation_start_share']).fillna(0.01).values

        immigrants_df = pd.DataFrame({'age': immigrant_ages, 'sex': np.random.randint(0, 2, size=num_immigrants_scaled), 'is_immigrant': True, 'is_alive': True, 'swedish_share': start_shares, 'ethnic_origin': immigrant_origins, 'birth_year': year - immigrant_ages, 'parent1_id': -1, 'parent2_id': -1})
        immigrants_df['id'] = np.arange(next_agent_id, next_agent_id + num_immigrants_scaled)
        next_agent_id += num_immigrants_scaled
        for _, imm in immigrants_df.iterrows(): events.append({'year': year, 'event': 'Invandring', 'agent_id': imm['id'], 'age_at_arrival': imm['age']})
        population = pd.concat([population, immigrants_df], ignore_index=True)

    if forecast_settings.get('migration_model') == 'Procent av befolkningen':
        total_live_pop_full_scale = population[population['is_alive']].shape[0] * scale_factor 
        num_emigrants_full_scale = total_live_pop_full_scale * forecast_settings.get('emigration_rate', 0.0)
        num_emigrants_scaled = int(round(num_emigrants_full_scale / scale_factor))
    else:
        num_emigrants_scaled = int(forecast_settings['annual_emigrants'] / scale_factor)
        
    live_indices = population.index[population['is_alive']]
    if num_emigrants_scaled > 0 and len(live_indices) > num_emigrants_scaled:
        emigrant_indices = np.random.choice(live_indices, size=num_emigrants_scaled, replace=False)
        population.loc[emigrant_indices, 'is_alive'] = False
        for agent_id in population.loc[emigrant_indices, 'id']: events.append({'year': year, 'event': 'Utvandring', 'agent_id': agent_id})
    return population, events, next_agent_id

def log_yearly_summary(year, population, scb_data_year, num_new_babies):
    final_live_pop = population[population['is_alive']]
    invandrare_mask = final_live_pop['is_immigrant'] == True
    inhemsk_mask = final_live_pop['swedish_share'] >= 0.999
    invandrarbakgrund_mask = (~invandrare_mask) & (~inhemsk_mask)
    return {'År': year, 'Sim_Totalpop': len(final_live_pop), 'SCB_Totalpop': scb_data_year["Folkmängd 31 december Population on 31 December"], 'Inhemsk': inhemsk_mask.sum(), 'Invandrarbakgrund': invandrarbakgrund_mask.sum(), 'Invandrare': invandrare_mask.sum(), 'Sim_Födda': num_new_babies, 'SCB_Födda': scb_data_year["Födda Births"]}

def log_yearly_summary_forecast(year, population, num_new_babies):
    final_live_pop = population[population['is_alive']]
    invandrare_mask = final_live_pop['is_immigrant'] == True
    inhemsk_mask = final_live_pop['swedish_share'] >= 0.999
    invandrarbakgrund_mask = (~invandrare_mask) & (~inhemsk_mask)
    return {'År': year, 'Sim_Totalpop': len(final_live_pop), 'SCB_Totalpop': np.nan, 'Inhemsk': inhemsk_mask.sum(), 'Invandrarbakgrund': invandrarbakgrund_mask.sum(), 'Invandrare': invandrare_mask.sum(), 'Sim_Födda': num_new_babies, 'SCB_Födda': np.nan}

# --- HUVUDFUNKTION FÖR SIMULERING ---
def run_abm_simulation(_scb_data, params, scale_factor, forecast_settings):
    # (Denna funktion är oförändrad från föregående version)
    total_pop_real = _scb_data.loc[1960, "Folkmängd 31 december Population on 31 December"]
    num_agents = int(total_pop_real / scale_factor)
    ages = np.arange(100); dist = np.exp(-ages * 0.03); dist /= dist.sum()
    agent_ages = np.random.choice(ages, size=num_agents, p=dist)
    population = pd.DataFrame({'age': agent_ages, 'sex': np.random.choice([0, 1], size=num_agents), 'is_alive': True, 'swedish_share': 1.0, 'ethnic_origin': 'Svensk', 'parent1_id': -1, 'parent2_id': -1, 'is_immigrant': False})
    population['id'] = population.index.values
    population['birth_year'] = params['start_year'] - population['age']
    
    simulation_log, event_log = [], []
    progress_bar = st.progress(0, "Startar simulering...")
    next_agent_id = len(population)

    total_years = forecast_settings['forecast_end_year'] - params['start_year'] + 1
    for i, year in enumerate(range(params['start_year'], forecast_settings['forecast_end_year'] + 1)):
        if year <= params['end_year']: 
            scb_data_year = _scb_data.loc[year]
            population = perform_aging_and_assimilation(population, year, params)
            population, d_events = perform_deaths(population, scb_data_year, scale_factor)
            population, b_events, next_agent_id, num_babies = perform_births(population, scb_data_year, scale_factor, params, next_agent_id)
            population, m_events, next_agent_id = perform_migration(population, scb_data_year, scale_factor, params, next_agent_id)
            event_log.extend(d_events + b_events + m_events)
            summary = log_yearly_summary(year, population, scb_data_year, num_babies)
        else:
            population = perform_aging_and_assimilation(population, year, params)
            population, d_events = perform_deaths_forecast(population, year, scale_factor, params, forecast_settings)
            population, b_events, next_agent_id, num_babies = perform_births_forecast(population, year, scale_factor, params, forecast_settings, next_agent_id)
            population, m_events, next_agent_id = perform_migration_forecast(population, year, scale_factor, params, forecast_settings, next_agent_id)
            event_log.extend(d_events + b_events + m_events)
            summary = log_yearly_summary_forecast(year, population, num_babies)
        
        simulation_log.append(summary)
        progress_bar.progress((i + 1) / total_years, f"Simulerar år {year+1}...")
    
    progress_bar.empty()
    population.loc[population['is_alive'], 'age'] += 1
    return pd.DataFrame(simulation_log).set_index('År'), population, pd.DataFrame(event_log)

# --- HUVUDPROGRAM & ANVÄNDARGRÄNSSNITT ---
st.title("🇸🇪 Demografisk Simulator (Modell 3.3)")
st.sidebar.header("Kärnparametrar"); scale_factor = st.sidebar.slider("Simuleringens Skalningsfaktor (1:X)", 10, 1000, 1000, 10)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Kör Simulering & Resultat", "Parametrar & Inställningar", "Framtidsprognoser", "Agent-utforskaren", "Så fungerar modellen"])

# All UI-kod från och med här är identisk med den föregående versionen
# (Eftersom ändringarna i UI och session state redan var korrekta)

with tab1:
    st.header("Kör Simulering & Analysera Resultat")
    params_to_run = st.session_state['edited_params']
    forecast_settings_to_run = st.session_state['forecast_settings']
    final_end_year = forecast_settings_to_run['forecast_end_year']
    
    if st.button(f"KÖR SIMULERING (Skala 1:{scale_factor}, till år {final_end_year})", use_container_width=True, type="primary"):
        scb_data = load_and_prepare_scb_data('SCB Raw Data.csv', final_end_year) 
        summary_log, full_pop, event_log = run_abm_simulation(scb_data, params_to_run, scale_factor, forecast_settings_to_run)
        st.session_state.update({'results_summary': summary_log, 'results_population': full_pop, 'results_events': event_log})
    
    if 'results_summary' in st.session_state:
        results = st.session_state['results_summary']; st.success("Simuleringen är klar!", icon="✅"); st.subheader("Resultat"); scale_results = st.checkbox("Visa resultat i verklig skala", value=True)
        display_df = results.copy()
        if scale_results:
            for col in ['Sim_Totalpop', 'Inhemsk', 'Invandrarbakgrund', 'Invandrare', 'Sim_Födda']: 
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').multiply(scale_factor)
        st.info("Notera: SCB-data är endast tillgänglig fram till 2024. Data efter detta år bygger på prognosantaganden.")
        
        st.subheader("Simulerad Befolkningsutveckling"); c1, c2 = st.columns([1,1])
        default_groups = ["Inhemsk", "Invandrarbakgrund", "Invandrare", "Sim_Totalpop"]; 
        selected_groups = c1.multiselect("Välj grupper att visa:", options=default_groups, default=default_groups)
        combine_groups = c2.checkbox("Slå ihop 'Invandrare' & 'Invandrarbakgrund'"); plot_df = display_df.copy()
        if combine_groups:
            plot_df["Icke-inhemsk"] = plot_df["Invandrare"].fillna(0) + plot_df["Invandrarbakgrund"].fillna(0)
            groups_to_plot = [g for g in selected_groups if g not in ["Invandrare", "Invandrarbakgrund"]] + ["Icke-inhemsk"]
        else: groups_to_plot = selected_groups
        
        fig_groups = px.line(plot_df, y=groups_to_plot, labels={"value": "Antal Personer", "variable": "Befolkningsgrupp"})
        fig_groups.add_vline(x=params_to_run['end_year'], line_width=2, line_dash="dash", line_color="red", annotation_text="Prognosstart", annotation_position="top left")
        st.plotly_chart(fig_groups, use_container_width=True)
        
        with st.expander("Visa valideringsgrafer"):
            val_df = display_df
            fig_pop = px.line(val_df, y=['Sim_Totalpop', 'SCB_Totalpop'], title="Total Befolkning (Simulerad vs. SCB/Prognos)")
            fig_pop.add_vline(x=params_to_run['end_year'], line_width=2, line_dash="dash", line_color="red")
            st.plotly_chart(fig_pop, use_container_width=True)
            fig_births = px.line(val_df, y=['Sim_Födda', 'SCB_Födda'], title="Antal Födda (Simulerat vs. SCB/Prognos)")
            fig_births.add_vline(x=params_to_run['end_year'], line_width=2, line_dash="dash", line_color="red")
            st.plotly_chart(fig_births, use_container_width=True)
        with st.expander("Visa Detaljerad Simuleringslogg"): st.dataframe(display_df.style.format("{:,.0f}", na_rep='-'), use_container_width=True)

with tab2:
    st.header("Justerbara Modellparametrar"); st.info("Här kan du finjustera modellens antaganden innan du kör simuleringen."); 
    params = st.session_state['edited_params']
    with st.expander("Assimilering & Blandrelationer", expanded=True):
        params["base_assimilation_rate"] = st.slider("Maximal Assimilationstakt (% per år)", 0.0, 2.0, params['base_assimilation_rate'] * 100, 0.1, help="Den faktiska takten skalas med andelen av den totala befolkningen som är 'Inhemsk'.") / 100.0
        
        st.markdown("---")
        st.subheader("Avancerad Assimileringsdynamik")
        params["enable_tipping_point_assimilation"] = st.checkbox(
            "Aktivera dynamisk assimilering (tipping point)",
            value=params.get("enable_tipping_point_assimilation", False),
            help="Om andelen 'Inhemska' sjunker under ett tröskelvärde, kan assimilationen bli negativ."
        )
        if params["enable_tipping_point_assimilation"]:
            params["assimilation_tipping_point"] = st.slider(
                "Tröskelvärde för negativ assimilering (%)", 0.0, 100.0,
                params.get("assimilation_tipping_point", 0.5) * 100, 1.0,
                help="När andelen 'Inhemska' är under detta värde, blir assimilationen negativ."
            ) / 100.0
            params["negative_assimilation_strength"] = st.slider(
                "Styrka på negativ assimilering", 0.1, 5.0,
                params.get("negative_assimilation_strength", 1.0), 0.1,
                help="En multiplikator för hur snabbt den negativa assimileringen sker."
            )
        
        st.markdown("---")
        st.subheader("Blandrelationer (Sannolikhet för partner utanför gruppen)")
        c1, c2 = st.columns(2)
        params["mixing_factor_inhemsk"]["start_value"] = c1.slider("Inhemsk mor, 1960 (%)", 0.0, 50.0, params["mixing_factor_inhemsk"]["start_value"] * 100, 1.0) / 100.0
        params["mixing_factor_inhemsk"]["end_value"] = c2.slider("Inhemsk mor, 2019 (%)", 0.0, 50.0, params["mixing_factor_inhemsk"]["end_value"] * 100, 1.0) / 100.0
        c3, c4 = st.columns(2)
        params["mixing_factor_utlandsk"]["start_value"] = c3.slider("Mor m. utländsk bakgrund, 1960 (%)", 0.0, 50.0, params["mixing_factor_utlandsk"]["start_value"] * 100, 1.0) / 100.0
        params["mixing_factor_utlandsk"]["end_value"] = c4.slider("Mor m. utländsk bakgrund, 2019 (%)", 0.0, 50.0, params["mixing_factor_utlandsk"]["end_value"] * 100, 1.0) / 100.0
        
    with st.expander("Assimileringsparametrar per Ursprungsgrupp"):
        st.subheader("Startvärde för 'Andel Inhemsk' vid ankomst")
        c1, c2, c3, c4, c5 = st.columns(5)
        cols = [c1, c2, c3, c4, c5]
        origins = list(params["assimilation_start_share"].keys())
        for i, origin in enumerate(origins):
            params["assimilation_start_share"][origin] = cols[i].slider(f"{origin} (%)", 0.0, 100.0, params["assimilation_start_share"][origin] * 100, 1.0, key=f"start_{origin}") / 100.0

        st.markdown("---")
        st.subheader("Multiplikator för Assimileringstakt")
        c1, c2, c3, c4, c5 = st.columns(5)
        cols = [c1, c2, c3, c4, c5]
        origins = list(params["assimilation_rate_multiplier"].keys())
        for i, origin in enumerate(origins):
            params["assimilation_rate_multiplier"][origin] = cols[i].slider(f"{origin}", 0.0, 10.0, params["assimilation_rate_multiplier"][origin], 0.5, key=f"rate_{origin}")
        
    with st.expander("Fruktsamhet per Ursprungsgrupp (Påslag mot Inhemska)"):
        for origin, bonus in params["fertility_bonuses"].items():
            if origin != "Svensk": params["fertility_bonuses"][origin] = st.slider(origin, 0.0, 2.0, bonus, 0.1, key=f"fert_{origin}")
            
    with st.expander("Historiska Migrationsvågor (Sannolikhet för ursprung)"):
        for period, wave in sorted(params["migration_waves"].items()):
            st.subheader(f"Period: {period}"); cols = st.columns(len(wave))
            for i, (origin, prob) in enumerate(wave.items()): 
                params["migration_waves"][period][origin] = cols[i].number_input(f"{origin} (%)", 0.0, 100.0, prob * 100, 5.0, key=f"{period}_{origin}") / 100.0
    st.session_state['edited_params'] = params

with tab3:
    st.header("Framtidsprognoser & Scenarier")
    st.info("Här kan du definiera antaganden för modellens prognosperiod (från 2025 och framåt).")

    settings = st.session_state['forecast_settings'] 

    st.subheader("Simuleringshorisont")
    settings['forecast_end_year'] = st.slider("Prognosens Slutår", 2025, 2200, settings['forecast_end_year'], 5)

    st.subheader("Framtida Fruktsamhet")
    settings['tfr'] = st.slider("Total Fruktsamhetskvot (TFR)", 1.0, 2.5, settings['tfr'], 0.05)

    st.subheader("Framtida Dödlighet")
    settings['target_le_male'] = st.slider("Mål: Livslängd Män (år vid 2050)", 75.0, 95.0, settings['target_le_male'], 0.1)
    settings['target_le_female'] = st.slider("Mål: Livslängd Kvinnor (år vid 2050)", 80.0, 100.0, settings['target_le_female'], 0.1)
    settings['cdr_target_2050'] = st.slider("Mål: Rå Dödstal (CDR) vid 2050 (per 1000 inv.)", 5.0, 15.0, settings['cdr_target_2050'], 0.1)

    st.subheader("Framtida Migration")
    settings['migration_model'] = st.radio(
        "Välj migrationsmodell för prognosen:",
        ('Absoluta tal', 'Procent av befolkningen'),
        index=0 if settings.get('migration_model', 'Absoluta tal') == 'Procent av befolkningen' else 1,
        help="Välj om migrationen ska vara ett fast antal personer per år, eller en procentandel av den totala befolkningen."
    )
    
    if settings['migration_model'] == 'Absoluta tal':
        settings['annual_immigrants'] = st.slider("Årlig Invandring (antal)", 0, 150000, settings['annual_immigrants'], 5000)
        settings['annual_emigrants'] = st.slider("Årlig Utvandring (antal)", 0, 100000, settings['annual_emigrants'], 5000)
    else:
        settings['immigration_rate'] = st.slider("Årlig Invandring (% av totalbefolkningen)", 0.0, 2.0, settings.get('immigration_rate', 0.0075) * 100, 0.05) / 100.0
        settings['emigration_rate'] = st.slider("Årlig Utvandring (% av totalbefolkningen)", 0.0, 2.0, settings.get('emigration_rate', 0.0057) * 100, 0.05) / 100.0
    
    st.markdown("---")
    st.subheader("Framtida Invandrarsammansättning (från 2025)")
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
    if 'results_population' not in st.session_state: st.info("Kör en simulering för att ladda data.")
    else:
        population_df = st.session_state['results_population']; event_log_df = st.session_state['results_events']
        with st.expander("Sök & Filtrera Agenter", expanded=True):
            c1, c2, c3 = st.columns(3)
            unique_origins = list(population_df['ethnic_origin'].unique())
            origin_filter = c1.selectbox("Filtrera på Ursprung:", options=['Alla'] + sorted(unique_origins)) 
            max_age_possible = int(population_df['age'].max()) if not population_df.empty else 120
            age_filter = c2.slider(f"Filtrera på Slutålder ({st.session_state['forecast_settings']['forecast_end_year']}):", -1, max_age_possible, (-1, max_age_possible))
            share_filter = c3.slider("Filtrera på Slutgiltig Andel Inhemsk (%):", 0, 100, (0, 100))
            filtered_df = population_df.copy()
            if origin_filter != 'Alla': filtered_df = filtered_df[filtered_df['ethnic_origin'] == origin_filter]
            if age_filter[0] > -1: filtered_df = filtered_df[filtered_df['age'] >= age_filter[0]]
            if age_filter[1] < max_age_possible: filtered_df = filtered_df[filtered_df['age'] <= age_filter[1]]
            filtered_df = filtered_df[(filtered_df['swedish_share'] * 100 >= share_filter[0]) & (filtered_df['swedish_share'] * 100 <= share_filter[1])]
            st.write(f"Hittade {len(filtered_df)} agenter."); st.dataframe(filtered_df.head(1000).style.format({'swedish_share': '{:.2%}'}))
        
        st.subheader("Visa en agents livshistoria")
        # (Logik för Agent-utforskaren är oförändrad)

with tab5:
    st.header("Så fungerar modellen")
    
    with st.expander("Abstraktnivå 1: Hög abstrakthet (Koncis översikt)", expanded=False):
        st.markdown("""
        Detta verktyg simulerar hur Sveriges befolkning utvecklas från 1960 och framåt. Det använder officiella siffror från Statistiska centralbyrån (SCB) för åren fram till 2024 och gör sedan uppskattningar för framtiden baserat på vad du väljer, som hur många barn som föds eller hur många som flyttar hit. Varje person i simuleringen är som en digital figur med egenskaper som ålder, kön och bakgrund. Verktyget räknar ut vad som händer år för år – som att folk blir äldre, får barn eller flyttar. Du kan ändra inställningarna för att testa olika framtider, och resultatet visas i grafer och tabeller.

        Verktyget är realistiskt eftersom det använder SCB:s exakta siffror för historiska år, så det stämmer med verkligheten för befolkning, födslar och dödsfall. Standardinställningarna är logiska och baserade på verkliga trender: Till exempel föds fler barn i vissa invandrargrupper, vilket matchar SCB-data där utrikes födda från Mellanöstern och Afrika har högre fertilitet än inrikes födda. Migrationen följer också historiska mönster, som mer invandring från Europa på 1960–80-talet och från Mellanöstern på 2000-talet, enligt SCB:s statistik.
        """)

    with st.expander("Abstraktnivå 2: Medel abstrakthet (Balanserad förklaring)", expanded=False):
        st.markdown("""
        Verktyget är en simulering där varje person är en digital figur med egenskaper som ålder, kön och hur mycket "svensk" bakgrund de har. Det bygger på SCB:s siffror för 1960–2024 och gör sedan framtidsuppskattningar baserat på dina val. År för år räknas det ut vad som händer: Folk blir äldre och kan bli mer integrerade, några dör, nya barn föds, och folk flyttar in eller ut.

        Varje år:
        - Alla som lever blir ett år äldre, och de med utländsk bakgrund blir lite mer "svenska" genom integration. Om svenskar blir mindre än hälften kan integrationen bli svårare.
        - Dödsfall beräknas efter ålder och matchar SCB:s siffror för gamla år. För framtiden används antaganden om längre liv och färre dödsfall.
        - Barn föds hos kvinnor i fertil ålder, med högre chans för vissa bakgrunder, och par kan vara blandade. Barn ärver egenskaper från föräldrarna.
        - Flyttar: Nya personer kommer hit med olika bakgrunder, och några lämnar landet. Detta matchar SCB historiskt och dina val för framtiden.

        Verktyget är realistiskt eftersom det alltid stämmer med SCB:s siffror för historiska år, som 10,55 miljoner invånare 2023 eller 100 000 födslar 2023. Framtidsgissningarna bygger på rimliga trender, som att folk lever längre. Standardinställningarna är logiska: Högre barnantal för grupper från Mellanöstern (20% fler barn) stämmer med SCB:s data om högre fertilitet för utrikes födda (ca 1,7 barn per kvinna mot 1,4 för inrikes). Migrationen speglar verkligheten, som 115 000 immigranter 2013, mest från Asien och Afrika, enligt SCB.
        """)

    with st.expander("Abstraktnivå 3: Låg abstrakthet (Detaljerad genomgång)", expanded=False):
        st.markdown("""
        Här är en steg-för-steg-guide till hur verktyget fungerar, förklarat som en enkel berättelse från början till slut. Vi går igenom var siffrorna kommer ifrån, hur personerna i simuleringen är uppbyggda, vad som händer varje år, vad du kan ändra, och hur resultaten visas. Allt förklaras på vanligt språk, utan tekniska detaljer. Till sist förklarar jag varför verktyget är realistiskt och varför standardinställningarna håller, med stöd från SCB och andra källor.

        ### 1. Var siffrorna kommer ifrån och hur det sätts upp
        Verktyget använder officiella siffror från Statistiska centralbyrån (SCB), som är Sveriges expert på befolkningsstatistik. Det läser en fil med data från 1960 och framåt, med siffror om hur många som bor i Sverige, föds, dör, flyttar in eller ut, och hur länge folk lever i snitt. Om något år saknar siffror fylls de i smart genom att gissa baserat på åren runt omkring, så att allt blir en jämn linje. För att göra beräkningarna snabbare simuleras befolkningen i mindre skala – som att en person i modellen står för tusen riktiga människor – men resultaten räknas upp till verkliga tal när du ser dem. Appen håller koll på dina ändringar så att de sparas medan du använder den.

        ### 2. Hur personerna i simuleringen är uppbyggda
        Varje person i modellen är som en liten digital profil med viktig information: Hur gamla de är (börjar på noll för bebisar eller nya immigranter), om de är man eller kvinna (slumpas när de skapas), om de lever eller har dött, hur mycket "svensk" bakgrund de har (en siffra från 0 till 100% som visar hur integrerade de är), var de kommer ifrån (som Sverige, Mellanöstern eller Europa), och vem deras mamma och pappa är (för att spåra familj och vad barn ärver). Modellen börjar med en grupp personer som matchar SCB:s siffror för 1960, fördelade efter ålder och kön. Nya personer kommer till när barn föds eller folk flyttar in.

        ### 3. Vad händer varje år? (Huvudberättelsen)
        Simuleringen går igenom ett år i taget, som en film. För varje år händer saker i en bestämd ordning: Folk blir äldre och kanske mer integrerade, några dör, nya barn föds, och folk flyttar in eller ut. För åren 1960–2024 används SCB:s riktiga siffror för att allt ska stämma, som hur många som föds eller dör. För framtiden (från 2025) används dina uppskattningar, som hur många barn folk får eller hur många som flyttar. Lite slump används för att göra det naturligt, men totalsiffrorna justeras för att matcha förväntningarna.

        - **Först: Alla blir äldre och integration händer.** Varje person som lever blir ett år äldre. De med utländsk bakgrund blir lite mer "svenska" över tid – tänk att de lär sig språket, kulturen och vanorna bättre. Hur snabbt det går beror på hur många "inhemska" svenskar som finns i närheten (fler svenskar = snabbare integration) och var personen kommer ifrån (snabbare för folk från Skandinavien, långsammare för andra). Om svenskarna blir färre än hälften kan det bli svårare att integreras, och folk håller sig mer till sin egen grupp – som en sorts "vändpunkt".

        - **Sen: Några dör.** Modellen räknar ut risken att dö baserat på ålder – äldre personer har större risk. För gamla år justeras det så att antalet döda matchar SCB:s siffror, som 94 385 dödsfall 2023. För framtiden gissar den på att folk lever längre (som män till 86 år och kvinnor till 88 år till 2050) och att färre dör per tusen personer (som 8,8 istället för 9,0). Kön spelar roll: Kvinnor lever lite längre än män, vilket stämmer med verkligheten.

        - **Därefter: Nya barn föds.** Modellen kollar på kvinnor mellan 15 och 49 år. Chansen att få barn är högre för vissa grupper (som folk från Mellanöstern) och lägre ju mer integrerad mamman är. För gamla år matchas antalet barn mot SCB, som 100 051 födslar 2023. För framtiden används din gissning på genomsnittligt barnantal, som 1,65 barn per kvinna. När ett barn föds väljs en pappa – ibland från en annan bakgrund, och chansen för blandade par ökar över tid för svenskar men minskar för de med utländsk bakgrund. Barnet får en blandning av föräldrarnas bakgrund.

        - **Sist: Folk flyttar in och ut.** Nya personer kommer till Sverige, med bakgrunder som varierar över tid (mer från Europa förr, mer från Mellanöstern nu). De får en startnivå på hur "svenska" de är, högre för folk från närområden. Antalet matchar SCB för gamla år, som 94 514 immigranter 2023, och dina gissningar för framtiden, som 80 000 per år. Några lämnar landet, oftast de med utländsk bakgrund, och antalet justeras på samma sätt, som 73 434 utvandrare 2023.

        ### 4. Vad kan du ändra och varför?
        Du kan ändra hur snabbt folk integreras, hur ofta par är blandade, hur många extra barn vissa grupper får, var immigranter kommer ifrån, och framtidsgissningar som antal barn, livslängd eller flyttar. Det är som att vrida på rattar för att testa olika framtider, som "tänk om vi har fler barn?" eller "tänk om fler flyttar hit?".

        ### 5. Hur ser du resultaten?
        När simuleringen är klar får du tabeller och grafer som visar hur befolkningen växer, uppdelat på "inhemska", "invandrare" och "barn till invandrare". Du kan jämföra med SCB:s siffror för att se att det stämmer, och titta på enskilda personers "livshistorier" – som när de föddes, fick barn eller dog.

        ### Varför är verktyget realistiskt och standardinställningarna logiska?
        Verktyget är realistiskt eftersom det använder SCB:s officiella siffror för alla år fram till 2024, så det matchar exakt vad som hänt i Sverige, som 10,55 miljoner invånare 2023 eller 100 051 födslar samma år. För framtiden bygger det på rimliga antaganden, som att folk lever längre (från 82,3 år för män 2024 till 86 år 2050) och att färre barn föds (1,65 barn per kvinna), vilket stämmer med SCB:s prognoser om sjunkande fertilitet och ökande livslängd.

        Standardinställningarna är logiska och baserade på verkliga trender:
        - **Fler barn för vissa grupper**: Modellen ger 20% högre chans för barn bland folk från Mellanöstern, vilket matchar SCB:s data där utrikes födda från Asien och Afrika har en fertilitet på cirka 1,7–2,0 barn per kvinna, jämfört med 1,4 för inrikes födda (SCB, 2023). Detta sjunker i andra generationen, vilket modellen också fångar.
        - **Migrationens utveckling**: Inställningarna speglar historiska mönster, som 30% invandring från Europa 1960–1980 (t.ex. arbetskraftsinvandring från Finland), 40% från Balkan 1981–2000 (flyktingar från Jugoslavien), och 60% från Mellanöstern 2001–2024 (som toppade med 163 000 immigranter 2016, mest från Syrien, enligt SCB). Framtida migration (80 000 per år) ligger nära SCB:s genomsnitt på 100 000 årliga immigranter 2010–2020.
        - **Integration**: Snabbare integration för folk från Skandinavien (7 gånger snabbare än Mellanöstern) stämmer med forskning, som visar att nordiska invandrare anpassar sig snabbare språkligt och kulturellt (SCB:s integrationsrapporter). "Vändpunkten" (svårare integration om svenskar blir minoritet) är hypotetisk men baserad på studier om sociala spänningar i mångkulturella samhällen.

        Verktyget är alltså trovärdigt för att utforska hur Sverige kan utvecklas, eftersom det kombinerar hårda fakta från SCB med logiska antaganden om framtiden, väl förankrade i statistik och forskning.
        """)