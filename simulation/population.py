# simulation/population.py

import pandas as pd
import numpy as np

# --- MODELLKONSTANTER (relevanta för denna modul) ---
ASSIMILATION_THRESHOLD = 0.999
FERTILITY_AGE_MIN = 15
FERTILITY_AGE_MAX = 49
PARTNER_AGE_MIN = 18
PARTNER_AGE_MAX = 60
IMMIGRANT_AGE_MEAN = 28
IMMIGRANT_AGE_STD_DEV = 8

class Population:
    """Hanterar den samlade populationen av agenter och alla demografiska processer."""
    def __init__(self, initial_agents_df):
        self.agents = initial_agents_df
        self.next_agent_id = len(initial_agents_df)
        self.event_log = []

    def perform_aging_and_assimilation(self, year, params):
        self.agents.loc[self.agents['is_alive'], 'age'] += 1
        live_pop_mask = self.agents['is_alive']
        if live_pop_mask.sum() == 0:
            return

        proportion_swedish_background = (live_pop_mask & (self.agents['swedish_share'] >= ASSIMILATION_THRESHOLD)).sum() / live_pop_mask.sum()
        origin_assimilation_multiplier = self.agents['ethnic_origin'].map(params['assimilation']['rate_multipliers']).fillna(1.0)
        
        dynamic_assimilation_factor = (proportion_swedish_background * 2) - 1
        assimilation_strength = params['assimilation']['strength_multiplier']
        
        assimilation_change = (params['assimilation']['base_rate'] * 
                               dynamic_assimilation_factor * 
                               assimilation_strength *
                               self.agents['swedish_share'] * 
                               origin_assimilation_multiplier)

        change_mask = live_pop_mask & (self.agents['swedish_share'] > 0) & (self.agents['swedish_share'] < 1.0)
        self.agents.loc[change_mask, 'swedish_share'] += assimilation_change
        self.agents['swedish_share'] = self.agents['swedish_share'].clip(0.0, 1.0)

    def perform_deaths(self, scb_data_year, scale_factor):
        live_pop_indices = self.agents.index[self.agents['is_alive']]
        if live_pop_indices.empty:
            return

        mortality_rate = 0.001 + (self.agents.loc[live_pop_indices, 'age'] / 100) ** 4
        scb_deaths_scaled = scb_data_year["Döda Deaths"] / scale_factor
        sim_deaths_expected = mortality_rate.sum()
        death_adj_factor = scb_deaths_scaled / sim_deaths_expected if sim_deaths_expected > 0 else 0
        death_prob = mortality_rate * death_adj_factor
        
        dies_mask = np.random.rand(len(live_pop_indices)) < death_prob
        newly_dead_indices = live_pop_indices[dies_mask]
        
        self.agents.loc[newly_dead_indices, 'is_alive'] = False
        for agent_id in self.agents.loc[newly_dead_indices, 'id']:
            self.event_log.append({'year': scb_data_year.name, 'event': 'Död', 'agent_id': agent_id})
    
    def perform_deaths_forecast(self, year, scale_factor, params, forecast_settings):
        live_pop_indices = self.agents.index[self.agents['is_alive']]
        if live_pop_indices.empty:
            return

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

        mortality_rate_base = 0.001 + (self.agents.loc[live_pop_indices, 'age'] / 100) ** 4
        adj_factor_male_le = (le_m_2024 / current_le_m) if current_le_m > 0 else 1.0
        adj_factor_female_le = (le_f_2024 / current_le_f) if current_le_f > 0 else 1.0

        mortality_rate_adjusted = mortality_rate_base.copy()
        male_mask = self.agents.loc[live_pop_indices, 'sex'] == 0
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

        total_current_population_full_scale = self.agents[self.agents['is_alive']].shape[0] * scale_factor
        target_deaths_full_scale = total_current_population_full_scale * (current_cdr / 1000)
        target_deaths_scaled = int(round(target_deaths_full_scale / scale_factor))

        sim_deaths_expected_base = mortality_rate_adjusted.sum()
        death_adj_factor = target_deaths_scaled / sim_deaths_expected_base if sim_deaths_expected_base > 0 else 0
        death_prob = (mortality_rate_adjusted * death_adj_factor).clip(0.0, 1.0)

        dies_mask = np.random.rand(len(live_pop_indices)) < death_prob.values
        newly_dead_indices = live_pop_indices[dies_mask]
        self.agents.loc[newly_dead_indices, 'is_alive'] = False
        for agent_id in self.agents.loc[newly_dead_indices, 'id']:
            self.event_log.append({'year': year, 'event': 'Död', 'agent_id': agent_id})
    
    def perform_births(self, scb_data_year, scale_factor, params):
        new_babies_list = []
        live_pop_mask = self.agents['is_alive']
        mothers = self.agents[live_pop_mask & (self.agents['sex'] == 1) & (self.agents['age'] >= FERTILITY_AGE_MIN) & (self.agents['age'] <= FERTILITY_AGE_MAX)]

        if not mothers.empty:
            scb_births_scaled = scb_data_year["Födda Births"] / scale_factor
            fertility_bonus = mothers['ethnic_origin'].map(params['fertility']['bonuses']).fillna(0)
            base_fertility = 1.0 - (0.8 * mothers['swedish_share'])
            birth_potential = (1 + (fertility_bonus * base_fertility))
            birth_prob_factor = scb_births_scaled / birth_potential.sum() if birth_potential.sum() > 0 else 0
            birth_prob = birth_potential * birth_prob_factor
            gives_birth = np.random.rand(len(mothers)) < birth_prob
            actual_mothers = mothers[gives_birth]
            men = self.agents[live_pop_mask & (self.agents['sex'] == 0) & (self.agents['age'] >= PARTNER_AGE_MIN) & (self.agents['age'] <= PARTNER_AGE_MAX)]

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
                    
                    baby = {'id': self.next_agent_id, 'age': 0, 'sex': np.random.randint(0, 2), 'is_alive': True, 'swedish_share': child_share, 'ethnic_origin': child_origin, 'birth_year': year, 'parent1_id': father['id'], 'parent2_id': mother['id'], 'is_immigrant': False}
                    new_babies_list.append(baby)
                    self.event_log.append({'year': year, 'event': 'Födsel', 'agent_id': self.next_agent_id, 'mother_id': mother['id'], 'father_id': father['id']})
                    self.next_agent_id += 1
                    
        if new_babies_list:
            self.agents = pd.concat([self.agents, pd.DataFrame(new_babies_list)], ignore_index=True)
        return len(new_babies_list)

    def perform_births_forecast(self, year, scale_factor, params, forecast_settings):
        new_babies_list = []
        live_pop_mask = self.agents['is_alive']
        mothers = self.agents[live_pop_mask & (self.agents['sex'] == 1) & (self.agents['age'] >= FERTILITY_AGE_MIN) & (self.agents['age'] <= FERTILITY_AGE_MAX)]
        if mothers.empty: return 0

        num_fertile_women_full_scale = len(mothers) * scale_factor
        annual_birth_rate_per_woman = forecast_settings['tfr'] / (FERTILITY_AGE_MAX - FERTILITY_AGE_MIN + 1)
        expected_births_full_scale = num_fertile_women_full_scale * annual_birth_rate_per_woman
        expected_births_scaled = int(round(expected_births_full_scale / scale_factor))
        if expected_births_scaled <= 0: return 0

        fertility_bonus = mothers['ethnic_origin'].map(params['fertility']['bonuses']).fillna(0)
        base_fertility = 1.0 - (0.8 * mothers['swedish_share'])
        birth_potential = (1 + (fertility_bonus * base_fertility))
        birth_prob_factor = expected_births_scaled / birth_potential.sum() if birth_potential.sum() > 0 else 0
        birth_prob = birth_potential * birth_prob_factor
        
        gives_birth = np.random.rand(len(mothers)) < birth_prob
        actual_mothers = mothers[gives_birth]
        men = self.agents[live_pop_mask & (self.agents['sex'] == 0) & (self.agents['age'] >= PARTNER_AGE_MIN) & (self.agents['age'] <= PARTNER_AGE_MAX)]

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
                
                baby = {'id': self.next_agent_id, 'age': 0, 'sex': np.random.randint(0, 2), 'is_alive': True, 'swedish_share': child_share, 'ethnic_origin': child_origin, 'birth_year': year, 'parent1_id': father['id'], 'parent2_id': mother['id'], 'is_immigrant': False}
                new_babies_list.append(baby)
                self.event_log.append({'year': year, 'event': 'Födsel', 'agent_id': self.next_agent_id, 'mother_id': mother['id'], 'father_id': father['id']})
                self.next_agent_id += 1
                    
        if new_babies_list:
            self.agents = pd.concat([self.agents, pd.DataFrame(new_babies_list)], ignore_index=True)
        return len(new_babies_list)

    def perform_migration(self, scb_data_year, scale_factor, params):
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

            immigrants_df = pd.DataFrame({
                'age': immigrant_ages, 'sex': np.random.randint(0, 2, size=scb_immigrants_scaled), 
                'is_immigrant': True, 'is_alive': True, 'swedish_share': start_shares, 
                'ethnic_origin': immigrant_origins, 'birth_year': scb_data_year.name - immigrant_ages, 
                'parent1_id': -1, 'parent2_id': -1
            })
            immigrants_df['id'] = np.arange(self.next_agent_id, self.next_agent_id + scb_immigrants_scaled)
            self.next_agent_id += scb_immigrants_scaled
            
            for _, imm in immigrants_df.iterrows(): 
                self.event_log.append({'year': scb_data_year.name, 'event': 'Invandring', 'agent_id': imm['id'], 'age_at_arrival': imm['age']})
            self.agents = pd.concat([self.agents, immigrants_df], ignore_index=True)

        # Utvandring
        scb_emigrants_scaled = int(scb_data_year["Utvandringar Out-migration"] / scale_factor)
        live_indices = self.agents.index[self.agents['is_alive']]
        if scb_emigrants_scaled > 0 and len(live_indices) > scb_emigrants_scaled:
            emigration_bias = params['migration']['emigration_bias']
            emigration_weights = 1 + (1 - self.agents.loc[live_indices, 'swedish_share']) * (emigration_bias - 1)
            probabilities = emigration_weights / emigration_weights.sum()
            emigrant_indices = np.random.choice(live_indices, size=scb_emigrants_scaled, replace=False, p=probabilities.values)
            self.agents.loc[emigrant_indices, 'is_alive'] = False
            for agent_id in self.agents.loc[emigrant_indices, 'id']: 
                self.event_log.append({'year': scb_data_year.name, 'event': 'Utvandring', 'agent_id': agent_id})
    
    def perform_migration_forecast(self, year, scale_factor, params, forecast_settings):
        if forecast_settings['migration_model'] == 'Procent av befolkningen':
            total_live_pop_full_scale = self.agents[self.agents['is_alive']].shape[0] * scale_factor
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

            immigrants_df = pd.DataFrame({
                'age': immigrant_ages, 'sex': np.random.randint(0, 2, size=num_immigrants_scaled), 
                'is_immigrant': True, 'is_alive': True, 'swedish_share': start_shares, 
                'ethnic_origin': immigrant_origins, 'birth_year': year - immigrant_ages, 
                'parent1_id': -1, 'parent2_id': -1
            })
            immigrants_df['id'] = np.arange(self.next_agent_id, self.next_agent_id + num_immigrants_scaled)
            self.next_agent_id += num_immigrants_scaled
            
            for _, imm in immigrants_df.iterrows(): 
                self.event_log.append({'year': year, 'event': 'Invandring', 'agent_id': imm['id'], 'age_at_arrival': imm['age']})
            self.agents = pd.concat([self.agents, immigrants_df], ignore_index=True)

        if forecast_settings['migration_model'] == 'Procent av befolkningen':
            total_live_pop_full_scale = self.agents[self.agents['is_alive']].shape[0] * scale_factor 
            num_emigrants_full_scale = total_live_pop_full_scale * forecast_settings['emigration_rate']
            num_emigrants_scaled = int(round(num_emigrants_full_scale / scale_factor))
        else:
            num_emigrants_scaled = int(forecast_settings['annual_emigrants'] / scale_factor)
            
        live_indices = self.agents.index[self.agents['is_alive']]
        if num_emigrants_scaled > 0 and len(live_indices) > num_emigrants_scaled:
            emigration_bias = params['migration']['emigration_bias']
            emigration_weights = 1 + (1 - self.agents.loc[live_indices, 'swedish_share']) * (emigration_bias - 1)
            probabilities = emigration_weights / emigration_weights.sum()
            emigrant_indices = np.random.choice(live_indices, size=num_emigrants_scaled, replace=False, p=probabilities.values)
            self.agents.loc[emigrant_indices, 'is_alive'] = False
            for agent_id in self.agents.loc[emigrant_indices, 'id']: 
                self.event_log.append({'year': year, 'event': 'Utvandring', 'agent_id': agent_id})