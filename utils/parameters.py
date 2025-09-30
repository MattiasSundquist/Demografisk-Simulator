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