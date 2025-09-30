import pandas as pd
import streamlit as st

@st.cache_data
def load_scb_data(filepath, max_year):
    """Lastar, rensar och interpolerar SCB-data från en CSV-fil."""
    df = pd.read_csv(filepath, index_col=0, na_values="..")
    df = df.transpose()
    df.index = pd.to_numeric(df.index)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.sort_index()
    full_yearly_index = pd.RangeIndex(start=1960, stop=max_year + 1, name='År')
    df_reindexed = df.reindex(full_yearly_index)
    df_interpolated = df_reindexed.interpolate(method='spline', order=2, limit_direction='both')
    
    return df_interpolated