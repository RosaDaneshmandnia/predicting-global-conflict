import os
import pandas as pd
import geopandas as gpd
from utils import data_cleaning, map_admin_regions
from config import settings
from utils.features.holidays import add_holiday_features
from utils.features.worldbank import add_worldbank_features
from utils.features.religion import add_religion_features
from utils.features.risk_merge import RiskIndicatorMerger
from utils.features.hol_rel import add_holiday_religion_features, get_new_feature_names
from utils.features.WBD_features import add_worldbank_engineered_features


def prepare_data_pipeline(clean_data: bool = False):
    """
    Builds or loads the model-ready DataFrame.
    If clean_data=True, load from saved file. Otherwise, run the full pipeline.
    """
    output_path = "data/processed/model_data.csv"

    if not clean_data and os.path.exists(output_path):
        print("Loading cleaned data from disk...")
        df = pd.read_csv(output_path, index_col=[0, 1])
        return df

    print("Running full data preprocessing pipeline...")
    df = pd.read_csv("data/raw/1997-01-01-2025-07-03.csv")
    df = df[df['year'] >= 2018].copy()
    df['date'] = pd.to_datetime(df['event_date'], format='%d %B %Y')
    df['month_year'] = df['date'].dt.to_period('M').dt.to_timestamp()

    gdf = gpd.read_file("data/raw/boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp")
    
    if 'country_code' not in df.columns:
       df['country_code'] = df['country']

# admin1 column naming fix
    if 'admin1' not in df.columns:
        if 'admin1_name' in df.columns:
            df['admin1'] = df['admin1_name']
        elif 'admin1_region' in df.columns:
            df['admin1'] = df['admin1_region']

    df_neighbours = map_admin_regions.add_admin1_neighbors(df, gdf)
     

    neighbour_data = data_cleaning.summarise_neighbour_events(df_neighbours)
    event_data = data_cleaning.get_monthly_events(df_neighbours)
    subevent_data = data_cleaning.get_monthly_subevents(
       df_neighbours, ['Excessive force against protesters', 'Agreement']
    )

    combined = pd.concat([event_data, subevent_data], axis=1).join(neighbour_data, how='left')
    combined = data_cleaning.add_lagged_columns(combined)
    combined = data_cleaning.add_time_trend_features(combined)
    combined = data_cleaning.add_importance_weights(combined)

    combined = add_worldbank_features(combined, gdf)
    wb_cols = [c for c in ['inflation', 'youth_unemployment', 'income_inequality', 'income_level_code'] if c in combined.columns]
    lagged_wb = combined[wb_cols].groupby(level='matched_admin1_id').shift(1)
    lagged_wb.columns = [f"{c} (t-1)" for c in lagged_wb.columns]
    combined = combined.drop(columns=wb_cols)
    combined = pd.concat([combined, lagged_wb], axis=1)

    combined = add_holiday_features(combined, gdf)
    holiday_cols = [c for c in combined.columns if 'holiday_count' in c]
    lagged_hol = combined[holiday_cols].groupby(level='matched_admin1_id').shift(1)
    lagged_hol.columns = [f"{c} (t-1)" for c in lagged_hol.columns]
    combined = pd.concat([combined, lagged_hol], axis=1)
    
    combined = add_religion_features(combined)
    combined = add_holiday_religion_features(combined)
    combined = add_worldbank_engineered_features(combined)

    


    model_data = combined[settings.predictors + settings.targets + ['importance_weight']]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_data.to_csv(output_path)

        # Risk merger reads from saved file, enriches it, saves again
    merger = RiskIndicatorMerger()
    model_data = merger.merge(output_path, "data/raw/master_raw.csv")
    model_data.to_csv(output_path, index=False)


    return model_data
    



def filter_admin1_data(df, admin1_region):
    return df.loc[admin1_region]