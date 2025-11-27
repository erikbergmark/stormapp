import streamlit as st
import pandas as pd
import numpy as np
import joblib

counties = pd.read_csv("illinois_counties.csv")
counties = counties.drop(["fips"], axis=1)
counties["county"] = counties["county"].str.title()

#create full dataframe from before
year2025 = pd.read_csv("StormEvents_details-ftp_v1.0_d2025_c20250818.csv")
year2024 = pd.read_csv("StormEvents_details-ftp_v1.0_d2024_c20250818.csv")
year2023 = pd.read_csv("StormEvents_details-ftp_v1.0_d2023_c20250731.csv")
year2022 = pd.read_csv("StormEvents_details-ftp_v1.0_d2022_c20250721.csv")
year2021 = pd.read_csv("StormEvents_details-ftp_v1.0_d2021_c20250520.csv")
year2020 = pd.read_csv("StormEvents_details-ftp_v1.0_d2020_c20250702.csv")
year2019 = pd.read_csv("StormEvents_details-ftp_v1.0_d2019_c20250520.csv")
year2018 = pd.read_csv("StormEvents_details-ftp_v1.0_d2018_c20250520.csv")
year2017 = pd.read_csv("StormEvents_details-ftp_v1.0_d2017_c20250520.csv")
year2016 = pd.read_csv("StormEvents_details-ftp_v1.0_d2016_c20250818.csv")
year2015 = pd.read_csv("StormEvents_details-ftp_v1.0_d2015_c20250818.csv")
storm_full = pd.concat([year2015, year2016, year2017, year2018, year2019, year2020, year2021, year2022, year2023, year2024, year2025], ignore_index=True)
illinois_full = storm_full[storm_full["STATE"] == "ILLINOIS"]
illinois= illinois_full[(illinois_full["BEGIN_YEARMONTH"] >= 201510) & 
                     (illinois_full["BEGIN_YEARMONTH"] <= 202510)]
illinois = illinois.drop(["END_YEARMONTH", "END_DAY", "END_TIME", "EPISODE_ID", "STATE", "STATE_FIPS", "END_LOCATION","EPISODE_NARRATIVE", "EVENT_NARRATIVE", "DATA_SOURCE", "BEGIN_LOCATION", "CATEGORY", "TOR_F_SCALE", "TOR_LENGTH", "TOR_WIDTH", "TOR_OTHER_WFO", "TOR_OTHER_CZ_STATE", "TOR_OTHER_CZ_FIPS", "TOR_OTHER_CZ_NAME", "BEGIN_DATE_TIME", "END_DATE_TIME", "CZ_TIMEZONE"], axis=1)
illinois["DAMAGE_PROPERTY_NUM"] = (
      illinois["DAMAGE_PROPERTY"].str.replace("K", "e3", regex=False)
                           .str.replace("M", "e6", regex=False)
                           .str.replace("B", "e9", regex=False)
                           .astype(float)
)
illinois = illinois.dropna(subset=["DAMAGE_PROPERTY_NUM"])
illinois["DAMAGE_OCCURRED"] = (illinois["DAMAGE_PROPERTY_NUM"] > 0).astype(int)
county_map = {
    "CENTRAL COOK COUNTY": "COOK",
    "NORTHERN COOK COUNTY": "COOK",
    "SOUTHERN COOK COUNTY": "COOK",
    "NORTHERN WILL COUNTY": "WILL",
    "SOUTHERN WILL COUNTY": "WILL",
    "EASTERN WILL COUNTY": "WILL",
    "CENRTRAL COOK COUNTY": "COOK"
}
illinois["CZ_NAME"].replace(county_map, inplace=True)
illinois["CZ_NAME"] = illinois["CZ_NAME"].str.title()
full = pd.merge(illinois, counties, left_on="CZ_NAME", right_on="county", how = "inner")
full = full.drop(["MAGNITUDE", "MAGNITUDE_TYPE", "FLOOD_CAUSE", "BEGIN_RANGE", "BEGIN_AZIMUTH", "END_RANGE", "END_AZIMUTH", "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"], axis=1)
full_log = full.drop(["EVENT_ID", "CZ_TYPE", "CZ_FIPS", "DAMAGE_CROPS", "DAMAGE_PROPERTY_NUM", "DAMAGE_PROPERTY", "BEGIN_YEARMONTH", "CZ_NAME"], axis=1)
full_log2 = full_log.drop(["BEGIN_DAY", "BEGIN_TIME", "YEAR", "MONTH_NAME", "WFO", "INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT", "SOURCE",], axis=1)
full_log2["county"] = full_log2["county"].str.title()

model = joblib.load("model.pkl")
numeric_scaler = joblib.load("numeric_scaler.pkl")
categorical_encoder = joblib.load("categorical_encoder.pkl")
numeric_features = joblib.load("numeric_features.pkl")
categorical_features = joblib.load("categorical_features.pkl")


st.title("Illinois Storm Damage Probability Predictor")

#county name dropdown
county_names = counties['county'].tolist()  # or whatever your column name is
county = st.selectbox("Select County", options=county_names)

#storm event type dropdown
storm_idx = categorical_features.index("EVENT_TYPE")
storm_categories = categorical_encoder.categories_[storm_idx]

storm_type = st.selectbox("Select Storm Event Type:", storm_categories)

st.write("---")

if st.button("Predict Damage Probability"):
    
    #build df of user inputs
    user_data = {}

    #add county numeric variables
    county_row = counties[counties['county'] == county].iloc[0]
    for col in numeric_features:
        if col in counties.columns:
            user_data[col] = county_row[col]

    #add cat variables
    user_data["county"] = county
    user_data["EVENT_TYPE"] = storm_type

    #convert to df
    df_user = pd.DataFrame([user_data])

    #preprocessing
    df_num = df_user[numeric_features]
    df_cat = df_user[categorical_features]

    df_cat = df_cat.astype(str)  

    X_num_scaled = numeric_scaler.transform(df_num)
    X_cat_encoded = categorical_encoder.transform(df_cat).toarray()

    X_preprocessed = np.hstack([X_num_scaled, X_cat_encoded])

    #predict w logistic regression model
    prob = model.predict_proba(X_preprocessed)[0][1]

    st.metric(
        label="Predicted Probability of Property Damage",
        value=f"{prob:.2%}"
    )

# storm proportion

county_events = full_log2[full_log2['county'] == county]  
total_events_county = len(county_events)
storm_events_count = len(county_events[county_events['EVENT_TYPE'] == storm_type])
if total_events_county > 0:
    storm_prop = storm_events_count / total_events_county
else:
    storm_prop = 0

st.metric(
    label=f"Proportion of {storm_type} events in {county}",
    value=f"{storm_prop:.2%}"
)
st.caption(f"({storm_events_count} {storm_type} events out of {total_events_county} total events in {county})")

#damage proportion
storm_damage_count = len(county_events[(county_events['EVENT_TYPE'] == storm_type) & (county_events["DAMAGE_OCCURRED"] == 1)])
if storm_damage_count > 0:
    damage_prop = storm_damage_count / storm_events_count
else:
    damage_prop = 0


if storm_events_count == 0:
    st.metric(
        label="",
        value=f"No recorded {storm_type} events in {county}"
    )
else:
    st.metric(
        label=f"Proportion of {storm_type} events in {county} that caused property damage",
        value=f"{damage_prop:.2%}"
    )
    st.caption(f"({storm_damage_count} out of {storm_events_count} {storm_type} events caused damage in {county})")
    

    



