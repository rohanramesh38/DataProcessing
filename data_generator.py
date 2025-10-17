import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
n_rows = 300

# ======================================================
# 🏠 REAL ESTATE DATASET WITH COMPREHENSIVE FEATURES
# ======================================================
# This dataset includes: time series, geo data, text, categorical, 
# numeric, missing values, outliers, skewed features, etc.

# 1. ID and Basic Info
property_ids = [f"PROP_{i:04d}" for i in range(1, n_rows + 1)]

# 2. Time-based features (listing dates, sale dates)
start_date = datetime(2020, 1, 1)
listing_dates = [start_date + timedelta(days=int(np.random.randint(0, 1460))) for _ in range(n_rows)]
days_to_sale = np.random.randint(5, 180, n_rows).astype(int)
sale_dates = [listing_dates[i] + timedelta(days=int(days_to_sale[i])) for i in range(n_rows)]

# 3. Geographic features (US cities)
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
          'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin']
city = np.random.choice(cities, n_rows)

# Realistic lat/long for each city
city_coords = {
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Phoenix': (33.4484, -112.0740),
    'Philadelphia': (39.9526, -75.1652),
    'San Antonio': (29.4241, -98.4936),
    'San Diego': (32.7157, -117.1611),
    'Dallas': (32.7767, -96.7970),
    'Austin': (30.2672, -97.7431)
}

latitudes = []
longitudes = []
for c in city:
    base_lat, base_lon = city_coords[c]
    # Add some random variance
    latitudes.append(base_lat + np.random.normal(0, 0.1))
    longitudes.append(base_lon + np.random.normal(0, 0.1))

# 4. Property features
property_types = ['Single Family', 'Condo', 'Townhouse', 'Multi Family', 'Villa']
property_type = np.random.choice(property_types, n_rows, p=[0.4, 0.25, 0.15, 0.1, 0.1])

bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_rows, p=[0.1, 0.2, 0.35, 0.25, 0.08, 0.02])
bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_rows, p=[0.15, 0.15, 0.3, 0.2, 0.12, 0.05, 0.03])

# Area in square feet (skewed distribution)
area = np.random.gamma(shape=2, scale=500, size=n_rows) + 500
area = np.clip(area, 500, 5000)

# Lot size in acres (skewed, with some missing)
lot_size = np.random.gamma(shape=1.5, scale=0.3, size=n_rows)
lot_size = np.clip(lot_size, 0.05, 3)

# Age of property (years)
age = np.random.gamma(shape=3, scale=10, size=n_rows)
age = np.clip(age, 0, 100)

# 5. Price (target variable - depends on multiple features)
base_price = (
    50000 +  # Base
    area * 150 +  # Price per sqft
    bedrooms * 30000 +  # Bedroom premium
    bathrooms * 20000 +  # Bathroom premium
    lot_size * 50000 +  # Lot size premium
    np.where(property_type == 'Single Family', 50000, 0) +
    np.where(property_type == 'Villa', 100000, 0) +
    np.random.normal(0, 50000, n_rows) -  # Random variation
    age * 1000  # Depreciation
)

# Add city premiums
city_premiums = {
    'New York': 200000, 'Los Angeles': 180000, 'San Francisco': 250000,
    'Chicago': 50000, 'Houston': 30000, 'Phoenix': 40000,
    'Philadelphia': 60000, 'San Antonio': 20000, 'San Diego': 150000,
    'Dallas': 45000, 'Austin': 80000
}

price = []
for i in range(n_rows):
    p = base_price[i] + city_premiums.get(city[i], 0)
    price.append(max(p, 100000))  # Minimum price

price = np.array(price)

# Add some outliers (luxury properties)
outlier_indices = np.random.choice(n_rows, size=15, replace=False)
price[outlier_indices] = price[outlier_indices] * np.random.uniform(2, 5, 15)

# 6. Categorical features
condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_rows, p=[0.2, 0.5, 0.25, 0.05])
parking = np.random.choice(['Garage', 'Carport', 'Street', 'None'], n_rows, p=[0.5, 0.2, 0.2, 0.1])
has_pool = np.random.choice(['True', 'False'], n_rows, p=[0.3, 0.7])
has_fireplace = np.random.choice(['Yes', 'No'], n_rows, p=[0.4, 0.6])

# 7. Text features (property descriptions)
description_templates = [
    "Beautiful {condition} {property_type} with {bedrooms} bedrooms in {city}. Spacious living area.",
    "Charming {property_type} featuring {bathrooms} bathrooms. Great location in {city}!",
    "Stunning {condition} property with modern amenities. {bedrooms}BR/{bathrooms}BA in prime {city} area.",
    "Lovely {property_type} home, well-maintained and move-in ready. Located in {city}.",
    "Spacious {bedrooms} bedroom {property_type}. Perfect for families! {city} location.",
]

descriptions = []
for i in range(n_rows):
    template = np.random.choice(description_templates)
    desc = template.format(
        condition=condition[i].lower(),
        property_type=property_type[i].lower(),
        bedrooms=bedrooms[i],
        bathrooms=bathrooms[i],
        city=city[i]
    )
    # Add some variation
    if np.random.random() > 0.7:
        desc += " Updated kitchen and appliances."
    if np.random.random() > 0.8:
        desc += " Close to schools and shopping."
    descriptions.append(desc)

# 8. Additional numeric features
year_built = 2024 - age.astype(int)
stories = np.random.choice([1, 2, 3], n_rows, p=[0.4, 0.5, 0.1])
garage_spaces = np.random.choice([0, 1, 2, 3], n_rows, p=[0.2, 0.3, 0.4, 0.1])

# HOA fees (some missing)
hoa_fee = np.random.gamma(shape=2, scale=100, size=n_rows)
hoa_fee = np.where(property_type == 'Condo', hoa_fee + 200, hoa_fee)

# Days on market
days_on_market = np.abs(np.random.normal(45, 30, n_rows))

# Number of views (online listing views)
views = np.random.poisson(lam=100, size=n_rows) + 20

# School rating (1-10)
school_rating = np.random.choice(range(1, 11), n_rows, p=[0.05, 0.05, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05])

# Walk score (0-100)
walk_score = np.random.beta(a=5, b=2, size=n_rows) * 100

# ======================================================
# 9. Introduce Missing Values (realistic patterns)
# ======================================================
# LOT SIZE: Missing for condos (no lot)
lot_size = np.where(property_type == 'Condo', np.nan, lot_size)

# HOA FEE: Missing for some single family homes
missing_hoa_mask = (property_type == 'Single Family') & (np.random.random(n_rows) > 0.6)
hoa_fee = np.where(missing_hoa_mask, np.nan, hoa_fee)

# YEAR BUILT: Random missing (5%)
year_built_mask = np.random.random(n_rows) > 0.95
year_built = np.where(year_built_mask, np.nan, year_built)

# DESCRIPTION: Some missing (3%)
description_mask = np.random.random(n_rows) > 0.97
descriptions = [desc if not description_mask[i] else np.nan for i, desc in enumerate(descriptions)]

# WALK SCORE: Random missing (10%)
walk_score_mask = np.random.random(n_rows) > 0.9
walk_score = np.where(walk_score_mask, np.nan, walk_score)

# ======================================================
# 10. Create DataFrame
# ======================================================
df = pd.DataFrame({
    'property_id': property_ids,
    'listing_date': listing_dates,
    'sale_date': sale_dates,
    'city': city,
    'latitude': latitudes,
    'longitude': longitudes,
    'property_type': property_type,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'area': area.round(0),
    'lot_size': lot_size,
    'year_built': year_built,
    'age': age.round(0),
    'condition': condition,
    'price': price.round(0),
    'parking': parking,
    'has_pool': has_pool,
    'has_fireplace': has_fireplace,
    'stories': stories,
    'garage_spaces': garage_spaces,
    'hoa_fee': hoa_fee.round(2),
    'days_on_market': days_on_market.round(0),
    'views': views,
    'school_rating': school_rating,
    'walk_score': walk_score.round(1),
    'description': descriptions
})

# Add some duplicate rows (2%)
duplicate_indices = np.random.choice(df.index, size=6, replace=False)
df_duplicates = df.loc[duplicate_indices].copy()
df = pd.concat([df, df_duplicates], ignore_index=True)

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ======================================================
# 11. Save to CSV
# ======================================================
df.to_csv('real_estate_data.csv', index=False)

print("✅ Sample dataset created successfully!")
print(f"📊 Shape: {df.shape}")
print(f"\n📋 Dataset Summary:")
print(f"  • Property listings: {len(df)}")
print(f"  • Time range: {df['listing_date'].min()} to {df['listing_date'].max()}")
print(f"  • Cities: {df['city'].nunique()}")
print(f"  • Property types: {df['property_type'].nunique()}")
print(f"  • Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"\n🔍 Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\n🔄 Duplicates: {df.duplicated().sum()}")
print("\n📁 File saved as: real_estate_data.csv")

# Display first few rows
print("\n👀 First 5 rows:")
print(df.head())

# Display data types
print("\n📊 Data Types:")
print(df.dtypes)