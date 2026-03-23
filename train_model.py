import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("diamonds.csv")

# Select features
df = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']]

# Convert categorical to numerical
cut_map = {'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4}
color_map = {'D':0, 'E':1, 'F':2, 'G':3, 'H':4, 'I':5, 'J':6}
clarity_map = {'I1':0, 'SI2':1, 'SI1':2, 'VS2':3, 'VS1':4, 'VVS2':5, 'VVS1':6, 'IF':7}

df['cut'] = df['cut'].map(cut_map)
df['color'] = df['color'].map(color_map)
df['clarity'] = df['clarity'].map(clarity_map)

# Features & target
X = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table']]
y = df['price']

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")