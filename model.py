import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# LOAD DATA
df = pd.read_csv("house_data.csv")

# ONE HOT ENCODING
df = pd.get_dummies(df, columns=["Location"])

# FEATURES
X = df.drop("Price", axis=1)
y = df["Price"]

# SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# MODELS
lr = LinearRegression()
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# COMPARISON
print("LR MAE:", mean_absolute_error(y_test, lr.predict(X_test)))
print("DT MAE:", mean_absolute_error(y_test, dt.predict(X_test)))
print("RF MAE:", mean_absolute_error(y_test, rf.predict(X_test)))

# CROSS REGION
urban = df[df["Location_Urban"] == 1]
rural = df[df["Location_Rural"] == 1]

if len(urban) > 1 and len(rural) > 1:
    X_train_cr = urban.drop("Price", axis=1)
    y_train_cr = urban["Price"]

    X_test_cr = rural.drop("Price", axis=1)
    y_test_cr = rural["Price"]

    scaler_cr = StandardScaler()
    X_train_cr = scaler_cr.fit_transform(X_train_cr)
    X_test_cr = scaler_cr.transform(X_test_cr)

    rf_cr = RandomForestRegressor()
    rf_cr.fit(X_train_cr, y_train_cr)

    pred_cr = rf_cr.predict(X_test_cr)
    print("Cross Region MAE:", mean_absolute_error(y_test_cr, pred_cr))

# SAVE MODEL
pickle.dump(rf, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model Ready")