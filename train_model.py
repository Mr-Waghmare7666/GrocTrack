import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load CSV
data = pd.read_csv('data/grocery_data.csv')
data['Purchase_Date'] = pd.to_datetime(data['Purchase_Date'])
data['Purchase_Day'] = data['Purchase_Date'].dt.day
data['Purchase_Month'] = data['Purchase_Date'].dt.month

# Encode category
le = LabelEncoder()
data['Category_Code'] = le.fit_transform(data['Category'])

X = data[['Category_Code','Storage_Temp','Storage_Humidity','Purchase_Day','Purchase_Month']]
y = data['Shelf_Life']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

# Save model and label encoder
joblib.dump(model,'model/expiry_model.pkl')
joblib.dump(le,'model/category_encoder.pkl')

print("Model trained successfully!")
