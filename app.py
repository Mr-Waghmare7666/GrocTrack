from flask import Flask, render_template, request
import joblib, pandas as pd
from datetime import datetime

app = Flask(__name__)

model = joblib.load('model/expiry_model.pkl')
le = joblib.load('model/category_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        item = request.form['item']
        category = request.form['category']
        purchase_date = datetime.strptime(request.form['purchase_date'], "%Y-%m-%d")
        temp = float(request.form['temp'])
        humidity = float(request.form['humidity'])
        category_code = le.transform([category])[0]
        purchase_day = purchase_date.day
        purchase_month = purchase_date.month
        features = [[category_code,temp,humidity,purchase_day,purchase_month]]
        predicted_shelf_life = model.predict(features)[0]
        expiry_date = purchase_date + pd.Timedelta(days=int(predicted_shelf_life))
        return render_template('result.html', item=item, category=category, expiry_date=expiry_date.date(), days=int(predicted_shelf_life))
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
