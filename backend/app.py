from flask import Flask, request, jsonify
import joblib
import json
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "Hi, this is the home page of our real estate price prediction server."


model = joblib.load('bangalore_home_prices_model.pkl')
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        loc_index = data_columns.index(data['location'].lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = data['total_sqft']
    x[1] = data['bath']
    x[2] = data['bhk']
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = model.predict([x])[0]
    return jsonify({
        'predicted_price_lakhs': round(predicted_price, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
