from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html",
                           area="", bedrooms="", age="",
                           distance="", schools="", location="")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    area = float(data['area'])
    bedrooms = float(data['bedrooms'])
    age = float(data['age'])
    distance = float(data['distance'])
    schools = float(data['schools'])
    location = data['location']

    # ONE-HOT ENCODING
    loc_urban = 1 if location == "Urban" else 0
    loc_rural = 1 if location == "Rural" else 0
    loc_semi = 1 if location == "Semi-Urban" else 0

    features = np.array([[area, bedrooms, age, distance, schools,
                          loc_rural, loc_semi, loc_urban]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    prediction = max(prediction, 100000)

    return render_template("index.html",
                           prediction_text=f"₹{int(prediction)}",
                           area=area,
                           bedrooms=bedrooms,
                           age=age,
                           distance=distance,
                           schools=schools,
                           location=location)

if __name__ == "__main__":
    app.run(debug=True)