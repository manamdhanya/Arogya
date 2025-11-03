from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
gender_encoder = pickle.load(open("gender_encoder.pkl", "rb"))
health_encoder = pickle.load(open("health_encoder.pkl", "rb"))

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return round(weight / (height_m ** 2), 2)

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/bmi")
def bmi_page():
    return render_template("bmi.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/collaboration")
def collaboration_page():
    return render_template("Collaboration.html")

@app.route("/period-tracker")
def period_tracker_page():
    return render_template("period-tracker.html")

@app.route("/health-assessment")
def health_assessment_page():
    return render_template("health-assessment.html")

@app.route("/blog")
def blog_page():
    return render_template("blog.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age = int(data["age"])
        gender = data["gender"]
        height = float(data["height"])
        weight = float(data["weight"])
        workout = float(data["workout"])
        calories = float(data["calories"])
        sleep = float(data["sleep"])
        water = float(data["water"])
        fastfood = float(data["fastfood"])
        health = data["health"]

        gender_encoded = gender_encoder.transform([gender])[0]
        health_encoded = health_encoder.transform([health])[0]

        X = np.array([[age, height, weight, workout, calories, sleep, water, fastfood, gender_encoded, health_encoded]])

        X_scaled = scaler.transform(X[:, :8])
        X_final = np.concatenate([X_scaled, X[:, 8:]], axis=1)


        cluster = int(model.predict(X_final)[0])

        bmi = calculate_bmi(weight, height)

        if bmi < 18.5: status = "Underweight"
        elif bmi < 25: status = "Healthy"
        elif bmi < 30: status = "Overweight"
        else: status = "Obese"

        lifestyle_result = ["Healthy", "Moderate Risk", "At-Risk"][cluster]

        return jsonify({
            "bmi": bmi,
            "status": status,
            "lifestyle": lifestyle_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)
