# Arogya – Health Prediction Web Application

## 📌 Description

Arogya is a **Machine Learning based web application** that predicts a person's health condition based on input data.
The application uses a trained ML model and provides predictions through a simple web interface.

## 🚀 Features

* Predict health condition using machine learning
* Web interface for user input
* Pre-trained ML model for predictions
* Encoders and scaler used for preprocessing
* Easy-to-use health prediction system

## 🛠 Technologies Used

* Python
* Flask
* Scikit-learn
* HTML
* CSS

## ⚙️ How It Works

1. The user enters health-related details through the web interface.
2. The input data is processed using:

   * **gender_encoder**
   * **health_encoder**
   * **scaler**
3. The processed data is passed to the trained **machine learning model**.
4. The model predicts the health outcome and displays the result.

## ▶️ Installation

Clone the repository:

```id="cl1vhn"
git clone https://github.com/yourusername/Arogya.git
```

Install dependencies:

```id="q2thqj"
pip install -r requirements.txt
```

Run the application:

```id="4z11i1"
python app.py
```

Open in browser:

```id="pj6gqu"
http://localhost:5000
```
