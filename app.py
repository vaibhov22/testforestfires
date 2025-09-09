import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model & scaler
model = pickle.load(open("ridge.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

# Replace your old predict function with this
@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[x]) for x in request.form]
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        # Determine severity class
        if prediction < 10:
            severity = "low"
        elif prediction <= 20:
            severity = "medium"
        else:
            severity = "high"

        return render_template("index.html", prediction=round(prediction,2), severity=severity)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}", severity="low")

if __name__ == "__main__":
    app.run(debug=True)
