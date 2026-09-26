import flask
from flask import Flask
from flask import request,render_template
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import LinearRegression

with open('scaling.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open("final_model.pkl", "rb") as t:
    reg = pickle.load(t)

app = Flask(__name__)


@app.route('/')
def main_page():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict_fun():

    age = float(request.form["age"])
    sex = float(request.form["sex"])
    cp = float(request.form["cp"])
    thalach = float(request.form["thalach"])
    oldpeak = float(request.form["oldpeak"])
    slope = float(request.form["slope"])
    thal = float(request.form["thal"])

    val = np.array([[
        age,
        sex,
        cp,
        thalach,
        oldpeak,
        slope,
        thal
    ]])

    val1 = scaler.transform(val)

    sol = reg.predict(val1)[0]

    if sol == 1:
        prediction_text = "You have heart disease"
    else:
        prediction_text = "You don't have heart disease"

    return render_template(
        "index.html",
        prediction_text=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)