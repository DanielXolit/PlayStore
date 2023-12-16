
from flask import Flask
from flask import request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/predice', methods=['POST'])
def predict():
    json_ = request.json

    query_df = pd.DataFrame(json_, index=[0])

    regressor = joblib.load('regressor.pkl')
    prediction = regressor.predict(query_df)
    respuesta = prediction[0]

    return str(respuesta)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
