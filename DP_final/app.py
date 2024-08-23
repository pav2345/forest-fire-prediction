from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load all models
models = {}
model_files = {
    'random_forest': 'fire_model_rf.pkl',
    'decision_tree': 'fire_model_dt.pkl',
    'logistic_regression': 'fire_model_lr.pkl',
    'ada_booster': 'fire_model_ab.pkl',
    'gradient_booster': 'fire_model_gb.pkl',
    'multi_layer_percepton': 'fire_model_mlp.pkl',
}

for name, file in model_files.items():
    if os.path.exists(file):
        models[name] = joblib.load(file)
    else:
        print(f"Model file {file} not found.")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        print(f"Selected algorithm: {algorithm}")  # Debugging line
        model = models.get(algorithm)

        if not model:
            return "Model not found", 404

        # Get input values
        try:
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            DC = float(request.form['DC'])
            ISI = float(request.form['ISI'])
            temp = float(request.form['temp'])
            RH = float(request.form['RH'])
            wind = float(request.form['wind'])
            rain = float(request.form['rain'])

            # Prepare data for prediction
            input_features = np.array([[FFMC, DMC, DC, ISI, temp, RH, wind, rain]])

            # Make prediction
            prediction = model.predict(input_features)[0]
        except Exception as e:
            return f"Error making prediction: {e}", 500

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
