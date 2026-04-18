from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model pipeline if it exists
MODEL_PATH = 'model_pipeline.pkl'
FEATURE_NAMES_PATH = 'feature_names.pkl'

try:
    model_pipeline = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model not found or failed to load: {e}")
    model_pipeline = None
    feature_names = None

@app.route('/')
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login_submit', methods=['POST'])
def login_submit():
    # In a real app we'd check credentials. Here we just proceed.
    return redirect(url_for('welcome'))

@app.route('/welcome')
def welcome():
    facts = [
        "Your heart beats about 100,000 times in one day and about 35 million times in a year.",
        "A woman's heart typically beats slightly faster than a man's heart.",
        "Laughter is good for your heart. It reduces stress and gives a boost to your immune system.",
        "Your heart pumps about 2,000 gallons of blood every day."
    ]
    return render_template('welcome.html', facts=facts)

@app.route('/predict_input')
def predict_input():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return "Model is not trained yet. Please run train_model.py.", 500
        
    try:
        # Extract features from form
        # The 13 features are: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        input_data = {
            'age': float(request.form['age']),
            'sex': float(request.form['sex']),
            'cp': float(request.form['cp']),
            'trestbps': float(request.form['trestbps']),
            'chol': float(request.form['chol']),
            'fbs': float(request.form['fbs']),
            'restecg': float(request.form['restecg']),
            'thalach': float(request.form['thalach']),
            'exang': float(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': float(request.form['slope']),
            'ca': float(request.form['ca']),
            'thal': float(request.form['thal'])
        }
        
        # Convert to DataFrame to match the format used during training
        input_df = pd.DataFrame([input_data])
        
        # Ensure column order matches training data
        if feature_names:
            input_df = input_df[feature_names]
            
        # Predict
        prediction = model_pipeline.predict(input_df)[0]
        probability = model_pipeline.predict_proba(input_df)[0][1]
        
        # Define symptoms and curations based on prediction
        if prediction == 1:
            result_text = "High Risk of Heart Disease"
            symptoms = "Common symptoms include chest pain (angina), shortness of breath, pain, numbness, weakness or coldness in your legs or arms, pain in the neck, jaw, throat, upper abdomen or back."
            curations = "Please consult a cardiologist immediately. Recommended actions: Regular cardiovascular screening, adopting a heart-healthy diet (low in saturated fats and cholesterol), managing stress, regular moderate exercise, and potentially medication as prescribed by a doctor."
            alert_class = "danger"
        else:
            result_text = "Low Risk of Heart Disease"
            symptoms = "You are currently not exhibiting primary indicators of heart disease based on the provided metrics."
            curations = "Maintain a healthy lifestyle! Continue with a balanced diet, regular physical activity, avoid smoking, and ensure regular check-ups with your healthcare provider to monitor your heart health."
            alert_class = "success"
            
        # Add friendly feature names for display
        display_data = {
            'Age': input_data['age'],
            'Sex': 'Male' if input_data['sex'] == 1 else 'Female',
            'Chest Pain Type': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][int(input_data['cp'])-1] if 1 <= input_data['cp'] <= 4 else input_data['cp'],
            'Resting Blood Pressure': f"{input_data['trestbps']} mm Hg",
            'Serum Cholesterol': f"{input_data['chol']} mg/dl",
            'Max Heart Rate': f"{input_data['thalach']} bpm"
        }
        
        return render_template('result.html', 
                               result_text=result_text, 
                               probability=round(probability*100, 2),
                               symptoms=symptoms, 
                               curations=curations,
                               alert_class=alert_class,
                               user_details=display_data)
                               
    except Exception as e:
        return f"An error occurred: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
