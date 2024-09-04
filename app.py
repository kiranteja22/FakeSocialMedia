from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('mrfc.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Extract data from form
    profile_pic = int(data['profile_pic'])
    num_by_num = float(data['num_by_num'])
    full_name = int(data['full_name'])
    num_by_char = float(data['num_by_char'])
    name_username = int(data['name_username'])
    bio_len = int(data['bio_len'])
    url = int(data['url'])
    private = int(data['private'])
    post = int(data['post'])
    followers = int(data['followers'])
    follows = int(data['follows'])

    # Create numpy array
    user_input = np.array([[
        profile_pic,
        num_by_num,
        full_name,
        num_by_char,
        name_username,
        bio_len,
        url,
        private,
        post,
        followers,
        follows
    ]])
    
    # Scale input data
    user_input_scaled = scaler.transform(user_input)
    
    # Make prediction
    prediction = model.predict(user_input_scaled)
    prob = model.predict_proba(user_input_scaled)
    
    result = 'Genuine account' if prediction == 0 else 'Spam account'
    prob_percentage = prob[:, prediction] * 100
    percentage_value = prob_percentage.item()
    
    return jsonify({
        'result': result,
        'probability': f'{percentage_value:.2f}%'
    })

if __name__ == '__main__':
    app.run(debug=True)
