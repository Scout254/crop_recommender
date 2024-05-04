from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

# Load the dataset from a CSV file
dataset = pd.read_csv('data/Crop_recommendation.csv')
X = dataset[['temperature', 'humidity', 'rainfall']]
y = dataset['label']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_classifier, "models/plant_prediction_model.joblib")

# Load the trained model
rf_classifier = joblib.load("models/plant_prediction_model.joblib")

# Evaluate the model's accuracy
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = request.json
    location = data['location']
    temperature, humidity, rainfall = get_user_input(location)
    crop_prediction_content_based = crop_prediction(temperature, humidity, rainfall)
    return jsonify({'crop_prediction': crop_prediction_content_based})

def get_user_input(location):
    api_key = '0e7244c57ab946d3b9672602242104'
    url = f'http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt=2024-01-04'
    response = requests.get(url)
    weather_data = response.json()
    temperature = weather_data['forecast']['forecastday'][0]['day']['avgtemp_c']
    humidity = weather_data['forecast']['forecastday'][0]['day']['avghumidity']
    rainfall = weather_data['forecast']['forecastday'][0]['day']['totalprecip_mm'] * 100 / 2
    return temperature, humidity, rainfall

def crop_prediction(temperature, humidity, rainfall):
    new_data = pd.DataFrame([[temperature, humidity, rainfall]], columns=['temperature', 'humidity', 'rainfall'])
    crop_prediction_content_based = rf_classifier.predict(new_data)[0]
    return crop_prediction_content_based

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
