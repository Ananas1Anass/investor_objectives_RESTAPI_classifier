from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the best trained model
best_classifier = joblib.load('best_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = data['Stats']
    prediction = best_classifier.predict([input_data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True,port=5001)
