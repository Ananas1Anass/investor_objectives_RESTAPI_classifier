from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
best_classifier = joblib.load('best_model.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = best_classifier.predict(final_features)

        output = int(prediction[0])
        output_text = 'More than 5 years' if output == 1 else 'Less than 5 years'
        print(output_text)

        return render_template('index.html', prediction_text=output_text)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
