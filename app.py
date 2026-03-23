from flask import Flask, send_file, jsonify, request
from flask_cors import CORS
import pickle
import os

app = Flask(__name__, static_folder='.', static_url_path='/')
CORS(app)

# Check if model exists, train if it doesn't
if not os.path.exists("model.pkl"):
    print("Model not found. Training now...")
    import train_model

# Load trained model from file
model = pickle.load(open('model.pkl', 'rb'))

# Home page - direct to app
@app.route('/')
def home():
    return send_file('index.html')

# Main app page
@app.route('/app')
def app_page():
    return send_file('index.html')


# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}

    cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_map = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
    clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

    try:
        X = [[
            float(data['carat']),
            cut_map[data['cut']],
            color_map[data['color']],
            clarity_map[data['clarity']],
            float(data['depth']),
            float(data['table']),
        ]]

        prediction = model.predict(X)
        price = max(0, prediction[0])
        return jsonify({'price': float(price)})

    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)