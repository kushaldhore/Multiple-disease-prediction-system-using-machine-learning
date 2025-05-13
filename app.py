# # # import pickle
# # # from flask import Flask, request, jsonify, render_template

# # # app = Flask(__name__)

# # # # Load the ML model
# # # with open('model/Multiple_Disease_prediction.pkl', 'rb') as model_file:
# # #     model = pickle.load(model_file)

# # # @app.route('/')
# # # def index():
# # #     return render_template('predict4.html')

# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     # Parse the JSON data from the frontend
# # #     data = request.json
# # #     print(f"Received data: {data}")  # Debug log
# # #     symptoms = data.get('symptoms', [])
# # #     bmi = data.get('bmi', 0)

# # #      # Validate data
# # #     if not symptoms or not isinstance(symptoms, list):
# # #         return jsonify({'error': 'Symptoms not provided or invalid'}), 400
# # #     if not bmi or not isinstance(bmi, (int, float)):
# # #         return jsonify({'error': 'BMI not provided or invalid'}), 400
# # #     # Preprocess inputs for the model
# # #     # (Ensure your model's input structure matches this part)
# # #     features = [1 if symptom in symptoms else 0 for symptom in model.symptoms] + [bmi]

# # #     # Make a prediction
# # #     prediction = model.predict([features])
# # #     return jsonify({'prediction': prediction[0]})

# # # if __name__ == '__main__':
# # #     app.run(debug=True)


# # import pickle
# # from flask import Flask, request, jsonify, render_template
# # from flask_cors import CORS

# # app = Flask(__name__)
# # CORS(app)
# # # Load the ML model
# # with open('model/Multiple_Disease_prediction.pkl', 'rb') as model_file:
# #     model = pickle.load(model_file)

# # # Define the list of symptoms based on your model's requirements
# # MODEL_SYMPTOMS = [
# #     "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", 
# #     "shivering", "chills", "joint_pain", "stomach_pain", "acidity", 
# #     "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
# #     "spotting_urination", "fatigue", "weight_gain", "anxiety"
# #     # Add all symptoms your model expects
# # ]

# # # @app.route('/')
# # # def index():
# # #     return render_template('predict4.html', symptoms=MODEL_SYMPTOMS)

# # @app.route('/predict', methods=['POST'])
# # def predict():
# # #     try: # Parse the JSON data from the frontend
# # # data = request.json
# # #     print(f"Received data: {data}")  # Debug log
# # #     symptoms = data.get('symptoms', [])
# # #     bmi = data.get('bmi', 0)
# #      try:
# #         data = request.json  # Get JSON data from the front-end
# #         print(f"Received data: {data}")  # Debug log

# #         symptoms = data.get('symptoms', [])  # Get symptoms list from the request
# #         bmi = data.get('bmi', 0)  # Get BMI value from the request
# #     # Validate data
# #     if not symptoms or not isinstance(symptoms, list):
# #         return jsonify({'error': 'Symptoms not provided or invalid'}), 400
# #     if not bmi or not isinstance(bmi, (int, float)):
# #         return jsonify({'error': 'BMI not provided or invalid'}), 400

# #     # Prepare input for the model
# #     features = [1 if symptom in symptoms else 0 for symptom in MODEL_SYMPTOMS]
# #     features.append(float(bmi))

# #     # Make a prediction
# #     try:
# #         prediction = model.predict([features])[0]
# #         return jsonify({'prediction': prediction})
# #     except Exception as e:
# #         print(f"Prediction error: {e}")
# #         return jsonify({'error': 'Model prediction failed'}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True)

# # import pickle
# # from flask import Flask, request, jsonify, render_template
# # from flask_cors import CORS  # Importing CORS for cross-origin requests

# # app = Flask(__name__)
# # CORS(app)  # Allow cross-origin requests (for front-end on a different port)

# # # Load the ML model
# # with open('model/Multiple_Disease_prediction.pkl', 'rb') as model_file:
# #     model = pickle.load(model_file)

# # # Define the list of symptoms expected by the model
# # MODEL_SYMPTOMS = [
# #     "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", 
# #     "shivering", "chills", "joint_pain", "stomach_pain", "acidity", 
# #     "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
# #     "spotting_urination", "fatigue", "weight_gain", "anxiety"
# #     # Add all symptoms your model expects
# # ]

# # @app.route('/')
# # def index():
# #     # Pass the symptoms list to the HTML template
# #     return render_template('predict4.html', symptoms=MODEL_SYMPTOMS)

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         data = request.json  # Get JSON data from the front-end
# #         print(f"Received data: {data}")  # Debug log

# #         symptoms = data.get('symptoms', [])  # Get symptoms list from the request
# #         bmi = data.get('bmi', 0)  # Get BMI value from the request

# #         # Validate the input data
# #         if not symptoms or not isinstance(symptoms, list):
# #             return jsonify({'error': 'Symptoms not provided or invalid'}), 400
# #         if not bmi or not isinstance(bmi, (int, float)):
# #             return jsonify({'error': 'BMI not provided or invalid'}), 400

# #         # Prepare the input features for prediction
# #         features = [1 if symptom in symptoms else 0 for symptom in MODEL_SYMPTOMS]
# #         features.append(float(bmi))  # Append BMI value to features

# #         # Make the prediction using the trained model
# #         prediction = model.predict([features])[0]  # Get the prediction result

# #         return jsonify({'prediction': prediction})  # Return the prediction as a JSON response

# #     except Exception as e:
# #         print(f"Error during prediction: {e}")
# #         return jsonify({'error': 'Model prediction failed'}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True)  # Start the Flask app in debug mode






# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS  # Handle CORS for cross-origin requests
# import pickle
# import numpy as np

# # Initialize the Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the trained model
# try:
#     with open("model/Multiple_Disease_prediction.pkl", "rb") as f:
#         model = pickle.load(f)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None

# @app.route('/')
# def home():
#     # Render the HTML form
#     return render_template('predict4.html')

# @app.route('/predict', methods=['POST'])
# def predict():
    
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No data provided or invalid JSON format'}), 400

#         print("Received data:", data)
#         # print("Request headers:", request.headers)
#         # print("Request data (raw):", request.data)
#         # print("Request JSON (parsed):", request.json)
#   # Debug log
#         # Get JSON data from the frontend
#         age = data['age']
#         sex = data['sex']
#         bmi = data['bmi']
#         symptoms = data['symptoms']
#         # data = request.json
#         # bmi = float(data['bmi'])
#         # symptoms = data['symptoms'].split(', ')  # Convert symptom string to a list
#         if age is None or sex is None or bmi is None or symptoms is None:
#             return jsonify({'error': 'Missing one or more required fields: age, sex, bmi, symptoms'}), 400

#         # Convert and process inputs
#         age = float(age)
#         bmi = float(bmi)

#         # Preprocess features to match training input
#         # Example: Encode sex as one-hot (2 features: Male/Female)
#         # Encode sex
#         sex = data['sex']
#         if sex == 'Male':
#             sex_encoded = [1, 0]
#         elif sex == 'Female':
#             sex_encoded = [0, 1]
#         else:
#             sex_encoded = [0, 0]  # Handle 'Other' or unknown cases
#         # Define symptom mapping (adjust this to match your model's expected inputs)
#         all_symptoms = {
#             "itching": 1, "skin_rash": 2, "nodal_skin_eruptions": 3,
#             "continuous_sneezing": 4, "shivering": 5, "chills": 6,
#             "joint_pain": 7, "stomach_pain": 8, "acidity": 9,
#             "ulcers_on_tongue": 10, "muscle_wasting": 11, "vomiting": 12,
#             # Add all other symptoms here in the same format
#         }
#         symptoms_encoded = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
#         # # Generate features based on symptoms
#         # symptom_features = [symptom_map.get(symptom, 0) for symptom in symptoms]
        

#         # Example: Combine features into a single vector
#         # Adjust this based on the model's training configuration
       
# #         # Create the input array for the model
# #         input_features = np.array([bmi] + symptom_features).reshape(1, -1)

# #         # Ensure model is loaded
# #         if model is None:
# #             return jsonify({'error': 'Model not loaded'}), 500

# #         # Predict using the model
# #         prediction = model.predict(input_features)

# #         # Return the prediction result
# #         return jsonify({'prediction': str(prediction[0])})
# #     except Exception as e:
# #         print(f"Error in prediction: {e}")
# #         return jsonify({'error': str(e)}), 400

# # if __name__ == '__main__':
# #     app.run(debug=True)

# # Combine features into a single vector
#         # # Adjust this order and logic based on your training data
#         # features = [age, bmi] + sex_encoded + symptoms_encoded
#          # Combine all features
#         features = [age, bmi] + sex_encoded + symptoms_encoded
#          # Adjust feature vector length to 132
#         if len(features) < 132:
#             features += [0] * (132 - len(features))  # Pad with zeros
#         elif len(features) > 132:
#             features = features[:132]  # Truncate to 132 features

#         features = np.array(features, dtype=np.float32).reshape(1, -1)
#         print("Final feature vector (length):", len(features[0]))
#         # print("Age:", age)
#         # print("BMI:", bmi)
#         # print("Sex encoded:", sex_encoded)
#         # print("Symptoms encoded:", symptoms_encoded)
#         # print("Final feature vector (length):", len(features), features)
#         # Ensure feature vector has the correct length
#         # if len(features) != 132:
#         #     raise ValueError(f"Expected 132 features, but got {len(features)}")
#         # if len(features) < 132:
#         #     features += [0] * (132 - len(features))
#         # elif len(features) > 132:
#         #     features = features[:132]
#         # Convert to NumPy array and reshape for the model
#         # features = np.array(features).reshape(1, -1)

#         # Make prediction
#         # prediction = model.predict(features)
#         # # Map numerical prediction to class names
#         # class_labels = {
#         #     1: "Fungal infection",
#         #     2: "Allergy",
#         #     3: "GERD",
#         #     4: "Chronic cholestasis",
#         #     5: "Drug Reaction",
#         #     6: "Peptic ulcer disease",
#         #     7: "AIDS",
#         #     8: "Diabetes",
#         #     9: "Gastroenteritis",
#         #     10: "Bronchial Asthama",
#         #     11: "Hypertension",
#         #     12: "Migrane",
#         #     13: "Cervical spondylosis",
#         #     14: "Paralysis(brain hemorrhage)",
#         #     15: "Jaundice",
#         #     16: "Malaria",
#         #     17: "Chiken pox",
#         #     18: "Dengue",
#         #     19: "Typhoid",
#         #     20: "hepatitis A",
#         #     21: "hepatitis B",
#         #     22: "Hepatitis C",
#         #     23: "Hepatitis D",
#         #     24: "hepatitis E",
#         #     25: "Alcoholic hepatitis",
#         #     26: "Tuberculosis",
#         #     27: "Common cold",
#         #     28: "Pneumonia",
#         #     29: "Piles",
#         #     30: "Heart Attack",
#         #     31: "Varicose veins",
#         #     32: "Hypothyroidism",
#         #     33: "Hyperthyroidism",
#         #     34: "Hypoglycemia",
#         #     35: "Osteoarthristis",
#         #     36: "Arthritis",
#         #     37: "(vertigo) Paroymsal  Positional Vertigo",
#         #     38: "Acne",
#         #     39: "Urinary tract infecction",
#         #     40: "Psoriasis",
#         #     41: "Impetigo",

#         # }
#         # # Map prediction result to a human-readable label
#         # condition_map = {0: "No Disease", 1: "Disease Detected"}
#         # if model is None:
#         #     return jsonify({'error': 'Model not loaded'}), 500

#         # Make prediction
#         prediction = model.predict(features)[0]

#         # Convert the prediction to a Python type
#         prediction = int(prediction)


#         # prediction_label = condition_map.get(prediction[0], "Unknown Condition")
#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         return jsonify({'error': str(e)}),400
    
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Handle CORS for cross-origin requests

# Load the trained model
try:
    with open("model/Multiple_Disease_prediction.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model is not loaded'}), 500

        # Parse incoming data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided or invalid JSON format'}), 400

        age = float(data['age'])
        bmi = float(data['bmi'])
        sex = data['sex']
        symptoms = data['symptoms']

        # Encode sex
        if sex == 'Male':
            sex_encoded = [1, 0]
        elif sex == 'Female':
            sex_encoded = [0, 1]
        else:
            sex_encoded = [0, 0]  # Default for 'Other'

        # Encode symptoms
        all_symptoms = [
            "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
            "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
            # Add all symptoms expected by your model
        ]
        symptoms_encoded = [1 if symptom in symptoms else 0 for symptom in all_symptoms]

        # Combine all features
        features = [age, bmi] + sex_encoded + symptoms_encoded

        # Adjust feature vector length to 132
        if len(features) < 132:
            features += [0] * (132 - len(features))  # Pad with zeros
        elif len(features) > 132:
            features = features[:132]  # Truncate to 132 features

        # Convert to numpy array
        features = np.array(features, dtype=np.float32).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(features)[0]

        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
