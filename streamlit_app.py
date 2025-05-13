# # "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
# #             "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting","burning_micturition", 
# #             "spotting_ urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy","patches_in_throat","irregular_sugar_level", "cough", "high_fever", "sunken_eyes", "breathlessness","sweating","dehydration", 
# #             "indigestion","headache","yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain",
# #           "constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine","yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach","swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements","pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness","cramps","bruising","obesity","swollen_legs","swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties","excessive_hunger","extra_marital_contacts","drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain", "muscle_weakness","stiff_neck","swelling_joints","movement_stiffness", "spinning_movements","loss_of_balance","unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort","foul_smell_of urine","continuous_feel_of_urine","passage_of_gases", "internal_itching","toxic_look_(typhos)", "depression","irritability","muscle_pain","altered_sensorium","red_spots_over_body","belly_pain","abnormal_menstruation","dischromic _patches","watering_from_eyes","increased_appetite","polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration","visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections","coma","stomach_bleeding","distention_of_abdomen","history_of_alcohol_consumption","fluid_overload","lood_in_sputum","prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples","blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister","red_sore_around_nose","yellow_crust_ooze"

# import streamlit as st
# import pickle
# import numpy as np

# # Load the trained model (directly in Streamlit)
# try:
#     with open("model/Multiple_Disease_prediction.pkl", "rb") as f:
#         model = pickle.load(f)
#     st.write("Model loaded successfully.")
# except Exception as e:
#     st.error(f"Error loading model: {e}")
#     model = None

# # Streamlit interface
# st.title("Disease Prediction System")

# # Check if the model was loaded correctly
# if model is None:
#     st.error("Model could not be loaded. Ensure the .pkl file is in the correct directory.")
# else:
#     # Input fields for height and weight
#     st.header("Enter Your Details")
#     height = st.number_input("Height (in cm)", min_value=1, max_value=250, step=1, value=170)
#     weight = st.number_input("Weight (in kg)", min_value=1, max_value=300, step=1, value=65)

#     # Calculate BMI
#     bmi = weight / (height / 100) ** 2
#     st.write(f"Your BMI: {bmi:.2f}")

#     # Input fields for other details
#     sex = st.selectbox("Sex", ["Male", "Female", "Other"])
#     age = st.number_input("Age", min_value=1, max_value=120, step=1, value=25)
#     symptoms = st.multiselect(
#         "Select Symptoms",
#         [
#             "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
#             "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting","burning_micturition", 
#             "spotting_ urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy","patches_in_throat","irregular_sugar_level", "cough", "high_fever", "sunken_eyes", "breathlessness","sweating","dehydration", 
#             "indigestion","headache","yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain",
#           "constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine","yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach","swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements","pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness","cramps","bruising","obesity","swollen_legs","swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties","excessive_hunger","extra_marital_contacts","drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain", "muscle_weakness","stiff_neck","swelling_joints","movement_stiffness", "spinning_movements","loss_of_balance","unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort","foul_smell_of urine","continuous_feel_of_urine","passage_of_gases", "internal_itching","toxic_look_(typhos)", "depression","irritability","muscle_pain","altered_sensorium","red_spots_over_body","belly_pain","abnormal_menstruation","dischromic _patches","watering_from_eyes","increased_appetite","polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration","visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections","coma","stomach_bleeding","distention_of_abdomen","history_of_alcohol_consumption","fluid_overload","lood_in_sputum","prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples","blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister","red_sore_around_nose","yellow_crust_ooze"
#             #Add all symptoms your model supports
#         ]
#     )

#     # Disease mapping (numeric prediction -> disease name)
#     disease_mapping = {
#         1: "Fungal infection",
#         2: "Allergy",
#         3: "GERD",
#         4: "Chronic cholestasis",
#         5: "Drug Reaction",
#         6: "Peptic ulcer disease",
#         7: "AIDS",
#         8: "Diabetes",
#         9: "Gastroenteritis",
#         10: "Bronchial Asthma",
#         11: "Hypertension",  # Mapping of disease index
#         12: "Migrane",
#         13: "Cervical spondylosis",
#         14: "Paralysis (brain hemorrhage)",
#         15: "Jaundice",
#         16: "Malaria",
#         17: "Chicken pox",
#         18: "Dengue",
#         19: "Typhoid",
#         20: "Hepatitis A",
#         21: "Hepatitis B",
#         22: "Hepatitis C",
#         23: "Hepatitis D",
#         24: "Hepatitis E",
#         25: "Alcoholic hepatitis",
#         26: "Tuberculosis",
#         27: "Common cold",
#         28: "Pneumonia",
#         29: "Piles",
#         30: "Heart Attack",
#         31: "Varicose veins",
#         32: "Hypothyroidism",
#         33: "Hyperthyroidism",
#         34: "Hypoglycemia",
#         35: "Osteoarthritis",
#         36: "Arthritis",
#         37: "Vertigo (Positional Vertigo)",
#         38: "Acne",
#         39: "Urinary tract infection",
#         40: "Psoriasis",
#         41: "Impetigo"
#     }

#     # Submit button for prediction
#     if st.button("Predict"):
#         # Ensure all necessary inputs are provided
#         if not symptoms:
#             st.error("Please select at least one symptom.")
#         else:
#             # Prepare data for the prediction request
#             data = {
#                 "age": age,
#                 "bmi": bmi,
#                 "sex": sex,
#                 "symptoms": symptoms
#             }

#             # Process input features
#             try:
#                 # Encode sex
#                 if sex == 'Male':
#                     sex_encoded = [1, 0]
#                 elif sex == 'Female':
#                     sex_encoded = [0, 1]
#                 else:
#                     sex_encoded = [0, 0]  # Default for 'Other'

#                 # Encode symptoms correctly
#                 all_symptoms = [
#                     "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
#             "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting","burning_micturition", 
#             "spotting_ urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy","patches_in_throat","irregular_sugar_level", "cough", "high_fever", "sunken_eyes", "breathlessness","sweating","dehydration", 
#             "indigestion","headache","yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain",
#           "constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine","yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach","swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements","pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness","cramps","bruising","obesity","swollen_legs","swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties","excessive_hunger","extra_marital_contacts","drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain", "muscle_weakness","stiff_neck","swelling_joints","movement_stiffness", "spinning_movements","loss_of_balance","unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort","foul_smell_of urine","continuous_feel_of_urine","passage_of_gases", "internal_itching","toxic_look_(typhos)", "depression","irritability","muscle_pain","altered_sensorium","red_spots_over_body","belly_pain","abnormal_menstruation","dischromic _patches","watering_from_eyes","increased_appetite","polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration","visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections","coma","stomach_bleeding","distention_of_abdomen","history_of_alcohol_consumption","fluid_overload","lood_in_sputum","prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples","blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister","red_sore_around_nose","yellow_crust_ooze"
#                     # Add all symptoms expected by your model
#                 ]
                
#                 # Ensure the symptoms are encoded properly as 1 or 0
#                 symptoms_encoded = [1 if symptom in symptoms else 0 for symptom in all_symptoms]

#                 # Combine all features
#                 features = [age, bmi] + sex_encoded + symptoms_encoded

#                 # Debug: Log the features being passed to the model
#                 st.write(f"Features for prediction: {features}")

#                 # Adjust feature vector length to 132 (match model input requirements)
#                 if len(features) < 132:
#                     features += [0] * (132 - len(features))  # Pad with zeros
#                 elif len(features) > 132:
#                     features = features[:132]  # Truncate to 132 features

#                 # Convert to numpy array
#                 features = np.array(features, dtype=np.float32).reshape(1, -1)

#                 # Make prediction using the model
#                 prediction = model.predict(features)[0]

#                 # Log the raw prediction value
#                 st.write(f"Raw prediction value: {prediction}")

#                 # Map the numeric prediction to a disease name
#                 disease_name = disease_mapping.get(prediction, "Unknown disease")

#                 # Display the prediction
#                 st.success(f"Prediction: {disease_name}")

#             except Exception as e:
#                 st.error(f"Error during prediction: {e}")




# import os
# import pandas as pd
# import numpy as np
# import streamlit as st
# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder

# # Load data for multiple disease prediction
# def load_data():
#     train_data_path = os.path.join(os.path.dirname(__file__), 'Training.csv')
#     test_data_path = os.path.join(os.path.dirname(__file__), 'Testing.csv')
#     train_data = pd.read_csv(train_data_path)
#     test_data = pd.read_csv(test_data_path)
#     return train_data, test_data

# # Train Random Forest model
# def train_model(train_data):
#     features = train_data.drop('prognosis', axis=1)
#     target = train_data['prognosis']
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(features, target)
#     return rf, features.columns.tolist()

# # Predict diseases based on symptoms
# def predict_disease(symptoms, symptom_columns, model):
#     # Convert user symptoms into feature vector
#     input_vector = np.zeros(len(symptom_columns), dtype=int)
#     for symptom in symptoms:
#         if symptom in symptom_columns:
#             idx = symptom_columns.index(symptom)
#             input_vector[idx] = 1

#     # Predict using the trained model
#     prediction = model.predict([input_vector])[0]
#     return prediction

# # Main Streamlit app
# def main():
#     # App title
#     st.title("Disease Prediction System")

#     # Load datasets
#     train_data, test_data = load_data()
#     model, symptom_columns = train_model(train_data)
#     encoder = LabelEncoder()
#     encoder.fit(train_data['prognosis'])

#     # User input: Demographics
#     st.header("Enter Your Details")
#     height = st.number_input("Height (in cm)", min_value=1, max_value=250, step=1, value=170)
#     weight = st.number_input("Weight (in kg)", min_value=1, max_value=300, step=1, value=65)
#     bmi = weight / (height / 100) ** 2
#     st.write(f"Your BMI: {bmi:.2f}")
#     sex = st.selectbox("Sex", ["Male", "Female", "Other"])
#     age = st.number_input("Age", min_value=1, max_value=120, step=1, value=25)

#     # User input: Symptoms
#     st.header("Enter Symptoms")
#     symptoms = st.multiselect(
#         "Select Symptoms",
#         symptom_columns,
#         help="Choose the symptoms you are experiencing."
#     )

#     # Predict button
#     if st.button("Predict Disease"):
#         if not symptoms:
#             st.error("Please select at least one symptom.")
#         else:
#             try:
#                 # Predict disease
#                 predicted_disease = predict_disease(symptoms, symptom_columns, model)
#                 decoded_disease = encoder.inverse_transform([predicted_disease])[0]
#                 st.success(f"Predicted Disease: {decoded_disease}")
#             except Exception as e:
#                 st.error(f"An error occurred during prediction: {e}")

#     # Allow viewing full symptom list
#     if st.checkbox("View All Symptoms"):
#         st.write(symptom_columns)

# if __name__ == "__main__":
#     main()





import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

# Load the trained model (directly in Streamlit)
try:
    with open("model\multiple_disease_prediction0.pkl", "rb") as f:
        model = pickle.load(f)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Function to train Random Forest model
def train_model(train_data):
    features = train_data.drop('prognosis', axis=1)
    target = train_data['prognosis']

    # Fit LabelEncoder on all unique labels in the training data
    encoder = LabelEncoder()
    target_encoded = encoder.fit_transform(target)

    # Train RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, target_encoded)

    return rf, features.columns.tolist(), encoder

# Function to predict diseases
def predict_disease(symptoms, symptom_columns, model, encoder):
    # Convert user symptoms into feature vector
    input_vector = np.zeros(len(symptom_columns), dtype=int)
    for symptom in symptoms:
        if symptom in symptom_columns:
            idx = symptom_columns.index(symptom)
            input_vector[idx] = 1

    # Predict using the trained model
    prediction_encoded = model.predict([input_vector])[0]
    
    # Decode the numeric prediction back to a label
    prediction = encoder.inverse_transform([prediction_encoded])[0]
    return prediction

# Streamlit interface
st.title("Disease Prediction System")

# Load data for multiple disease prediction
train_data_path = os.path.join(os.path.dirname(__file__), 'Training.csv')
test_data_path = os.path.join(os.path.dirname(__file__), 'Testing.csv')
train_data = pd.read_csv(train_data_path)

# Train the model and get encoder
model, symptom_columns, encoder = train_model(train_data)

# Input fields for height and weight
st.header("Enter Your Details")
height = st.number_input("Height (in cm)", min_value=1, max_value=250, step=1, value=170)
weight = st.number_input("Weight (in kg)", min_value=1, max_value=300, step=1, value=65)

# Calculate BMI
bmi = weight / (height / 100) ** 2
st.write(f"Your BMI: {bmi:.2f}")

# BMI Classification Description
if bmi < 18.5:
    st.write("BMI Category: Underweight (Less than 18.5)")
elif 18.5 <= bmi < 24.9:
    st.write("BMI Category: Normal weight (18.5 - 24.9)")
elif 25 <= bmi < 29.9:
    st.write("BMI Category: Overweight (25 - 29.9)")
else:
    st.write("BMI Category: Obesity (30 and above)")

# Input fields for other details
sex = st.selectbox("Sex", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=1, max_value=120, step=1, value=25)

# # User input for symptoms
# symptoms_multiple = st.text_input("Enter Symptoms (comma-separated)")

# # Submit button for prediction
# if st.button("Check Symptoms"):
#     if symptoms_multiple:
#         symptoms_list = symptoms_multiple.split(",")
#         # Remove leading/trailing spaces from symptoms
#         symptoms_list = [symptom.strip() for symptom in symptoms_list]
        
#         # Call the predict_disease function with user input
#         disease = predict_disease(symptoms_list, symptom_columns, model, encoder)
#         st.success(f"Predicted Disease: {disease}")
#     else:
#         st.error("Please enter symptoms.")

# # Disease mapping (numeric prediction -> disease name)
# disease_mapping = {
#     1: "Fungal infection",
#     2: "Allergy",
#     3: "GERD",
#     4: "Chronic cholestasis",
#     5: "Drug Reaction",
#     6: "Peptic ulcer disease",
#     7: "AIDS",
#     8: "Diabetes",
#     9: "Gastroenteritis",
#     10: "Bronchial Asthma",
#     11: "Hypertension",  # Mapping of disease index
#     12: "Migrane",
#     13: "Cervical spondylosis",
#     14: "Paralysis (brain hemorrhage)",
#     15: "Jaundice",
#     16: "Malaria",
#     17: "Chicken pox",
#     18: "Dengue",
#     19: "Typhoid",
#     20: "Hepatitis A",
#     21: "Hepatitis B",
#     22: "Hepatitis C",
#     23: "Hepatitis D",
#     24: "Hepatitis E",
#     25: "Alcoholic hepatitis",
#     26: "Tuberculosis",
#     27: "Common cold",
#     28: "Pneumonia",
#     29: "Piles",
#     30: "Heart Attack",
#     31: "Varicose veins",
#     32: "Hypothyroidism",
#     33: "Hyperthyroidism",
#     34: "Hypoglycemia",
#     35: "Osteoarthritis",
#     36: "Arthritis",
#     37: "Vertigo (Positional Vertigo)",
#     38: "Acne",
#     39: "Urinary tract infection",
#     40: "Psoriasis",
#     41: "Impetigo"
# }



# # Handle the model prediction when the button is clicked
# if model:
#     st.write(f"Prediction using the model is possible now.")
# User input for symptoms with multiselect and search option
st.header("Select Symptoms")
symptoms_multiple = st.multiselect(
    "Select Symptoms (Searchable List)", 
    [
        "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
        "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", 
        "muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue", 
        "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", 
        "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", 
        "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", 
        "headache", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", 
        "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea", 
        "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", 
        "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise", 
        "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes", 
        "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", 
        "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", 
        "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity", 
        "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", 
        "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", 
        "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain", 
        "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", 
        "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", 
        "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine", 
        "passage_of_gases", "internal_itching", "toxic_look_(typhos)", "depression", 
        "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain", 
        "abnormal_menstruation", "dischromic _patches", "watering_from_eyes", "increased_appetite", 
        "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration", 
        "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections", 
        "coma", "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption", 
        "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf", "palpitations", 
        "painful_walking", "pus_filled_pimples", "blackheads", "scarring", "skin_peeling", 
        "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister", 
        "red_sore_around_nose", "yellow_crust_ooze"
        # Add all symptoms your model supports
    ], 
    help="Select multiple symptoms by searching and clicking on them."
)

# Submit button for prediction
if st.button("Check Symptoms"):
    if symptoms_multiple:
        # Call the predict_disease function with user input
        disease = predict_disease(symptoms_multiple, symptom_columns, model, encoder)
        st.success(f"Predicted Disease: {disease}")
    else:
        st.error("Please select symptoms.")

# Disease mapping (numeric prediction -> disease name)
disease_mapping = {
    1: "Fungal infection",
    2: "Allergy",
    3: "GERD",
    4: "Chronic cholestasis",
    5: "Drug Reaction",
    6: "Peptic ulcer disease",
    7: "AIDS",
    8: "Diabetes",
    9: "Gastroenteritis",
    10: "Bronchial Asthma",
    11: "Hypertension",  # Mapping of disease index
    12: "Migrane",
    13: "Cervical spondylosis",
    14: "Paralysis (brain hemorrhage)",
    15: "Jaundice",
    16: "Malaria",
    17: "Chicken pox",
    18: "Dengue",
    19: "Typhoid",
    20: "Hepatitis A",
    21: "Hepatitis B",
    22: "Hepatitis C",
    23: "Hepatitis D",
    24: "Hepatitis E",
    25: "Alcoholic hepatitis",
    26: "Tuberculosis",
    27: "Common cold",
    28: "Pneumonia",
    29: "Piles",
    30: "Heart Attack",
    31: "Varicose veins",
    32: "Hypothyroidism",
    33: "Hyperthyroidism",
    34: "Hypoglycemia",
    35: "Osteoarthritis",
    36: "Arthritis",
    37: "Vertigo (Positional Vertigo)",
    38: "Acne",
    39: "Urinary tract infection",
    40: "Psoriasis",
    41: "Impetigo"
}

# Handle the model prediction when the button is clicked
if model:
    st.write(f"Prediction using the model is possible now.")
