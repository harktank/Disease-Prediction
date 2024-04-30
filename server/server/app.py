# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np

# app = Flask(__name__)

# # Load your trained model
# model = joblib.load('trained_bot.joblib')

# # Load your dataset
# df = pd.read_csv('Data/dataset.csv')  # Update with your actual dataset file

# # Extract all symptoms from the dataset
# all_symptoms = set(df.columns[1:])  # Assuming the symptoms start from the second column

# # Create symptoms_dict
# symptoms_dict = {symptom: index for index, symptom in enumerate(all_symptoms)}


# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     user_input = request.args.get("symptom", default="")
#     symptoms = user_input['symptoms']

#     # Prepare response
#     result = {
#         'prediction'
#     }

#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(port=5000)





# def Search():
#     user_query = request.args.get("query", default="")

#     data1 = pd.read_csv('temp_1.csv')
#     data2 = pd.read_csv('temp_2.csv')

#     result_df = search_food_items(user_query, data1, data2)

#     try:
#         result_list = result_df.to_dict(orient='records')

#         return jsonify(result_list), 200, {"Content-Type": "application/json"}
#     except Exception as e:
        
#         return jsonify({"error": str(e)}), 500  # 500 indicates internal server error



# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
clf = joblib.load('trained_bot.joblib')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Retrieve symptoms from the request
        user_input = request.args.get("symptoms", default="fever")
        symptoms = user_input.split(',')  # Assuming symptoms are provided as a comma-separated string

        # Check if all symptoms are valid
        invalid_symptoms = [symptom for symptom in symptoms if symptom not in clf.classes_]
        if invalid_symptoms:
            return jsonify({'error': f'Invalid symptoms: {", ".join(invalid_symptoms)}'}), 400

        # Prepare input vector based on the trained model features
        input_vector = [1 if symptom in symptoms else 0 for symptom in clf.classes_]

        # Use the trained model to predict the disease
        predicted_disease = clf.predict([input_vector])[0]

        # You can include additional logic for confidence levels or other information if needed

        # Prepare response
        result = {
            'prediction': predicted_disease
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
