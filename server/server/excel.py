import pandas as pd

data = {
    "Disease": [
        "Malaria", "Allergy", "Hypothyroidism", "Psoriasis", "GERD",
        "Chronic cholestasis", "Hepatitis A", "Osteoarthristis", 
        "(Vertigo) Paroxysmal Positional Vertigo", "Hypoglycemia", 
        "Acne", "Diabetes", "Impetigo", "Hypertension", 
        "Dimorphic Hemorrhoids (Piles)", "Common Cold", "Chickenpox", 
        "Cervical Spondylosis", "Hyperthyroidism", "Urinary Tract Infection", 
        "Varicose Veins", "AIDS", "Paralysis (Brain Hemorrhage)", 
        "Typhoid", "Hepatitis B", "Fungal Infection", "Hepatitis C", 
        "Migraine", "Bronchial Asthma", "Alcoholic Hepatitis", 
        "Jaundice", "Hepatitis E", "Dengue", "Hepatitis D", 
        "Heart Attack", "Pneumonia", "Arthritis", "Gastroenteritis", 
        "Tuberculosis"
    ],
    "Suitable Medication": [
        "Chloroquine, Artemisinin-based drugs", 
        "Antihistamines, Corticosteroids", 
        "Levothyroxine", 
        "Topical corticosteroids, Methotrexate", 
        "Proton pump inhibitors (e.g., Omeprazole)", 
        "Ursodeoxycholic acid", 
        "Rest, supportive care", 
        "Acetaminophen, NSAIDs", 
        "Epley maneuver", 
        "Glucose tablets, Sugary foods", 
        "Topical or oral antibiotics, Retinoids", 
        "Insulin, Oral hypoglycemic agents", 
        "Antibiotics (Topical or oral)", 
        "ACE inhibitors, Beta-blockers", 
        "High-fiber diet, Pain relievers", 
        "Decongestants, Analgesics", 
        "Antiviral medications, Analgesics", 
        "Physical therapy, Analgesics", 
        "Antithyroid medications", 
        "Antibiotics", 
        "Compression stockings, Lifestyle changes", 
        "Antiretroviral therapy (ART)", 
        "Rehabilitation, Medications as needed", 
        "Antibiotics", 
        "Antiviral medications", 
        "Antifungal medications", 
        "Antiviral medications", 
        "Analgesics, Triptans", 
        "Bronchodilators, Inhaled corticosteroids", 
        "Abstinence, Nutritional support", 
        "Supportive care, Treatment of underlying cause", 
        "Supportive care, Rest", 
        "Supportive care, Fluid replacement", 
        "Antiviral medications", 
        "Emergency medical attention, Medications as needed", 
        "Antibiotics, Oxygen therapy", 
        "NSAIDs, Disease-modifying antirheumatic drugs", 
        "Rehydration, Antiemetics", 
        "Antibiotics (e.g., Rifampicin, Isoniazid)"
    ]
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('disease_medication.csv', index=False)
