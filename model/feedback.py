import os
import csv
from datetime import datetime

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), 'user_feedback.csv')

# Ensure feedback file exists
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'image_name', 'soil_type', 'confidence', 'moisture', 'salinity', 'om_index', 'ph_tendency', 'soil_health', 'expert_explanation', 'user_comment', 'user_correction'
        ])

def save_feedback(image_name, soil_type, confidence, moisture, salinity, om_index, ph_tendency, soil_health, expert_explanation, user_comment, user_correction):
    with open(FEEDBACK_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(), image_name, soil_type, confidence, moisture, salinity, om_index, ph_tendency, soil_health, expert_explanation, user_comment, user_correction
        ])
