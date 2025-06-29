def get_health_tips(disease, level):
    tips = {
        "heart": {
            "Low": "Keep up healthy habits and regular checkups.",
            "High": "Consult a cardiologist and manage cholesterol/stress."
        },
        "hypertension": {
            "Low": "Maintain a low-sodium diet and regular activity.",
            "High": "Monitor BP, reduce salt, and follow medical advice."
        },
        "diabetes": {
            "Low": "Maintain your sugar levels and healthy weight.",
            "Medium": "Cut back on sugars, walk daily.",
            "High": "Seek medical guidance and manage diet strictly."
        }
    }
    return tips[disease].get(level, "No advice available.")
