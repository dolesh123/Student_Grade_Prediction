from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('grade_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        subject = request.form['subject']
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        previous_marks = float(request.form['previous_marks'])
        sleep = float(request.form['sleep'])
        social_media = float(request.form['social_media'])
        stress = float(request.form['stress'])
        
        # Create input dataframe with all features
        input_df = pd.DataFrame({
            'study_hours': [study_hours],
            'attendance': [attendance],
            'previous_marks': [previous_marks],
            'sleep': [sleep],
            'social_media': [social_media],
            'stress': [stress]
        })
        
        # Predict score
        predicted_score = model.predict(input_df)[0]
        
        # Ensure score is within 0-100 range
        predicted_score = max(0, min(100, predicted_score))
        
        return render_template('result.html', 
                             subject=subject,
                             prediction=round(predicted_score, 1))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)