import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,r2_score
import joblib
#Load dataset
df= pd.read_csv('StudentsPerformance.csv')
X =df[['study_hours','attendance','previous_marks','sleep','social_media','stress']]
y = df['score']
print("=" * 60)
print(" STUDENT GRADE PREDICTION MODEL")
print("=" * 60)
print("\n Feature Impact Analysis:")
print(" study_hours   -> Positive ")
print(" attendece     -> Positive ")
print(" previous_marks ->  strong Positive")
print(" sleep          -> Balanced ")
print(" social_media    ->Negative")
print(" stress          ->Mixed")

model = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('regressor',RandomForestRegressor(n_estimators=100,random_state=42))

])
#Train/test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#train model
print("\n Training model...")
model.fit(X_train,y_train)
#evaluate 
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"\n Model Performace:")
print(f" Mean Aboslute Error: {mae:2f} points")
print(f" R² Score: {r2:.4f}")
#feature importance
print("\n Feature importance:")
feature_importance = dict(zip(X.columns,model.named_steps['regressor'].feature_importances_))
for feature,importace in sorted(feature_importance.items(),key=lambda x:x[1],reverse=True):
    print(f"  {feature}: {importace:.4f}")

#Save model
joblib.dump(model,'grade_model.pkl')
print("\ Model trained and saved as grade_model.pkl")
print("=" * 60)