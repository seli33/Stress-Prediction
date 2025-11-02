import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# Load CSV files
df1 = pd.read_csv("StressLevelDataset.csv")
df2 = pd.read_csv("Student_ScreenTime_Sleep_MentalFocus_2025.csv")
df3 = pd.read_csv("students_mental_health_survey.csv")
df4 = pd.read_csv("merged_data.csv")

# Check shapes
print("Shapes:", df1.shape, df2.shape, df3.shape, df4.shape)

# Match lengths
min_len = min(len(df1), len(df2), len(df3))
df1, df2, df3 = df1[:min_len], df2[:min_len], df3[:min_len]

# Add Student_ID to all three
for df in [df1, df2, df3]:
    df["Student_ID"] = ["S" + str(1000 + i) for i in range(len(df))]

# Merge datasets
temp = pd.merge(df1, df2, on="Student_ID", how="inner")
merged = pd.merge(temp, df3, on="Student_ID", how="inner")
print("Merged shape:", merged.shape)

# Clean and rename columns
df4.rename(columns={
    'Age_x': 'Age',
    'Physical_Activity_y': 'Physical_Activity'
}, inplace=True)
df4.columns = df4.columns.str.strip()

# Select columns safely (handles both old and renamed column names)
columns_to_use = [
    'self_esteem',
    'stress_level',
    'sleep_quality',
    'Age_x' if 'Age_x' in df4.columns else 'Age',
    'Hours_of_Screen_Time',
    'Physical_Activity_y' if 'Physical_Activity_y' in df4.columns else 'Physical_Activity',
    'Gender',
    'study_load'
]
selected_df = df4[columns_to_use]


# Drop duplicates and clean strings
selected_df = selected_df.drop_duplicates()
selected_df['Gender'] = selected_df['Gender'].str.strip().str.capitalize()
selected_df['Physical_Activity'] = selected_df['Physical_Activity'].str.strip().str.capitalize()

# Convert numeric and filter outliers
selected_df['Age'] = pd.to_numeric(selected_df['Age'], errors='coerce')
Q1 = selected_df['Hours_of_Screen_Time'].quantile(0.25)
Q3 = selected_df['Hours_of_Screen_Time'].quantile(0.75)
IQR = Q3 - Q1
selected_df = selected_df[
    (selected_df['Hours_of_Screen_Time'] >= Q1 - 1.5 * IQR) &
    (selected_df['Hours_of_Screen_Time'] <= Q3 + 1.5 * IQR)
]

# Encode categorical values
Physical_Activity_map = {'Low': 1, 'Moderate': 2, 'High': 3}
Gender_map = {'Male': 0, 'Female': 1}
selected_df['Physical_Activity_encoded'] = selected_df['Physical_Activity'].map(Physical_Activity_map)
selected_df['Gender_encoded'] = selected_df['Gender'].map(Gender_map)

# Define features and target
features = ['self_esteem', 'sleep_quality', 'Hours_of_Screen_Time', 'study_load',
            'Gender_encoded', 'Physical_Activity_encoded']
X = selected_df[features]
y = selected_df['stress_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "stress_model.pkl")

# Test with sample input 1
new_input = pd.DataFrame([{
    'self_esteem': 20,
    'sleep_quality': 3,
    'Hours_of_Screen_Time': 7.5,
    'study_load': 6,
    'Gender_encoded': 1,
    'Physical_Activity_encoded': 0
}])
scaled_input = scaler.transform(new_input)
prediction = model.predict(scaled_input)[0]
stress_map = {0: "Normal", 1: "Stressed", 2: "Highly Prone"}
print("Predicted Stress Level (Test 1):", stress_map.get(prediction, prediction))

# Test with sample input 2
new_input = pd.DataFrame([{
    'self_esteem': 7,
    'sleep_quality': 8,
    'Hours_of_Screen_Time': 11,
    'study_load': 2,
    'Gender_encoded': 0,
    'Physical_Activity_encoded': 1
}])

scaled_input = scaler.transform(new_input)

prediction = model.predict(scaled_input)[0]
print("Predicted Stress Level (Test 2):", stress_map.get(prediction, prediction))
