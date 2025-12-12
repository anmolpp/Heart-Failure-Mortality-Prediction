import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import os, urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------------------
dataset_path = "heart_failure_clinical_records_dataset.csv"
if not os.path.exists(dataset_path):
    try:
        print("📥 Downloading dataset...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/heart_failure_clinical_records_dataset.csv"
        urllib.request.urlretrieve(url, dataset_path)
        print("✅ Dataset downloaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Dataset download failed: {e}")

df = pd.read_csv(dataset_path)

# ------------------------------------------------------------
# STEP 2: Train Model
# ------------------------------------------------------------
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
accuracy = model.score(X_test_scaled, y_test)

# ------------------------------------------------------------
# STEP 3: GUI Layout
# ------------------------------------------------------------
root = tk.Tk()
root.title("🫀 Heart Failure Mortality Prediction System")
root.geometry("1350x900")
root.configure(bg="#e8f4fc")

main_frame = tk.Frame(root, bg="#e8f4fc")
main_frame.pack(fill="both", expand=True)

left_frame = tk.Frame(main_frame, bg="#e8f4fc", width=500)
left_frame.pack(side="left", fill="y", padx=20, pady=20)

right_frame = tk.Frame(main_frame, bg="#f9fbfd", width=750, relief="sunken", bd=2)
right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

title = tk.Label(left_frame, text="Heart Failure Mortality Prediction",
                 font=("Arial", 18, "bold"), bg="#e8f4fc", fg="#0d47a1")
title.pack(pady=10)

# ------------------------------------------------------------
# STEP 4: Input Fields + Normal Ranges
# ------------------------------------------------------------
fields = [
    ("Age", "18–65"),
    ("Anaemia (1=yes,0=no)", "0 or 1"),
    ("Creatinine Phosphokinase (mcg/L)", "30–200"),
    ("Diabetes (1=yes,0=no)", "0 or 1"),
    ("Ejection Fraction (%)", "50–70"),
    ("High Blood Pressure (1=yes,0=no)", "0 or 1"),
    ("Platelets (kiloplatelets/mL)", "150000–400000"),
    ("Serum Creatinine (mg/dL)", "0.6–1.3"),
    ("Serum Sodium (mEq/L)", "135–145"),
    ("Sex (1=Male,0=Female)", "0 or 1"),
    ("Smoking (1=yes,0=no)", "0 or 1"),
    ("Follow-up Period (days)", "1–365")
]

entries = {}
for field, normal_range in fields:
    row = tk.Frame(left_frame, bg="#e8f4fc")
    lbl = tk.Label(row, text=f"{field}:", font=("Arial", 11), bg="#e8f4fc")
    ent = tk.Entry(row, font=("Arial", 11), width=10)
    normal_lbl = tk.Label(row, text=f"(Normal: {normal_range})", font=("Arial", 9, "italic"),
                          bg="#e8f4fc", fg="#0d47a1")
    row.pack(anchor="w", pady=3)
    lbl.pack(side=tk.LEFT, padx=5)
    ent.pack(side=tk.LEFT, padx=5)
    normal_lbl.pack(side=tk.LEFT)
    entries[field] = ent

# ------------------------------------------------------------
# STEP 5: Prediction Function
# ------------------------------------------------------------
def predict_manual():
    try:
        user_data = {
            'age': float(entries["Age"].get()),
            'anaemia': int(entries["Anaemia (1=yes,0=no)"].get()),
            'creatinine_phosphokinase': float(entries["Creatinine Phosphokinase (mcg/L)"].get()),
            'diabetes': int(entries["Diabetes (1=yes,0=no)"].get()),
            'ejection_fraction': float(entries["Ejection Fraction (%)"].get()),
            'high_blood_pressure': int(entries["High Blood Pressure (1=yes,0=no)"].get()),
            'platelets': float(entries["Platelets (kiloplatelets/mL)"].get()),
            'serum_creatinine': float(entries["Serum Creatinine (mg/dL)"].get()),
            'serum_sodium': float(entries["Serum Sodium (mEq/L)"].get()),
            'sex': int(entries["Sex (1=Male,0=Female)"].get()),
            'smoking': int(entries["Smoking (1=yes,0=no)"].get()),
            'time': float(entries["Follow-up Period (days)"].get())
        }

        user_df = pd.DataFrame([user_data])
        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]
        prob = model.predict_proba(user_scaled)[0][1] * 100

        if prediction == 1:
            msg = f"⚠️ High Risk of Mortality ({prob:.1f}% chance)"
            color = "red"
        else:
            msg = f"✅ Low Risk of Mortality ({100 - prob:.1f}% chance)"
            color = "green"

        result_label.config(text=msg, fg=color)
        messagebox.showinfo("Prediction Result", msg)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

# ------------------------------------------------------------
# STEP 6: Upload CSV
# ------------------------------------------------------------
def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    try:
        user_df = pd.read_csv(file_path)
        if 'DEATH_EVENT' in user_df.columns:
            user_df = user_df.drop(columns=['DEATH_EVENT'])

        missing_cols = set(X.columns) - set(user_df.columns)
        if missing_cols:
            messagebox.showerror("Error", f"Missing columns: {missing_cols}")
            return

        user_scaled = scaler.transform(user_df[X.columns])
        predictions = model.predict(user_scaled)
        probs = model.predict_proba(user_scaled)[:, 1] * 100

        user_df['Predicted_Risk'] = ['High' if p == 1 else 'Low' for p in predictions]
        user_df['Risk_Probability_%'] = probs
        save_path = os.path.splitext(file_path)[0] + "_predicted.csv"
        user_df.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Predictions saved to:\n{save_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Error processing file:\n{e}")

# ------------------------------------------------------------
# STEP 7: Multiple Graphs Display (VERTICAL Layout)
# ------------------------------------------------------------
# ------------------------------------------------------------
# STEP 7: Multiple Graphs Display (VERTICAL Layout + SCROLLBAR)
# ------------------------------------------------------------
def show_multiple_graphs():
    # Clear existing widgets
    for widget in right_frame.winfo_children():
        widget.destroy()

    # Create a canvas and scrollbar inside right_frame
    canvas = tk.Canvas(right_frame, bg="#f9fbfd")
    scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#f9fbfd")

    # Configure scrolling region
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # --- Create Matplotlib Figure (taller for vertical scrolling) ---
    fig, axes = plt.subplots(4, 1, figsize=(6, 12))  # taller figure
    fig.subplots_adjust(hspace=0.8)

    # 1️⃣ Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:6]
    axes[0].barh(np.array(X.columns)[indices][::-1], importances[indices][::-1], color='teal')
    axes[0].set_title("Feature Importance", fontsize=9)
    axes[0].tick_params(axis='y', labelsize=8)
    axes[0].tick_params(axis='x', labelsize=7)

    # 2️⃣ Correlation Heatmap
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr[['DEATH_EVENT']].sort_values(by='DEATH_EVENT', ascending=False),
                ax=axes[1], cmap='coolwarm', cbar=False, annot=True, fmt=".2f", annot_kws={"size": 7})
    axes[1].set_title("Correlation with Mortality", fontsize=9)
    axes[1].tick_params(axis='y', labelsize=7)

    # 3️⃣ Healthy vs User Comparison (Bar Graph)
    try:
        user_vals = [float(entries[f[0]].get()) for f in fields]
        healthy_vals = [40, 0, 100, 0, 60, 0, 250000, 1.0, 140, 1, 0, 365]
        scale = [1, 1, 1000, 1, 1, 1, 100000, 1, 1, 1, 1, 1]
        user_scaled = [u / s for u, s in zip(user_vals, scale)]
        healthy_scaled = [h / s for h, s in zip(healthy_vals, scale)]
        x = np.arange(len(user_scaled))
        axes[2].bar(x - 0.2, healthy_scaled, 0.4, label="Healthy", color="green")
        axes[2].bar(x + 0.2, user_scaled, 0.4, label="User", color="orange")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f[0][:4] for f in fields], rotation=45, fontsize=6)
        axes[2].legend(fontsize=6)
        axes[2].set_title("Healthy vs User", fontsize=9)
    except:
        axes[2].text(0.3, 0.5, "Enter Inputs", fontsize=8, color="red")

    # 4️⃣ Age vs Mortality Scatter
    axes[3].scatter(df['age'], df['DEATH_EVENT'], s=8, c=df['DEATH_EVENT'], cmap='coolwarm')
    axes[3].set_xlabel("Age", fontsize=8)
    axes[3].set_ylabel("Death (0/1)", fontsize=8)
    axes[3].set_title("Age vs Death", fontsize=9)
    axes[3].tick_params(axis='x', labelsize=7)
    axes[3].tick_params(axis='y', labelsize=7)

    # Embed the figure into Tkinter scrollable frame
    chart_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)


# ------------------------------------------------------------
# STEP 7.5: Medicine Suggestion Based on Abnormal Factors
# ------------------------------------------------------------
def suggest_medicine():
    try:
        # Get user inputs
        user_vals = {
            "Age": float(entries["Age"].get()),
            "Anaemia": int(entries["Anaemia (1=yes,0=no)"].get()),
            "Creatinine Phosphokinase": float(entries["Creatinine Phosphokinase (mcg/L)"].get()),
            "Diabetes": int(entries["Diabetes (1=yes,0=no)"].get()),
            "Ejection Fraction": float(entries["Ejection Fraction (%)"].get()),
            "High Blood Pressure": int(entries["High Blood Pressure (1=yes,0=no)"].get()),
            "Platelets": float(entries["Platelets (kiloplatelets/mL)"].get()),
            "Serum Creatinine": float(entries["Serum Creatinine (mg/dL)"].get()),
            "Serum Sodium": float(entries["Serum Sodium (mEq/L)"].get()),
            "Smoking": int(entries["Smoking (1=yes,0=no)"].get())
        }

        # Normal reference ranges
        normal_ranges = {
            "Age": (18, 65),
            "Creatinine Phosphokinase": (30, 200),
            "Ejection Fraction": (50, 70),
            "Platelets": (150000, 400000),
            "Serum Creatinine": (0.6, 1.3),
            "Serum Sodium": (135, 145)
        }

        # Medicine suggestions dictionary
        med_suggestions = {
            "Anaemia": "Iron supplements (Ferrous sulfate), Vitamin B12, Folic acid",
            "High Blood Pressure": "ACE inhibitors (Enalapril), Beta blockers (Metoprolol)",
            "Diabetes": "Metformin, Insulin, Glimepiride",
            "Creatinine Phosphokinase": "Consult doctor – may indicate muscle injury",
            "Ejection Fraction": "Diuretics (Furosemide), ACE inhibitors, Beta blockers",
            "Serum Creatinine": "Avoid NSAIDs, ensure hydration, consider nephroprotective agents",
            "Serum Sodium": "Oral rehydration salts (ORS), adjust fluid intake",
            "Platelets": "Folic acid, Vitamin B12, healthy diet, avoid alcohol",
            "Smoking": "Nicotine replacement therapy, Bupropion"
        }

        # Check for abnormalities
        abnormal_factors = []

        for key, (low, high) in normal_ranges.items():
            val = user_vals[key]
            if val < low or val > high:
                abnormal_factors.append(key)

        # Check binary disease flags
        if user_vals["Anaemia"] == 1:
            abnormal_factors.append("Anaemia")
        if user_vals["High Blood Pressure"] == 1:
            abnormal_factors.append("High Blood Pressure")
        if user_vals["Diabetes"] == 1:
            abnormal_factors.append("Diabetes")
        if user_vals["Smoking"] == 1:
            abnormal_factors.append("Smoking")

        if not abnormal_factors:
            messagebox.showinfo("💊 Medicine Suggestion", "All parameters are within normal limits.\nNo medicine required. Stay healthy!")
            return

        # Prepare suggestion text
        suggestion_text = "💊 Medicine Suggestions:\n\n"
        for factor in abnormal_factors:
            suggestion_text += f"• {factor}: {med_suggestions.get(factor, 'Consult your physician.')}\n"

        # Show medicine suggestions
        messagebox.showinfo("💊 Medicine Suggestions", suggestion_text)

        # Also show result on right panel
        for widget in right_frame.winfo_children():
            widget.destroy()
        lbl = tk.Label(right_frame, text=suggestion_text, font=("Arial", 11), justify="left", bg="#f9fbfd", wraplength=700)
        lbl.pack(padx=20, pady=20, anchor="nw")

    except Exception as e:
        messagebox.showerror("Error", f"Please enter all inputs correctly.\n{e}")

# ------------------------------------------------------------
# STEP 8: Buttons
# ------------------------------------------------------------
btn_predict = tk.Button(left_frame, text="🔍 Predict (Manual Input)", font=("Arial", 13, "bold"),
                        bg="#64b5f6", fg="white", command=predict_manual)
btn_predict.pack(pady=8)

btn_csv = tk.Button(left_frame, text="📂 Upload CSV and Predict", font=("Arial", 13, "bold"),
                    bg="#42a5f5", fg="white", command=upload_csv)
btn_csv.pack(pady=8)

btn_graph = tk.Button(left_frame, text="📊 Show All Graphs", font=("Arial", 13, "bold"),
                      bg="#1e88e5", fg="white", command=show_multiple_graphs)
btn_graph.pack(pady=8)

acc_label = tk.Label(left_frame, text=f"Model Accuracy: {accuracy*100:.2f}%",
                     font=("Arial", 11, "italic"), bg="#e8f4fc", fg="#0d47a1")
acc_label.pack(pady=10)

result_label = tk.Label(left_frame, text="", font=("Arial", 13, "bold"), bg="#e8f4fc")
result_label.pack(pady=10)

btn_medicine = tk.Button(left_frame, text="💊 Suggest Medicines", font=("Arial", 13, "bold"),
                         bg="#00bfa5", fg="white", command=suggest_medicine)
btn_medicine.pack(pady=8)

root.mainloop()
