# CodeAlpha Data Analytics Internship
# Tasks 2 & 3: Exploratory Data Analysis and Data Visualization
# Intern: Kondam Madhumitha

import io
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# -----------------------------
# 1. Load dataset
# -----------------------------
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

raw = urllib.request.urlopen(URL, timeout=30).read()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    with z.open("student-mat.csv") as f:
        df = pd.read_csv(f, sep=";")

print("=" * 60)
print("CODEALPHA - STUDENT PERFORMANCE ANALYSIS")
print("=" * 60)
print("Dataset shape:", df.shape)

# -----------------------------
# 2. Basic EDA
# -----------------------------
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe().T)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# -----------------------------
# 3. Analytical summaries
# -----------------------------
print("\n--- FINAL GRADE SUMMARY ---")
print("Mean G3:", round(df["G3"].mean(), 2))
print("Median G3:", round(df["G3"].median(), 2))
print("Minimum G3:", df["G3"].min())
print("Maximum G3:", df["G3"].max())

print("\n--- STUDY TIME VS FINAL GRADE ---")
print(df.groupby("studytime")["G3"].agg(["count", "mean", "median"]))

print("\n--- GENDER VS FINAL GRADE ---")
print(df.groupby("sex")["G3"].agg(["count", "mean", "median"]))

print("\n--- ABSENCES VS FINAL GRADE ---")
print("Correlation:", round(df["absences"].corr(df["G3"]), 4))

print("\n--- CORRELATIONS WITH G3 ---")
numeric_cols = df.select_dtypes(include=np.number).columns
print(df[numeric_cols].corr()["G3"].sort_values(ascending=False))

# -----------------------------
# 4. Create output folder
# -----------------------------
os.makedirs("visualizations", exist_ok=True)

# -----------------------------
# 5. Task 3 - Visualizations
# -----------------------------

# 5.1 Final grade distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=range(0, 21), discrete=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("visualizations/01_final_grade_distribution.png", dpi=200)
plt.show()

# 5.2 Study time vs final grade
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="studytime", y="G3", estimator="mean", errorbar=None)
plt.title("Average Final Grade by Study Time")
plt.xlabel("Study Time Category")
plt.ylabel("Average Final Grade")
plt.tight_layout()
plt.savefig("visualizations/02_study_time_vs_grade.png", dpi=200)
plt.show()

# 5.3 Absences vs final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="absences", y="G3", alpha=0.7)
sns.regplot(data=df, x="absences", y="G3", scatter=False)
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("visualizations/03_absences_vs_grade.png", dpi=200)
plt.show()

# 5.4 Gender vs final grade
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="sex", y="G3", estimator="mean", errorbar=None)
plt.title("Average Final Grade by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Final Grade")
plt.tight_layout()
plt.savefig("visualizations/04_gender_vs_grade.png", dpi=200)
plt.show()

# 5.5 Parental education vs final grade
parental = df.groupby(["Medu", "Fedu"])["G3"].mean().reset_index()
heat = parental.pivot(index="Medu", columns="Fedu", values="G3")

plt.figure(figsize=(9, 6))
sns.heatmap(heat, annot=True, fmt=".2f")
plt.title("Average Final Grade by Parental Education")
plt.xlabel("Father's Education Level")
plt.ylabel("Mother's Education Level")
plt.tight_layout()
plt.savefig("visualizations/05_parental_education_heatmap.png", dpi=200)
plt.show()

# 5.6 Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.savefig("visualizations/06_correlation_heatmap.png", dpi=200)
plt.show()

# 5.7 G3 box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["G3"])
plt.title("Box Plot of Final Grade")
plt.xlabel("Final Grade")
plt.tight_layout()
plt.savefig("visualizations/07_g3_boxplot.png", dpi=200)
plt.show()

print("\nAnalysis complete.")
print("Seven visualizations were saved in the 'visualizations' folder.")
