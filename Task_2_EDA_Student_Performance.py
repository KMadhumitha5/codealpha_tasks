# CodeAlpha Data Analytics Internship
# TASK 2 - Exploratory Data Analysis (EDA)
# Intern: Kondam Madhumitha
# Project: Student Performance Analysis

import io, zipfile, urllib.request
import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
raw = urllib.request.urlopen(url, timeout=30).read()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    with z.open("student-mat.csv") as f:
        df = pd.read_csv(f, sep=";")

print("TASK 2 - EXPLORATORY DATA ANALYSIS")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:")
print(df.describe().T)

print("\nQUESTION 1: Final grade distribution")
print("Mean G3:", round(df["G3"].mean(), 2))
print("Median G3:", round(df["G3"].median(), 2))
print("Minimum G3:", df["G3"].min())
print("Maximum G3:", df["G3"].max())

print("\nQUESTION 2: Study time vs final grade")
print(df.groupby("studytime")["G3"].agg(["count", "mean", "median"]))

print("\nQUESTION 3: Gender vs final grade")
print(df.groupby("sex")["G3"].agg(["count", "mean", "median"]))

print("\nQUESTION 4: Absences vs final grade")
print("Correlation:", round(df["absences"].corr(df["G3"]), 4))

print("\nQUESTION 5: Numerical correlations with G3")
numeric_cols = df.select_dtypes(include=np.number).columns
print(df[numeric_cols].corr()["G3"].sort_values(ascending=False))

q1, q3 = df["G3"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(df["G3"] < q1 - 1.5*iqr) | (df["G3"] > q3 + 1.5*iqr)]
print("\nPotential G3 outliers using IQR:", len(outliers))

print("\nCONCLUSION")
print("The EDA examines structure, data quality, distributions, group differences,")
print("correlations and potential anomalies. These are descriptive associations,")
print("not proof of causation.")
