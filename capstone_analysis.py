# ==========================================================
# CAPSTONE PROJECT
# Integrated Business Intelligence & Customer Analytics
# Week 8 - The Developers Arena
# Developed By: Ruchika Singh
# ==========================================================

# =========================
# Import Libraries
# =========================

import os
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# =========================
# Project Settings
# =========================

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

print("="*70)
print("WEEK 8 CAPSTONE PROJECT")
print("Integrated Business Intelligence & Customer Analytics")
print("="*70)

# =========================
# Folder Structure
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIS_FOLDER = os.path.join(BASE_DIR, "visualizations")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

os.makedirs(VIS_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# =========================
# Load Datasets
# =========================

print("\nLoading datasets...")

sales = pd.read_csv("sales_data.csv")

churn = pd.read_csv("customer_churn.csv")

house = pd.read_csv("house_prices.csv")

print("Datasets Loaded Successfully.")

# =========================
# Dataset Information
# =========================

print("\n================ SALES DATA ================")
print(sales.head())

print("\nShape :", sales.shape)

print("\n================ CUSTOMER CHURN ================")
print(churn.head())

print("\nShape :", churn.shape)

print("\n================ HOUSE PRICE ================")
print(house.head())

print("\nShape :", house.shape)

# =========================
# Missing Values
# =========================

print("\nChecking Missing Values...")

print("\nSales")
print(sales.isnull().sum())

print("\nCustomer Churn")
print(churn.isnull().sum())

print("\nHouse Prices")
print(house.isnull().sum())

# =========================
# Duplicate Records
# =========================

print("\nDuplicate Records")

print("Sales :", sales.duplicated().sum())

print("Customer :", churn.duplicated().sum())

print("House :", house.duplicated().sum())

# Remove duplicates

sales.drop_duplicates(inplace=True)

churn.drop_duplicates(inplace=True)

house.drop_duplicates(inplace=True)

# =========================
# Basic Cleaning
# =========================

sales.columns = sales.columns.str.strip()

churn.columns = churn.columns.str.strip()

house.columns = house.columns.str.strip()

# Date Conversion (Sales)

if "Date" in sales.columns:
    sales["Date"] = pd.to_datetime(sales["Date"])

# =========================
# Basic Statistics
# =========================

print("\nSales Statistics")
print(sales.describe(include="all"))

print("\nCustomer Statistics")
print(churn.describe(include="all"))

print("\nHouse Price Statistics")
print(house.describe(include="all"))

# =========================
# Data Types
# =========================

print("\nSales Data Types")
print(sales.dtypes)

print("\nCustomer Data Types")
print(churn.dtypes)

print("\nHouse Data Types")
print(house.dtypes)

print("\nData Cleaning Completed Successfully.")

print("="*70)
print("PART 1 COMPLETED")
print("="*70)# ==========================================================
# PART 2 : SALES DATA ANALYSIS
# ==========================================================

print("\n" + "="*70)
print("SALES DATA ANALYSIS")
print("="*70)

# -----------------------------
# Display Columns
# -----------------------------
print("\nSales Columns:")
print(sales.columns.tolist())

# -----------------------------
# Auto Detect Revenue Column
# -----------------------------
revenue_col = None

possible_revenue = [
    "Revenue",
    "Sales",
    "Amount",
    "Total",
    "TotalSales",
    "Sale Amount",
    "Price"
]

for col in possible_revenue:
    if col in sales.columns:
        revenue_col = col
        break

if revenue_col is None:
    revenue_col = sales.select_dtypes(include=np.number).columns[-1]

print(f"\nRevenue Column Selected : {revenue_col}")

# -----------------------------
# Basic KPIs
# -----------------------------
total_revenue = sales[revenue_col].sum()
average_sale = sales[revenue_col].mean()
highest_sale = sales[revenue_col].max()
lowest_sale = sales[revenue_col].min()

print("\nKEY PERFORMANCE INDICATORS")
print("-"*40)
print(f"Total Revenue : {total_revenue:,.2f}")
print(f"Average Sale  : {average_sale:,.2f}")
print(f"Highest Sale  : {highest_sale:,.2f}")
print(f"Lowest Sale   : {lowest_sale:,.2f}")

# ==========================================================
# Chart 1 : Revenue Distribution
# ==========================================================

plt.figure(figsize=(10,6))
sns.histplot(sales[revenue_col], bins=30, kde=True)
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(os.path.join(VIS_FOLDER,
                         "sales_revenue_distribution.png"))

plt.show()

# ==========================================================
# Chart 2 : Box Plot
# ==========================================================

plt.figure(figsize=(8,6))
sns.boxplot(y=sales[revenue_col])

plt.title("Revenue Box Plot")

plt.tight_layout()

plt.savefig(os.path.join(VIS_FOLDER,
                         "sales_boxplot.png"))

plt.show()

# ==========================================================
# Monthly Revenue
# ==========================================================

if "Date" in sales.columns:

    sales["Month"] = sales["Date"].dt.to_period("M").astype(str)

    monthly_sales = sales.groupby("Month")[revenue_col].sum().reset_index()

    plt.figure(figsize=(12,6))

    sns.lineplot(
        data=monthly_sales,
        x="Month",
        y=revenue_col,
        marker="o"
    )

    plt.xticks(rotation=45)

    plt.title("Monthly Revenue")

    plt.tight_layout()

    plt.savefig(os.path.join(
        VIS_FOLDER,
        "monthly_sales_trend.png"
    ))

    plt.show()

# ==========================================================
# Product Analysis
# ==========================================================

product_col = None

possible_products = [
    "Product",
    "Product Name",
    "Product_Name",
    "Item"
]

for col in possible_products:
    if col in sales.columns:
        product_col = col
        break

if product_col:

    top_products = (
        sales.groupby(product_col)[revenue_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(12,6))

    top_products.plot(kind="bar")

    plt.title("Top Products by Revenue")

    plt.ylabel("Revenue")

    plt.tight_layout()

    plt.savefig(os.path.join(
        VIS_FOLDER,
        "top_products.png"
    ))

    plt.show()

# ==========================================================
# Region Analysis
# ==========================================================

region_col = None

possible_region = [
    "Region",
    "State",
    "City",
    "Location"
]

for col in possible_region:
    if col in sales.columns:
        region_col = col
        break

if region_col:

    region_sales = (
        sales.groupby(region_col)[revenue_col]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12,6))

    region_sales.plot(kind="bar")

    plt.title("Revenue by Region")

    plt.ylabel("Revenue")

    plt.tight_layout()

    plt.savefig(os.path.join(
        VIS_FOLDER,
        "region_sales.png"
    ))

    plt.show()

# ==========================================================
# Interactive Dashboard
# ==========================================================

fig = px.histogram(
    sales,
    x=revenue_col,
    title="Interactive Revenue Dashboard",
    nbins=40
)

fig.write_html(
    os.path.join(
        VIS_FOLDER,
        "sales_dashboard.html"
    )
)

print("\nInteractive Dashboard Saved Successfully.")

print("\nSales Analysis Completed Successfully.")
print("="*70)
print("PART 2 COMPLETED")
print("="*70)# ==========================================================
# PART 3 : CUSTOMER CHURN ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("CUSTOMER CHURN ANALYSIS")
print("=" * 70)

# ----------------------------------------------------------
# Display Available Columns
# ----------------------------------------------------------

print("\nCustomer Dataset Columns:")
print(churn.columns.tolist())

# ----------------------------------------------------------
# Detect Churn Column
# ----------------------------------------------------------

churn_col = None

possible_churn = [
    "Churn",
    "Exited",
    "Attrition",
    "Status",
    "Customer_Status"
]

for col in possible_churn:
    if col in churn.columns:
        churn_col = col
        break

if churn_col is None:
    print("\nChurn column not found!")
else:
    print(f"\nDetected Churn Column : {churn_col}")

# ----------------------------------------------------------
# Customer KPIs
# ----------------------------------------------------------

total_customers = len(churn)

print("\nTotal Customers :", total_customers)

if churn_col:

    print("\nCustomer Status Count")

    print(churn[churn_col].value_counts())

    churn_rate = (
        churn[churn_col].value_counts(normalize=True) * 100
    )

    print("\nCustomer Churn Percentage")

    print(churn_rate)

# ----------------------------------------------------------
# Chart 1 : Churn Distribution
# ----------------------------------------------------------

if churn_col:

    plt.figure(figsize=(7,5))

    sns.countplot(data=churn, x=churn_col)

    plt.title("Customer Churn Distribution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VIS_FOLDER,
            "customer_churn_distribution.png"
        )
    )

    plt.show()

# ----------------------------------------------------------
# Chart 2 : Gender Analysis
# ----------------------------------------------------------

gender_col = None

possible_gender = [
    "Gender",
    "gender",
    "Sex"
]

for col in possible_gender:
    if col in churn.columns:
        gender_col = col
        break

if gender_col and churn_col:

    plt.figure(figsize=(8,5))

    sns.countplot(
        data=churn,
        x=gender_col,
        hue=churn_col
    )

    plt.title("Gender vs Churn")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VIS_FOLDER,
            "gender_vs_churn.png"
        )
    )

    plt.show()

# ----------------------------------------------------------
# Chart 3 : Contract Type
# ----------------------------------------------------------

contract_col = None

possible_contract = [
    "Contract",
    "ContractType",
    "Contract Type"
]

for col in possible_contract:
    if col in churn.columns:
        contract_col = col
        break

if contract_col and churn_col:

    plt.figure(figsize=(10,5))

    sns.countplot(
        data=churn,
        x=contract_col,
        hue=churn_col
    )

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VIS_FOLDER,
            "contract_vs_churn.png"
        )
    )

    plt.show()

# ----------------------------------------------------------
# Chart 4 : Monthly Charges
# ----------------------------------------------------------

monthly_col = None

possible_monthly = [
    "MonthlyCharges",
    "Monthly Charges",
    "Charges"
]

for col in possible_monthly:
    if col in churn.columns:
        monthly_col = col
        break

if monthly_col and churn_col:

    plt.figure(figsize=(10,6))

    sns.boxplot(
        data=churn,
        x=churn_col,
        y=monthly_col
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VIS_FOLDER,
            "monthly_charges_vs_churn.png"
        )
    )

    plt.show()

# ----------------------------------------------------------
# Chart 5 : Tenure Distribution
# ----------------------------------------------------------

tenure_col = None

possible_tenure = [
    "Tenure",
    "tenure",
    "Months"
]

for col in possible_tenure:
    if col in churn.columns:
        tenure_col = col
        break

if tenure_col:

    plt.figure(figsize=(10,6))

    sns.histplot(
        churn[tenure_col],
        kde=True,
        bins=30
    )

    plt.title("Customer Tenure Distribution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VIS_FOLDER,
            "customer_tenure_distribution.png"
        )
    )

    plt.show()

# ----------------------------------------------------------
# Correlation Heatmap
# ----------------------------------------------------------

numeric_df = churn.select_dtypes(include=np.number)

if len(numeric_df.columns) > 1:

    plt.figure(figsize=(12,8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Customer Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VIS_FOLDER,
            "customer_heatmap.png"
        )
    )

    plt.show()

# ----------------------------------------------------------
# Interactive Pie Chart
# ----------------------------------------------------------

if churn_col:

    fig = px.pie(
        churn,
        names=churn_col,
        title="Customer Churn Dashboard"
    )

    fig.write_html(
        os.path.join(
            VIS_FOLDER,
            "customer_dashboard.html"
        )
    )

print("\nInteractive Customer Dashboard Saved.")

# ----------------------------------------------------------
# Business Insights
# ----------------------------------------------------------

print("\nBUSINESS INSIGHTS")

if churn_col:

    print("- Customer churn behaviour analyzed.")

if gender_col:

    print("- Gender wise churn identified.")

if contract_col:

    print("- Contract impact evaluated.")

if monthly_col:

    print("- Monthly charges compared.")

if tenure_col:

    print("- Customer tenure analysed.")

print("\nCustomer Churn Analysis Completed Successfully.")

print("="*70)
print("PART 3 COMPLETED")
print("="*70)# ==========================================================
# PART 4 : HOUSE PRICE ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("HOUSE PRICE ANALYSIS")
print("=" * 70)

# ----------------------------------------------------------
# Display Columns
# ----------------------------------------------------------

print("\nHouse Dataset Columns:")
print(house.columns.tolist())

# ----------------------------------------------------------
# Detect Price Column
# ----------------------------------------------------------

price_col = None

possible_price = [
    "Price",
    "SalePrice",
    "price",
    "SellingPrice"
]

for col in possible_price:
    if col in house.columns:
        price_col = col
        break

if price_col is None:
    price_col = house.select_dtypes(include=np.number).columns[-1]

print(f"\nDetected Price Column : {price_col}")

# ----------------------------------------------------------
# Basic KPIs
# ----------------------------------------------------------

print("\nHOUSE PRICE KPIs")

print("-"*40)

print("Total Houses :", len(house))
print("Average Price :", round(house[price_col].mean(),2))
print("Highest Price :", round(house[price_col].max(),2))
print("Lowest Price :", round(house[price_col].min(),2))
print("Median Price :", round(house[price_col].median(),2))

# ==========================================================
# Chart 1 : Price Distribution
# ==========================================================

plt.figure(figsize=(10,6))

sns.histplot(house[price_col], bins=30, kde=True)

plt.title("House Price Distribution")

plt.tight_layout()

plt.savefig(os.path.join(
    VIS_FOLDER,
    "house_price_distribution.png"
))

plt.show()

# ==========================================================
# Chart 2 : Price Boxplot
# ==========================================================

plt.figure(figsize=(8,6))

sns.boxplot(y=house[price_col])

plt.title("House Price Boxplot")

plt.tight_layout()

plt.savefig(os.path.join(
    VIS_FOLDER,
    "house_price_boxplot.png"
))

plt.show()

# ==========================================================
# Correlation Heatmap
# ==========================================================

numeric_house = house.select_dtypes(include=np.number)

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_house.corr(),
    cmap="coolwarm",
    annot=True
)

plt.title("House Feature Correlation")

plt.tight_layout()

plt.savefig(os.path.join(
    VIS_FOLDER,
    "house_correlation_heatmap.png"
))

plt.show()

# ==========================================================
# Scatter Plot
# ==========================================================

numeric_cols = numeric_house.columns.tolist()

feature = None

for col in numeric_cols:
    if col != price_col:
        feature = col
        break

if feature:

    plt.figure(figsize=(10,6))

    sns.scatterplot(
        data=house,
        x=feature,
        y=price_col
    )

    plt.title(f"{feature} vs House Price")

    plt.tight_layout()

    plt.savefig(os.path.join(
        VIS_FOLDER,
        "feature_vs_price.png"
    ))

    plt.show()

# ==========================================================
# Linear Regression
# ==========================================================

features = numeric_house.drop(columns=[price_col], errors="ignore")

target = house[price_col]

if len(features.columns) > 0:

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nMODEL PERFORMANCE")

    print("-"*40)

    print("MAE :", round(mean_absolute_error(y_test,predictions),2))

    print("RMSE :", round(
        np.sqrt(mean_squared_error(y_test,predictions)),2
    ))

    print("R2 Score :", round(
        r2_score(y_test,predictions),3
    ))

    # Prediction Plot

    plt.figure(figsize=(8,6))

    plt.scatter(y_test,predictions)

    plt.xlabel("Actual Price")

    plt.ylabel("Predicted Price")

    plt.title("Actual vs Predicted House Prices")

    plt.tight_layout()

    plt.savefig(os.path.join(
        VIS_FOLDER,
        "house_prediction.png"
    ))

    plt.show()

# ==========================================================
# Interactive Dashboard
# ==========================================================

fig = px.scatter(
    house,
    x=feature,
    y=price_col,
    title="Interactive House Price Dashboard"
)

fig.write_html(os.path.join(
    VIS_FOLDER,
    "house_dashboard.html"
))

print("\nInteractive Dashboard Saved.")

# ==========================================================
# Business Insights
# ==========================================================

print("\nBUSINESS INSIGHTS")

print("- Average property value identified.")

print("- Price distribution analysed.")

print("- Feature correlation calculated.")

print("- Linear Regression model built.")

print("- Price prediction completed.")

print("- Market trend explored.")

print("\nHouse Price Analysis Completed Successfully.")

print("="*70)
print("PART 4 COMPLETED")
print("="*70)# ==========================================================
# PART 5 : STATISTICAL ANALYSIS & BUSINESS INSIGHTS
# ==========================================================

print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

# ----------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ----------------------------------------------------------

print("\nDESCRIPTIVE STATISTICS")

print("\nSales Dataset")
print(sales.describe())

print("\nCustomer Dataset")
print(churn.describe())

print("\nHouse Dataset")
print(house.describe())

# ----------------------------------------------------------
# SALES CORRELATION
# ----------------------------------------------------------

print("\nGenerating Sales Correlation Matrix...")

sales_numeric = sales.select_dtypes(include=np.number)

if len(sales_numeric.columns) > 1:

    corr = sales_numeric.corr()

    plt.figure(figsize=(10,8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="viridis"
    )

    plt.title("Sales Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VIS_FOLDER,
            "sales_heatmap.png"
        )
    )

    plt.show()

# ----------------------------------------------------------
# HYPOTHESIS TESTING
# ----------------------------------------------------------

print("\nHypothesis Testing")

numeric_columns = sales.select_dtypes(include=np.number).columns

if len(numeric_columns) >= 2:

    col1 = numeric_columns[0]
    col2 = numeric_columns[1]

    statistic, p_value = stats.ttest_ind(
        sales[col1],
        sales[col2]
    )

    print("\nT-Test Results")

    print("Column 1 :", col1)
    print("Column 2 :", col2)

    print("T Statistic :", statistic)
    print("P Value :", p_value)

    if p_value < 0.05:

        print("Result : Significant Difference Found")

    else:

        print("Result : No Significant Difference")

# ----------------------------------------------------------
# ANOVA TEST
# ----------------------------------------------------------

print("\nRunning ANOVA Test")

categorical = sales.select_dtypes(include="object").columns

if len(categorical) > 0:

    category = categorical[0]

    revenue = sales.select_dtypes(include=np.number).columns[-1]

    groups = []

    for value in sales[category].unique():

        groups.append(
            sales[
                sales[category] == value
            ][revenue]
        )

    if len(groups) > 1:

        f_value, p = stats.f_oneway(*groups)

        print("\nANOVA Results")

        print("F Value :", f_value)

        print("P Value :", p)

# ----------------------------------------------------------
# CUSTOMER CHURN RATE
# ----------------------------------------------------------

print("\nCustomer Churn Summary")

if churn_col:

    churn_summary = churn[churn_col].value_counts()

    print(churn_summary)

    plt.figure(figsize=(7,7))

    plt.pie(
        churn_summary,
        labels=churn_summary.index,
        autopct="%1.1f%%"
    )

    plt.title("Customer Churn Percentage")

    plt.savefig(
        os.path.join(
            VIS_FOLDER,
            "customer_churn_pie.png"
        )
    )

    plt.show()

# ----------------------------------------------------------
# HOUSE PRICE SUMMARY
# ----------------------------------------------------------

print("\nHouse Price Summary")

print(house[price_col].describe())

# ----------------------------------------------------------
# BUSINESS INSIGHTS
# ----------------------------------------------------------

business_insights = [

"Increase customer retention programs.",

"Improve regional sales performance.",

"Focus on high revenue products.",

"Optimize pricing strategy.",

"Identify high churn customer segments.",

"Launch loyalty programs.",

"Target premium housing customers.",

"Improve marketing campaigns using analytics.",

"Use predictive analytics for future growth.",

"Regularly monitor KPIs through dashboards."

]

print("\nBUSINESS RECOMMENDATIONS")

print("-"*60)

for i, insight in enumerate(business_insights,1):

    print(f"{i}. {insight}")

# ----------------------------------------------------------
# SAVE BUSINESS REPORT
# ----------------------------------------------------------

report_path = os.path.join(
    REPORT_FOLDER,
    "Business_Insights.txt"
)

with open(report_path,"w") as f:

    f.write("BUSINESS INSIGHTS\n\n")

    for item in business_insights:

        f.write(item+"\n")

print("\nBusiness Report Saved Successfully.")

print("="*70)
print("PART 5 COMPLETED")
print("="*70)# ==========================================================
# PART 6 : FINAL DASHBOARD & PROJECT SUMMARY
# ==========================================================

print("\n" + "="*70)
print("FINAL BUSINESS DASHBOARD")
print("="*70)

# ----------------------------------------------------------
# KPI Summary
# ----------------------------------------------------------

print("\nPROJECT KPI SUMMARY")

try:
    print(f"Total Sales Records      : {len(sales)}")
except:
    pass

try:
    print(f"Total Customers          : {len(churn)}")
except:
    pass

try:
    print(f"Total Houses             : {len(house)}")
except:
    pass

try:
    print(f"Total Revenue            : {total_revenue:,.2f}")
except:
    pass

try:
    print(f"Average Revenue          : {average_sale:,.2f}")
except:
    pass

try:
    print(f"Average House Price      : {house[price_col].mean():,.2f}")
except:
    pass

# ----------------------------------------------------------
# Dashboard Charts
# ----------------------------------------------------------

dashboard = go.Figure()

# Sales Revenue Histogram
try:
    dashboard.add_trace(
        go.Histogram(
            x=sales[revenue_col],
            name="Sales Revenue"
        )
    )
except:
    pass

# Customer Churn Pie
try:
    churn_counts = churn[churn_col].value_counts()

    dashboard.add_trace(
        go.Pie(
            labels=churn_counts.index,
            values=churn_counts.values,
            name="Customer Churn",
            domain={"x":[0.55,1],"y":[0.55,1]}
        )
    )
except:
    pass

dashboard.update_layout(
    title="Integrated Business Intelligence Dashboard",
    template="plotly_white",
    height=700,
    width=1100
)

dashboard.write_html(
    os.path.join(
        VIS_FOLDER,
        "Integrated_Dashboard.html"
    )
)

print("\nInteractive Dashboard Saved Successfully.")

# ----------------------------------------------------------
# Project Summary Report
# ----------------------------------------------------------

summary_file = os.path.join(
    REPORT_FOLDER,
    "Project_Summary.txt"
)

with open(summary_file,"w",encoding="utf-8") as f:

    f.write("CAPSTONE PROJECT SUMMARY\n")
    f.write("="*60+"\n\n")

    f.write("Datasets Used\n")
    f.write("------------------------\n")
    f.write("1. Sales Dataset\n")
    f.write("2. Customer Churn Dataset\n")
    f.write("3. House Price Dataset\n\n")

    f.write("Analysis Performed\n")
    f.write("------------------------\n")
    f.write("- Data Cleaning\n")
    f.write("- Exploratory Data Analysis\n")
    f.write("- Statistical Analysis\n")
    f.write("- Correlation Analysis\n")
    f.write("- Hypothesis Testing\n")
    f.write("- Customer Churn Analysis\n")
    f.write("- Sales Analysis\n")
    f.write("- House Price Analysis\n")
    f.write("- Linear Regression\n")
    f.write("- Business Intelligence Dashboard\n\n")

    f.write("Business Recommendations\n")
    f.write("------------------------\n")

    for item in business_insights:
        f.write(f"- {item}\n")

print("Project Summary Generated.")

# ----------------------------------------------------------
# Requirements File
# ----------------------------------------------------------

requirements = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "plotly",
    "scipy",
    "scikit-learn"
]

req_path = os.path.join(BASE_DIR,"requirements.txt")

with open(req_path,"w") as f:

    for lib in requirements:
        f.write(lib+"\n")

print("requirements.txt Generated.")

# ----------------------------------------------------------
# Completion Message
# ----------------------------------------------------------

print("\n"+"="*70)
print("CAPSTONE PROJECT COMPLETED SUCCESSFULLY")
print("="*70)

print("""
Project Includes

✔ Data Cleaning
✔ Sales Analysis
✔ Customer Churn Analysis
✔ House Price Analysis
✔ Descriptive Statistics
✔ Correlation Analysis
✔ Hypothesis Testing
✔ Linear Regression
✔ Business Recommendations
✔ Interactive Dashboard
✔ Reports
✔ Visualizations
✔ requirements.txt

Project Status : READY FOR GITHUB
""")

print("="*70)