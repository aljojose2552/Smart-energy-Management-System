🔋 Smart Energy Management Using Behaviour Clustering
📌 Project Overview

This project explores how household electricity consumption and solar PV usage can be analysed using data analytics and machine learning to support smarter energy management decisions. Instead of treating all households the same, the system identifies distinct behavioural patterns and evaluates how targeted demand-response strategies can reduce grid dependency during peak hours.

The work is data-driven and simulation-based, focusing on feasibility analysis rather than deploying physical IoT infrastructure.

🎯 Objectives

Analyse residential electricity consumption and solar production behaviour

Identify distinct household energy profiles using unsupervised learning

Quantify peak-hour demand and grid reliance

Simulate load-shifting and solar self-consumption strategies

Evaluate potential reductions in grid dependency

📊 Dataset

High-resolution smart-meter data (hourly)

Multiple households with:

Electricity consumption (Wh)

Solar PV production (Wh)

Grid import (Wh)

One full year of data per household (2020)

All datasets are pre-processed, cleaned, and standardised before analysis.

🧠 Methodology

Data Pre-processing

Cleaning, resampling, and feature engineering

Creation of behavioural metrics (peak usage, solar utilisation, grid reliance)

Behavioural Clustering

K-Means clustering with feature scaling

Cluster validation using silhouette analysis

Interpretation of clusters based on real-world energy behaviour

Scenario Simulation

Load-shifting during evening peak hours

Increased solar self-consumption

Comparison of baseline vs simulated outcomes

Visual Analytics

Load profiles

Solar vs grid dependency

Cluster-wise comparisons

🛠️ Technologies Used

Python

pandas, NumPy

scikit-learn (K-Means, StandardScaler)

matplotlib / seaborn

Streamlit (interactive dashboard)

Jupyter Notebook

📈 Key Outcomes

Clear identification of household energy behaviour clusters

Evidence that one-size-fits-all energy strategies are ineffective

Demonstrated potential for:

Reduced peak-hour grid load

Improved solar utilisation

More targeted demand-response policies

🚀 Future Enhancements

Integrate weather data to improve solar production analysis

Extend clustering to larger household samples

Add reinforcement learning for adaptive load control

Deploy as a real-time smart energy decision support system

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smart-energy-management-system.git
cd smart-energy-management-system
