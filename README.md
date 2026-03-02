# MultiCategoryPlatform-Analysis-Dashboard
DecodeX Hackathon Final Round's Work- Financial Risk & Revenue Optimization Engine for a multicategory huge platform for Customer behavioural segmentation, product mix, Structural Shock sensitivity and trade off recognition
# Financial Risk & Revenue Optimization Engine

### *A Three-Stage Prescriptive Analytics Framework*

## Project Overview

This project addresses the **Revenue-Risk Paradox**: the financial phenomenon where aggressive customer acquisition and gross volume growth lead to non-linear increases in return-related losses. Using a multi-year transactional dataset, I developed a prescriptive engine that transitions from descriptive clustering to risk-adjusted profit maximization.

## The Three-Stage Architecture

### **Stage 1: Behavioral Baseline (Descriptive)**

* **Objective:** Identify core customer archetypes.
* **Methodology:** Engineered RFM (Recency, Frequency, Monetary) features and applied **K-Means Clustering**.
* **Key Metric:** Established the **Purchase Likelihood ($\hat{P}$)** using Min-Max normalization of behavioral density.

### **Stage 2: Validation & Calibration (Diagnostic)**

* **Objective:** Verify model stability across time (2019–2022).
* **Methodology:** "Backtested" Stage 1 clusters against a hold-out test set to measure temporal decay.
* **Discovery:** Identified a "Pareto of Risk" where **12% of users generated 54% of all return costs**, justifying the need for a selective targeting rule.

### **Stage 3: Prescriptive Optimization (Decisioning)**

* **Objective:** Maximize Net Portfolio Value.
* **Methodology:** Deployed a **Multi-Objective Rank & Select Rule**.
* **Logic:** Calculated **Expected Net Value (ENV)** by penalizing Gross Revenue with predicted Return Risk and a **Structural Fragility Penalty** to prevent over-reliance on "Whale" accounts.

---

## Mathematical Framework

### **Expected Net Value (ENV)**

The primary decision metric that accounts for "Revenue Erosion":


$$ENV = (\hat{P}_u \times E(BV_u)) \times (1 - RR_u)$$

### **The Selection Score ($S_u$)**

To ensure the portfolio is robust and diversified, the engine applies a concentration penalty:


$$S_u = ENV_u \times [1 - (\omega \cdot \text{Fragility Penalty})]$$

---

## Key Results

* **14.2% Increase** in projected Net Realization by eliminating high-risk/low-margin targeting.
* **31.2% Reduction** in portfolio risk variance.
* **Anti-Fragile Positioning:** Optimized the **Herfindahl-Hirschman Index (HHI)**, reducing top-tier dependency from 42% to 28%.
* **Efficiency Frontier:** Identified the "Sweet Spot" for targeting at a **25% Reach Cap**.

## Tech Stack

* **Analytics:** Python (Pandas, NumPy, Scikit-Learn)
* **Visualization:** Plotly, Matplotlib, Streamlit
* **Concepts:** K-Means Clustering, Pareto Analysis, Portfolio Theory, Robustness Testing

## Recommended CSV files attached for quick uplook.
## link: https://multicategoryplatform-analysis-dashboard-xafgs3yd3yiojbcfcjgp4.streamlit.app/

---

Are you planning to include the actual Python scripts and notebooks in this repository, or are you keeping it as a high-level documentation of your process?
