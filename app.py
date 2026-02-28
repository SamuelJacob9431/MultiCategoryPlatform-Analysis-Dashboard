import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from collections import Counter

# --- Configuration ---
EXCHANGE_RATE = 83 
st.set_page_config(page_title="Customer & Technical Analytics", layout="wide")

# --- 1. Data Processing Functions ---
@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)
    df["EventType"] = df["EventType"].astype(str).str.strip().str.lower()
    df["EventDateTime"] = pd.to_datetime(df["EventDateTime"])
    df["OrderValue"] = (df["UnitPrice"] * df["Quantity"]) * EXCHANGE_RATE
    return df

def perform_rfm_clustering(df):
    purchase_df = df[df["EventType"] == "purchased"]
    today = purchase_df["EventDateTime"].max() + timedelta(days=1)
    rfm = purchase_df.groupby("UserID").agg(
        Recency=("EventDateTime", lambda x: (today - x.max()).days),
        Frequency=("EventID", "nunique"),
        Monetary=("OrderValue", "sum")
    )
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    cp = rfm.groupby("Cluster").mean().sort_values("Recency")
    rfm["UserType"] = "Moderate"
    rfm.loc[rfm["Cluster"] == cp.index[0], "UserType"] = "Active"
    rfm.loc[rfm["Cluster"] == cp.index[-1], "UserType"] = "Dormant"
    return rfm

def get_product_mix(df):
    purchase_df = df[df["EventType"] == "purchased"]
    prod_rev = purchase_df.groupby("ProductName")["OrderValue"].sum()
    prod_qty = purchase_df.groupby("ProductName")["Quantity"].sum()
    basket = purchase_df.groupby("EventID")["OrderValue"].sum()
    temp_df = purchase_df.merge(basket.rename("BasketValue"), on="EventID")
    prod_basket = temp_df.groupby("ProductName")["BasketValue"].mean()
    
    mix_df = pd.DataFrame({"Revenue": prod_rev, "Quantity": prod_qty, "AvgBasket": prod_basket}).dropna()
    mix_norm = (mix_df - mix_df.mean()) / mix_df.std()
    mix_df["Type"] = "Standard"
    mix_df.loc[(mix_norm["Revenue"] > 1) & (mix_norm["AvgBasket"] > 1), "Type"] = "Premium Driver"
    mix_df.loc[(mix_norm["Quantity"] > 1) & (mix_norm["Revenue"] > 1), "Type"] = "Volume Driver"
    mix_df.loc[(mix_norm["Quantity"] > 1) & (mix_norm["AvgBasket"] < -0.5), "Type"] = "Traffic Driver"
    return mix_df

def get_basket_affinity(df):
    purchase_df = df[df["EventType"] == "purchased"]
    basket_products = purchase_df.groupby("EventID")["ProductName"].apply(list)
    pairs = Counter()
    for items in basket_products:
        for pair in combinations(sorted(set(items)), 2):
            pairs[pair] += 1
    top_pairs = pd.DataFrame(pairs.most_common(10), columns=["Pair", "Frequency"])
    top_pairs["Pair"] = top_pairs["Pair"].astype(str)
    return top_pairs

def get_financial_metrics_per_user(df, rfm_df):
    purchase_df = df[df["EventType"] == "purchased"]
    basket_val = purchase_df.groupby(["UserID", "EventID"])["OrderValue"].sum().groupby("UserID").mean()
    return_counts = df.groupby("UserID")["EventType"].apply(lambda x: (x == "returned").mean())
    
    rfm_numeric = rfm_df[["Recency", "Frequency", "Monetary"]]
    rfm_norm = (rfm_numeric - rfm_numeric.min()) / (rfm_numeric.max() - rfm_numeric.min())
    likelihood_score = ((rfm_norm["Frequency"] + rfm_norm["Monetary"] + (1 - rfm_norm["Recency"])) / 3) * 100
    
    user_metrics = pd.DataFrame({
        "User Type": rfm_df["UserType"],
        "Purchase Likelihood (%)": likelihood_score,
        "Expected Basket Value (₹)": basket_val,
        "Return Risk Rate": return_counts
    }).fillna(0)
    return user_metrics

# --- 2. Sidebar Navigation ---
st.sidebar.title("📌 Navigation")
nav = st.sidebar.selectbox("Select View:", ["Business Insights", "Technical Model Performance"])
uploaded_file = st.sidebar.file_uploader("Upload Transactions CSV", type="csv")

# --- 3. Main Logic Execution ---
if uploaded_file:
    # GLOBAL CALCULATIONS (Available to all tabs)
    df = load_and_clean(uploaded_file)
    rfm_data = perform_rfm_clustering(df)
    mix_df = get_product_mix(df)
    affinity_df = get_basket_affinity(df)
    user_financials = get_financial_metrics_per_user(df, rfm_data)
    
    purchase_df = df[df["EventType"] == "purchased"]
    return_df = df[df["EventType"] == "returned"]
    
    # Common stats
    e_gross = purchase_df["OrderValue"].sum()
    e_return = return_df["OrderValue"].sum()
    e_net = e_gross - e_return
    erpu = e_net / df["UserID"].nunique()
    high_risk_count = (user_financials["Return Risk Rate"] > 0.3).sum()

    if nav == "Business Insights":
        st.title("📊 Business Insights Dashboard")
        t1, t2, t3, t4 = st.tabs(["Segments", "Product Mix", "Market Basket", "Financial Risk"])

        with t1:
            st.subheader("Customer Segments")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Users", len(rfm_data))
            c2.metric("Repeat Prob.", f"{(rfm_data['Frequency']>1).mean():.1%}")
            c3.metric("ERPU", f"₹{erpu:,.2f}")
            fig_rfm = px.scatter(rfm_data, x="Recency", y="Monetary", color="UserType", size="Frequency")
            st.plotly_chart(fig_rfm, use_container_width=True)

        with t2:
            st.subheader("Product Strategy")
            fig_matrix = px.scatter(mix_df, x="Quantity", y="Revenue", color="Type", size="AvgBasket", hover_name=mix_df.index)
            st.plotly_chart(fig_matrix, use_container_width=True)

        with t3:
            st.subheader("Market Basket")
            fig_affinity = px.bar(affinity_df, x="Frequency", y="Pair", orientation='h', color="Frequency")
            st.plotly_chart(fig_affinity, use_container_width=True)

        with t4:
            st.subheader("Financial Risk")
            st.metric("High Return-Risk Users", high_risk_count)
            st.dataframe(user_financials.style.format({"Purchase Likelihood (%)": "{:.1f}%", "Return Risk Rate": "{:.1%}"}))

    else:
        st.title("🧪 Technical Model Performance (Stage 1 Baseline)")
        
        # Defining Tabs based on Section A, B, C, D requirements
        tab_a, tab_b, tab_c, tab_d = st.tabs([
            "A. Purchase Likelihood", 
            "B. Basket Modeling", 
            "C. Net Revenue (ERPU)", 
            "D. Behavioral Risk"
        ])

        # --- Section A: Purchase Likelihood Modeling ---
        with tab_a:
            st.header("Purchase Likelihood & Engagement")
            col1, col2 = st.columns(2)
            
            # Estimate probability of repeat purchase
            repeat_prob = (rfm_data["Frequency"] > 1).mean()
            col1.metric("Repeat Purchase Probability", f"{repeat_prob:.2%}")
            
            # Model transaction frequency per user
            fig_freq = px.histogram(rfm_data, x="Frequency", nbins=50, 
                                   title="Transaction Frequency Distribution",
                                   color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig_freq, use_container_width=True)
            
            # Detect dormant vs active clusters
            st.subheader("High-Engagement Segment Analysis")
            cluster_stats = rfm_data.groupby("UserType").agg({
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean",
                "Cluster": "count"
            }).rename(columns={"Cluster": "User Count"})
            st.dataframe(cluster_stats.style.background_gradient(cmap="YlGnBu"))

        # --- Section B: Basket Size Modeling ---
        with tab_b:
            st.header("Basket Size & Revenue Distribution")
            
            # Estimate expected basket value per transaction
            basket_per_txn = purchase_df.groupby("EventID")["OrderValue"].sum()
            e_gross_txn = basket_per_txn.mean()
            
            m1, m2 = st.columns(2)
            m1.metric("E(Gross Value) per Txn", f"₹{e_gross_txn:,.2f}")
            m2.metric("Median Basket Value", f"₹{basket_per_txn.median():,.2f}")
            
            # Detect heavy-tail revenue effects (Pareto logic)
            st.subheader("SKU/Category Contribution & Heavy-Tail Analysis")
            sku_rev = purchase_df.groupby("ProductName")["OrderValue"].sum().sort_values(ascending=False)
            sku_cum = sku_rev.cumsum() / sku_rev.sum()
            
            fig_heavy = px.area(x=np.arange(len(sku_cum)), y=sku_cum, 
                               title="Revenue Concentration (Heavy-Tail Effect)",
                               labels={'x': 'Number of SKUs', 'y': 'Cumulative Revenue %'})
            fig_heavy.add_hline(y=0.8, line_dash="dash", annotation_text="80% Revenue Mark")
            st.plotly_chart(fig_heavy, use_container_width=True)

        # --- Section C: Net Revenue Estimation ---
        with tab_c:
            st.header("Net Revenue Baseline Estimation")
            st.info("Formula: E(Net Revenue) = E(Gross Value) - E(Return Value)")
            
            # Define: E(Gross) and E(Return)
            e_gross = purchase_df["OrderValue"].sum()
            e_return = return_df["OrderValue"].sum()
            e_net = e_gross - e_return
            
            # Compute: Expected Revenue Per User (ERPU)
            total_users = df["UserID"].nunique()
            erpu = e_net / total_users
            
            c1, c2, c3 = st.columns(3)
            c1.metric("E(Gross Value)", f"₹{e_gross:,.0f}")
            c2.metric("E(Return Value)", f"₹{e_return:,.0f}", delta_color="inverse")
            c3.metric("Net Baseline", f"₹{e_net:,.0f}")
            
            st.subheader("Expected Revenue Per User (ERPU)")
            st.metric("Global ERPU", f"₹{erpu:,.2f}", help="Total Net Revenue / Total Unique Users")
            
            # Waterfall chart for visual Net breakdown
            fig_water = go.Figure(go.Waterfall(
                orientation = "v",
                measure = ["relative", "relative", "total"],
                x = ["Gross Revenue", "Returns", "Net Revenue"],
                y = [e_gross, -e_return, e_net],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            st.plotly_chart(fig_water, use_container_width=True)

        # --- Section D: Behavioral Segmentation ---
        with tab_d:
            st.header("Behavioral Risk & Exposure Analysis")
            
            # 1. Revenue concentration (Top 10% share)
            user_rev = purchase_df.groupby("UserID")["OrderValue"].sum().sort_values(ascending=False)
            top_10_users = int(len(user_rev) * 0.1)
            concentration = user_rev.head(top_10_users).sum() / user_rev.sum()
            
            # 2. Return-prone user clusters
            return_rate = df.groupby("UserID")["EventType"].apply(lambda x: (x == "returned").mean())
            high_risk_users = (return_rate > 0.3).sum()
            
            # 3. SKU Dependency Exposure
            top_sku_share = (sku_rev.head(5).sum() / sku_rev.sum())
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Top 10% Revenue Share", f"{concentration:.2%}")
            k2.metric("Return-Prone Users", high_risk_users, delta=">30% Return Rate")
            k3.metric("Top 5 SKU Dependency", f"{top_sku_share:.2%}")
            
            st.subheader("Category Contribution Risk")
            # Assuming ProductName contains categories or specific SKUs
            sku_share_df = (sku_rev / sku_rev.sum()).head(15).reset_index()
            sku_share_df.columns = ["Product", "Revenue Share"]
            fig_risk = px.bar(sku_share_df, x="Revenue Share", y="Product", orientation='h', 
                             title="Top 15 SKU Contribution Risk", color="Revenue Share")
            st.plotly_chart(fig_risk, use_container_width=True)

            
else:
    st.info("Please upload a CSV file to begin.")