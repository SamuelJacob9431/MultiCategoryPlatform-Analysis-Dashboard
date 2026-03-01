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
def optimize_targeting_strategy(user_financials, rfm_data, targeting_cap, risk_tolerance):
    # Merge financial metrics with RFM segments
    analysis_df = user_financials.copy()
    analysis_df['Monetary'] = rfm_data['Monetary']
    
    # 1. SCORING: Expected Net Value (ENV)
    # Formula: (Likelihood * Basket Value) - (Risk * Basket Value)
    analysis_df['Expected_Net_Value'] = (
        (analysis_df['Purchase Likelihood (%)'] / 100) * analysis_df['Expected Basket Value (₹)']
    ) * (1 - analysis_df['Return Risk Rate'])
    
    # 2. RANKING: Sort by Efficiency
    analysis_df = analysis_df.sort_values("Expected_Net_Value", ascending=False)
    
    # 3. QUANTIFY: Apply Constraints
    cutoff = int(len(analysis_df) * (targeting_cap / 100))
    analysis_df['Targeted'] = False
    # Only target users below the risk tolerance threshold
    eligible_mask = analysis_df['Return Risk Rate'] <= (risk_tolerance / 100)
    
    # Final selection based on rank + eligibility
    targeted_indices = analysis_df[eligible_mask].head(cutoff).index
    analysis_df.loc[targeted_indices, 'Targeted'] = True
    
    return analysis_df

# --- 2. Sidebar Navigation ---
st.sidebar.title("📌 Navigation")
nav = st.sidebar.selectbox("Select View:", ["Business Insights", "Technical Model Performance", "Strategy Optimizer"])
uploaded_file = st.sidebar.file_uploader("Upload Transactions CSV", type="csv")

# --- 3. Main Logic Execution ---
if uploaded_file:
    # GLOBAL CALCULATIONS (Available to all tabs)
    df = load_and_clean(uploaded_file)
    rfm_data = perform_rfm_clustering(df)
    mix_df = get_product_mix(df)
    affinity_df = get_basket_affinity(df)
    
    # Generate financial metrics
    user_financials = get_financial_metrics_per_user(df, rfm_data)
    
    # 🔥 CORE FIX: Calculate 'ENV' here so it exists for ALL subsequent logic
    user_financials['ENV'] = (
        (user_financials['Purchase Likelihood (%)'] / 100) * user_financials['Expected Basket Value (₹)']
    ) * (1 - user_financials['Return Risk Rate'])
    
    # Common stats for the whole dashboard
    purchase_df = df[df["EventType"] == "purchased"]
    return_df = df[df["EventType"] == "returned"]
    e_net = purchase_df["OrderValue"].sum() - return_df["OrderValue"].sum()
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

    elif nav == "Technical Model Performance":
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
    elif nav == "Strategy Optimizer":
        st.title("🎯 Prescriptive Strategy & Stress Testing")
        
        # --- SIDEBAR CONTROLS ---
        st.sidebar.divider()
        st.sidebar.header("Optimization Constraints")
        reach = st.sidebar.slider("Targeting Cap (%)", 5, 100, 25)
        risk_weight = st.sidebar.slider("Risk Aversion Weight", 0.0, 2.0, 1.0)
        conc_penalty = st.sidebar.checkbox("Control Concentration Amplification", value=True)

        # --- B. RANK AND SELECT LOGIC ---
        df_opt = user_financials.copy()
        env_mean = df_opt['ENV'].mean()
        
        # Multi-Objective Scoring: Revenue minus Risk Penalty
        df_opt['Selection_Score'] = df_opt['ENV'] - (risk_weight * df_opt['Return Risk Rate'] * env_mean)
        
        # Concentration Control (Penalty for outliers to prevent fragility)
        if conc_penalty:
            limit = env_mean + (3 * df_opt['ENV'].std())
            df_opt['Selection_Score'] = np.where(df_opt['ENV'] > limit, df_opt['Selection_Score'] * 0.8, df_opt['Selection_Score'])

        # Risk-Adjusted Ranking
        df_opt = df_opt.sort_values("Selection_Score", ascending=False)
        
        # Threshold Optimization: Apply Targeting Cap
        cutoff_idx = int(len(df_opt) * (reach / 100))
        df_opt['Targeted'] = False
        df_opt.iloc[:cutoff_idx, df_opt.columns.get_loc('Targeted')] = True
        
        targeted_only = df_opt[df_opt['Targeted']]

        # --- UI LAYOUT ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Projected Net Revenue", f"₹{targeted_only['ENV'].sum():,.0f}")
        col2.metric("Avg. Selection Risk", f"{targeted_only['Return Risk Rate'].mean():.1%}")
        col3.metric("Selected User Count", len(targeted_only))

        tab_frontier, tab_tradeoff = st.tabs(["Efficiency Frontier", "⚖️ Trade-off Comparison"])

        with tab_frontier:
            st.subheader("The Efficiency Curve")
            
            df_opt['Cum_Net'] = df_opt['ENV'].cumsum()
            fig_curve = px.line(df_opt.reset_index(), x=np.arange(len(df_opt)), y='Cum_Net',
                               title="Cumulative Expected Net Value (ENV) Ranking",
                               labels={'x': 'User Rank (Most Efficient to Least)', 'y': 'Total Portfolio Value'})
            fig_curve.add_vline(x=cutoff_idx, line_dash="dash", line_color="red", annotation_text="Your Cap")
            st.plotly_chart(fig_curve, use_container_width=True)

        with tab_tradeoff:
            st.subheader("Strategy Trade-off Comparison")
            
            # Comparison Benchmarks
            bench_rev = df_opt.sort_values("ENV", ascending=False).head(cutoff_idx)
            bench_safe = df_opt.sort_values("Return Risk Rate", ascending=True).head(cutoff_idx)
            
            tradeoff_data = {
                "Metric": ["Net Revenue", "Avg Return Risk", "Top 1% Dependency"],
                "Your Optimized": [
                    f"₹{targeted_only['ENV'].sum():,.0f}", 
                    f"{targeted_only['Return Risk Rate'].mean():.1%}",
                    f"{(targeted_only['ENV'].head(int(len(targeted_only)*0.1)).sum() / targeted_only['ENV'].sum() if targeted_only['ENV'].sum() > 0 else 0):.1%}"
                ],
                "Revenue Focused": [
                    f"₹{bench_rev['ENV'].sum():,.0f}", 
                    f"{bench_rev['Return Risk Rate'].mean():.1%}",
                    f"{(bench_rev['ENV'].head(int(len(bench_rev)*0.1)).sum() / bench_rev['ENV'].sum()):.1%}"
                ],
                "Risk Averse": [
                    f"₹{bench_safe['ENV'].sum():,.0f}", 
                    f"{bench_safe['Return Risk Rate'].mean():.1%}",
                    f"{(bench_safe['ENV'].head(int(len(bench_safe)*0.1)).sum() / bench_safe['ENV'].sum()):.1%}"
                ]
            }
            st.table(pd.DataFrame(tradeoff_data).set_index("Metric"))
            st.info("The 'Top 1% Dependency' indicates if your strategy is over-reliant on a few high-value customers.")
            # --- NEW RADAR CHART VISUALIZATION ---
            st.write("---")
            st.subheader("Strategy Footprint Analysis")

            # Prepare data for Radar Chart
            # We normalize the values so they can be compared on the same scale (0 to 1)
            categories = ['Net Revenue', 'Risk Control', 'Robustness (Low Dependency)']
            
            def get_radar_metrics(target_df, total_df):
                rev = target_df['ENV'].sum() / total_df['ENV'].sum()
                risk = 1 - target_df['Return Risk Rate'].mean() # Invert so higher is "better"
                dep = 1 - (target_df['ENV'].head(int(len(target_df)*0.1)).sum() / target_df['ENV'].sum())
                return [rev, risk, dep]

            fig_radar = go.Figure()

            # Add "Your Optimized" Trace
            fig_radar.add_trace(go.Scatterpolar(
                r=get_radar_metrics(targeted_only, df_opt),
                theta=categories,
                fill='toself',
                name='Your Optimized',
                line_color='#00CC96'
            ))

            # Add "Revenue Focused" Trace
            fig_radar.add_trace(go.Scatterpolar(
                r=get_radar_metrics(bench_rev, df_opt),
                theta=categories,
                fill='toself',
                name='Revenue Focused',
                line_color='#EF553B'
            ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Performance Dimensions: Optimized vs. Aggressive"
            )

            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.caption("""
            **Interpretation:** A larger area represents a more 'all-around' strategy. 
            Notice how the **Revenue Focused** strategy might reach further in 'Net Revenue' 
            but shrinks significantly in 'Risk Control'.
            """)

else:
    st.info("Please upload a CSV file to begin.")
            
                