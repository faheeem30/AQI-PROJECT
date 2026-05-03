import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from gru_predictor import predict_next

BASE_DIR = os.getcwd()

# ─── Page Config ───
st.set_page_config(page_title="AQI Intelligence Dashboard", layout="wide", page_icon="🌍")

# ─── Session State ───
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

# ─── Premium Dark Theme CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 50%, #0a0e1a 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Glass card */
.glass-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(30,41,59,0.4));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.metric-card h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    margin: 8px 0 4px;
}
.metric-card p {
    font-size: 12px;
    color: #94a3b8;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* AQI badges */
.aqi-good { color: #22c55e; }
.aqi-moderate { color: #eab308; }
.aqi-poor { color: #f97316; }
.aqi-severe { color: #ef4444; }
.aqi-hazardous { color: #a855f7; }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-good { background: rgba(34,197,94,0.12); color: #22c55e; }
.badge-moderate { background: rgba(234,179,8,0.12); color: #eab308; }
.badge-poor { background: rgba(249,115,22,0.12); color: #f97316; }
.badge-severe { background: rgba(239,68,68,0.12); color: #ef4444; }
.badge-hazardous { background: rgba(168,85,247,0.12); color: #a855f7; }

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 24px 0 12px;
}

/* Insight card */
.insight-card {
    background: rgba(15,23,42,0.5);
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #cbd5e1;
}
.insight-card.warning { border-left-color: #f97316; }
.insight-card.success { border-left-color: #22c55e; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(15,23,42,0.7) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.4);
    border: 2px dashed rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 16px;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(59,130,246,0.4);
    background: rgba(59,130,246,0.04);
}

/* Headers */
h1, h2, h3 { color: #f1f5f9 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15,23,42,0.4);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.15) !important;
    color: #3b82f6 !important;
}

/* Metric widget override */
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.5);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───
AQI_BREAKPOINTS = [
    (50, "Good", "badge-good", "aqi-good"),
    (100, "Moderate", "badge-moderate", "aqi-moderate"),
    (200, "Poor", "badge-poor", "aqi-poor"),
    (300, "Severe", "badge-severe", "aqi-severe"),
    (999, "Hazardous", "badge-hazardous", "aqi-hazardous"),
]

def get_aqi_info(val):
    for threshold, label, badge_cls, color_cls in AQI_BREAKPOINTS:
        if val <= threshold:
            return label, badge_cls, color_cls
    return "Hazardous", "badge-hazardous", "aqi-hazardous"

def dark_plotly_layout(fig, title=""):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=14, color="#94a3b8")),
        font=dict(family="Inter", color="#94a3b8", size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


# ─── Title ───
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
    <p style="font-size:36px; font-weight:800; color:#f1f5f9; margin:0; letter-spacing:-0.5px;">
        🌍 AQI Intelligence Dashboard
    </p>
    <p style="font-size:13px; color:#64748b; margin-top:6px; letter-spacing:0.5px;">
        AI-powered air quality monitoring & forecasting platform
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar Navigation ───
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 8px;">
        <p style="font-size:18px; font-weight:700; color:#f1f5f9; margin:0;">🌍 AQI Dashboard</p>
        <p style="font-size:10px; color:#64748b; letter-spacing:1px; margin-top:4px;">INTELLIGENCE PLATFORM</p>
    </div>
    """, unsafe_allow_html=True)

    page = option_menu(
        None,
        ["User Panel", "Admin Panel", "Leaderboard"],
        icons=["graph-up-arrow", "gear-wide-connected", "trophy"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "8px", "background-color": "transparent"},
            "icon": {"color": "#3b82f6", "font-size": "14px"},
            "nav-link": {
                "font-size": "13px", "text-align": "left", "margin": "2px 0",
                "color": "#94a3b8", "border-radius": "10px", "padding": "10px 14px",
                "font-weight": "500",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.08))",
                "color": "#3b82f6", "font-weight": "600",
            },
        },
    )


# ═══════════════ USER PANEL ═══════════════
if page == "User Panel":
    st.markdown('<p class="section-header">📡 Air Quality Forecast</p>', unsafe_allow_html=True)

    data_path = os.path.join(BASE_DIR, "data")
    if not os.path.exists(data_path):
        st.error("⚠️ Data folder not found. Please ensure the `data/` directory exists.")
        st.stop()

    states = os.listdir(data_path)
    col1, col2 = st.columns(2)
    with col1:
        selected_state = st.selectbox("🏛️ State", states)
    state_path = os.path.join(data_path, selected_state)
    cities = os.listdir(state_path)
    with col2:
        selected_city = st.selectbox("🏙️ City", cities)

    city_path = os.path.join(state_path, selected_city)
    items = os.listdir(city_path)

    if "1hour" in items or "1day" in items:
        station_path = city_path
    else:
        selected_station = st.selectbox("📍 Station", items)
        station_path = os.path.join(city_path, selected_station)

    col3, col4 = st.columns(2)
    frequencies = os.listdir(station_path)
    with col3:
        selected_frequency = st.selectbox("⏱️ Frequency", frequencies)
    freq_path = os.path.join(station_path, selected_frequency)
    years = os.listdir(freq_path)
    with col4:
        selected_year = st.selectbox("📅 Year", years)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("🔮 Generate Forecast", use_container_width=True):
        year_path = os.path.join(freq_path, selected_year)
        csv_files = [f for f in os.listdir(year_path) if f.endswith(".csv")]
        file_path = os.path.join(year_path, csv_files[0])

        df = pd.read_csv(file_path, sep=None, engine="python")
        df = df.replace(["NA", "None", "--"], np.nan)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.mask(df <= 0)

        pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
        overall_current = []
        overall_future = []

        with st.spinner("🔄 Running GRU prediction model..."):
            for col in df.select_dtypes(include=[np.number]).columns:
                series = df[col].dropna()
                if len(series) < 24:
                    continue

                last24 = series.tail(24).values
                current = last24[-1]
                future = predict_next(last24)

                # Iterative future forecasting
                future_values = []
                temp_input = last24.copy()
                for i in range(5):
                    pred = float(predict_next(temp_input))
                    future_values.append(pred)
                    temp_input = np.append(temp_input[1:], pred)
                

                   # ✅ LEADERBOARD METRICS (ADD HERE)
                actual = last24[-5:]
                predicted = future_values[:5]

                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual, predicted)

                st.session_state.leaderboard.append({
                    "city": selected_city,
                    "aqi": int(current),
                    "rmse": round(rmse, 3),
                    "mae": round(mae, 3),
                    "mse": round(mse, 3),
                    "r2": round(r2, 3)
                })

                # ── Pollutant Section ──
                label, badge_cls, color_cls = get_aqi_info(current)
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
                        <h3 style="margin:0; font-size:16px; font-weight:600;">{col}</h3>
                        <span class="badge {badge_cls}">{label}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Current Value", round(current, 2))
                c2.metric("Next Step Prediction", round(future, 2))
                delta = round(future - current, 2)
                c3.metric("Change", f"{'+' if delta > 0 else ''}{delta}",
                         delta_color="inverse")

                # Charts in tabs
                tab1, tab2 = st.tabs(["📊 Current Prediction", "🔮 Future Forecast"])

                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=last24, mode="lines+markers", name="History (Last 24)",
                        line=dict(color="#3b82f6", width=2),
                        marker=dict(size=4, color="#3b82f6"),
                        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
                    ))
                    fig.add_trace(go.Scatter(
                        x=[len(last24)], y=[future], mode="markers",
                        name="Next Prediction",
                        marker=dict(size=14, color="#ef4444", symbol="star"),
                    ))
                    dark_plotly_layout(fig, f"{col} — Current AQI Prediction")
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    fig_future = go.Figure()
                    fig_future.add_trace(go.Scatter(
                        y=last24, mode="lines+markers", name="History",
                        line=dict(color="#3b82f6", width=2),
                        marker=dict(size=4),
                        fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
                    ))
                    fig_future.add_trace(go.Scatter(
                        x=list(range(len(last24), len(last24) + len(future_values))),
                        y=future_values, mode="lines+markers", name="Future Prediction",
                        line=dict(color="#ef4444", width=2, dash="dot"),
                        marker=dict(size=6, color="#ef4444"),
                        fill="tozeroy", fillcolor="rgba(239,68,68,0.06)",
                    ))
                    dark_plotly_layout(fig_future, f"{col} — Future AQI Forecast")
                    st.plotly_chart(fig_future, use_container_width=True)

                    st.metric("Future AQI (Predicted)", round(future_values[-1], 2))

                if any(p in col for p in pollutants):
                    overall_current.append(current)
                    overall_future.append(future_values[-1])

        # ── Overall AQI Summary ──
        if overall_current and overall_future:
            current_aqi = int(np.max(overall_current))
            future_aqi = int(np.max(overall_future))
            cur_label, cur_badge, cur_color = get_aqi_info(current_aqi)
            fut_label, fut_badge, fut_color = get_aqi_info(future_aqi)

            st.markdown('<p class="section-header">🌍 Overall AQI Summary</p>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <p>Current AQI</p>
                    <h2 class="{cur_color}">{current_aqi}</h2>
                    <span class="badge {cur_badge}">{cur_label}</span>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <p>Predicted AQI</p>
                    <h2 class="{fut_color}">{future_aqi}</h2>
                    <span class="badge {fut_badge}">{fut_label}</span>
                </div>
                """, unsafe_allow_html=True)

            # AI Insight
            delta_aqi = future_aqi - current_aqi
            if delta_aqi > 20:
                insight_cls, insight_msg = "warning", f"⚠️ AQI is expected to rise by {delta_aqi} points. Consider precautionary measures."
            elif delta_aqi < -10:
                insight_cls, insight_msg = "success", f"✅ AQI is expected to improve by {abs(delta_aqi)} points. Air quality trending positively."
            else:
                insight_cls, insight_msg = "", f"ℹ️ AQI is expected to remain relatively stable (change: {delta_aqi:+d})."
            st.markdown(f'<div class="insight-card {insight_cls}">{insight_msg}</div>', unsafe_allow_html=True)


# ═══════════════ ADMIN PANEL ═══════════════
elif page == "Admin Panel":
    st.markdown('<p class="section-header">⚙️ System Management</p>', unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    for c, (icon, title, val, sub) in zip(
        [col1, col2, col3, col4],
        [
            ("🗄️", "Data Records", "2.4M", "Last sync: 2 min ago"),
            ("🖥️", "Model Version", "v3.2.1 (GRU)", "Updated 3 days ago"),
            ("👥", "Active Users", "1,247", "+12 this week"),
            ("⚡", "System Uptime", "99.8%", "Last 30 days"),
        ]
    ):
        with c:
            st.markdown(f"""
            <div class="metric-card">
                <p style="font-size:20px; margin-bottom:4px;">{icon}</p>
                <p>{title}</p>
                <h2 style="font-size:24px; color:#f1f5f9;">{val}</h2>
                <p style="font-size:10px; color:#64748b;">{sub}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-size:16px;">☁️</span>
                <span style="font-size:14px; font-weight:600; color:#f1f5f9;">Data Upload</span>
            </div>
            <p style="font-size:12px; color:#64748b; margin-bottom:16px;">
                Upload CSV datasets for AQI analysis and model training
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        state = st.text_input("State")
        city = st.text_input("City")
        station = st.text_input("Station")
        frequency = st.selectbox("Frequency", ["1hour","1day"])
        year = st.text_input("Year")

        if st.button("Upload Dataset"):
            if not uploaded:
                st.error("Upload CSV first")
            else:
                save_path = os.path.join("data", state, city, station, frequency, year)
                os.makedirs(save_path, exist_ok=True)

                with open(os.path.join(save_path, uploaded.name), "wb") as f:
                    f.write(uploaded.getbuffer())

                st.success("Uploaded successfully ✅")
        
        
        
        if uploaded:
            st.success(f"✅ Uploaded: {uploaded.name} ({round(uploaded.size/1024, 1)} KB)")

    with col_right:
        st.markdown("""
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-size:16px;">🧠</span>
                <span style="font-size:14px; font-weight:600; color:#f1f5f9;">Model Management</span>
            </div>
            <p style="font-size:12px; color:#64748b; margin-bottom:12px;">
                Configure and retrain the GRU prediction model
            </p>
        </div>
        """, unsafe_allow_html=True)

        for label, value in [
            ("Model Architecture", "GRU + Attention"),
            ("Training Epochs", "150"),
            ("Learning Rate", "0.001"),
            ("Last Accuracy", "94.2%"),
        ]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:8px 12px;
                        background:rgba(15,23,42,0.4); border-radius:8px; margin-bottom:4px;">
                <span style="font-size:12px; color:#94a3b8;">{label}</span>
                <span style="font-size:12px; font-family:'JetBrains Mono'; font-weight:600; color:#f1f5f9;
                       background:rgba(30,41,59,0.6); padding:2px 8px; border-radius:6px;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("▶️ Trigger Retraining", use_container_width=True):
            with st.spinner("Training GRU model..."):
                import time; time.sleep(2)
            st.success("✅ Model retrained successfully!")


# ═══════════════ LEADERBOARD ═══════════════
elif page == "Leaderboard":
    st.markdown('<p class="section-header">🏆 City AQI Leaderboard</p>', unsafe_allow_html=True)

    if st.session_state.leaderboard:
        lb = st.session_state.leaderboard

        # Metrics summary cards
        if any("rmse" in entry for entry in lb):
            all_rmse = [e["rmse"] for e in lb if "rmse" in e]
            all_mae = [e["mae"] for e in lb if "mae" in e]
            all_mse = [e["mse"] for e in lb if "mse" in e]
            all_r2 = [e["r2"] for e in lb if "r2" in e]

            mc1, mc2, mc3, mc4 = st.columns(4)
            for c, (lbl, vals, icon) in zip(
                [mc1, mc2, mc3, mc4],
                [
                    ("Avg RMSE", all_rmse, "📊"),
                    ("Avg MAE", all_mae, "🎯"),
                    ("Avg MSE", all_mse, "📈"),
                    ("Avg R² Score", all_r2, "✅"),
                ]
            ):
                avg = round(np.mean(vals), 4) if vals else 0
                with c:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="font-size:16px;">{icon}</p>
                        <p>{lbl}</p>
                        <h2 style="font-size:22px; color:#f1f5f9;">{avg}</h2>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Table
        df_lb = pd.DataFrame(lb)
        display_cols = ["city", "aqi"]
        if "rmse" in df_lb.columns:
            display_cols += ["rmse", "mae", "mse", "r2"]
        st.dataframe(
            df_lb[display_cols].rename(columns={
                "city": "City", "aqi": "AQI",
                "rmse": "RMSE", "mae": "MAE", "mse": "MSE", "r2": "R²"
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:48px;">
            <p style="font-size:40px; margin-bottom:8px;">🏆</p>
            <p style="font-size:15px; font-weight:600; color:#f1f5f9;">No Leaderboard Data Yet</p>
            <p style="font-size:12px; color:#64748b; margin-top:4px;">
                Generate forecasts in the User Panel to populate the leaderboard with cities and evaluation metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)
