import streamlit as st
import plotly.graph_objects as go
from main import FuzzySet
import numpy as np

st.set_page_config(page_title="Fuzzy Toolbox", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    body {background-color: #0e1117; color: #fafafa;}
    </style>
""", unsafe_allow_html=True) 
tabs = st.tabs(["Fuzzy Sets", "Fuzzy Operations", "Fuzzy Implications", "Defuzzification"])

# Store sets
if "sets" not in st.session_state:
    st.session_state["sets"] = {}

# --- Tab 1: Define Fuzzy Sets ---
with tabs[0]:
    st.header("Define Fuzzy Sets")

    name = st.text_input("Set Name")
    universe = np.linspace(0, 10, 200)

    mtype = st.selectbox("Membership Type",
                         ["Triangular", "Trapezoidal", "Gaussian", "Bell", "Sigmoid", "Manual"])

    fs = None  # default

    # ---------------- TRIANGULAR ----------------
    if mtype == "Triangular":
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.slider("a", 0.0, 10.0, 2.0)
        with col2:
            b = st.slider("b", 0.0, 10.0, 5.0)
        with col3:
            c = st.slider("c", 0.0, 10.0, 8.0)

        fs = FuzzySet.triangular(universe, a, b, c)

    # ---------------- TRAPEZOIDAL ----------------
    elif mtype == "Trapezoidal":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            a = st.slider("a", 0.0, 10.0, 2.0)
        with col2:
            b = st.slider("b", 0.0, 10.0, 4.0)
        with col3:
            c = st.slider("c", 0.0, 10.0, 6.0)
        with col4:
            d = st.slider("d", 0.0, 10.0, 8.0)

        fs = FuzzySet.trapezoidal(universe, a, b, c, d)

    # ---------------- GAUSSIAN ----------------
    elif mtype == "Gaussian":
        c, sigma = st.slider("Parameters (c, sigma)", 0.0, 10.0, (5.0, 1.0))
        fs = FuzzySet.gaussian(universe, c, sigma)

    # ---------------- BELL ----------------
    elif mtype == "Bell":
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.slider("a (width)", 0.1, 10.0, 2.0)
        with col2:
            b = st.slider("b (slope)", 0.1, 10.0, 2.0)
        with col3:
            c = st.slider("c (center)", 0.1, 10.0, 5.0)

        fs = FuzzySet.bell(universe, a, b, c)

    # ---------------- SIGMOID ----------------
    elif mtype == "Sigmoid":
        a, c = st.slider("Parameters (a, c)", -10.0, 10.0, (1.0, 5.0))
        fs = FuzzySet.sigmoid(universe, a, c)

    # ---------------- MANUAL ----------------
    elif mtype == "Manual":

        values = st.text_area("Enter membership values (comma-separated)",
                              key="manual_input")

        if values.strip():
            try:
                parts = [v.strip() for v in values.split(",") if v.strip()]
                raw = np.array([float(v) for v in parts])

                # interpolate to 200-point universe
                original_x = np.linspace(0, 10, len(raw))
                resampled = np.interp(universe, original_x, raw)

                fs = FuzzySet.manual(universe, resampled)

            except ValueError:
                st.error("Invalid values! Use only numbers separated by commas.")
                fs = None

    # ------------- ADD SET BUTTON -------------
    if st.button("Add Fuzzy Set"):
        if fs is not None and name.strip():
            st.session_state["sets"][name] = fs
            st.success(f"Added Fuzzy Set: {name}")
        else:
            st.error("Invalid set or name!")

    # ------------- PLOT THE FUZZY SET -------------
    if fs is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fs.universe, y=fs.membership,
                                 mode="lines", name=name))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Define parameters to show the membership function.")
 
