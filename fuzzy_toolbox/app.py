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

# ----------------- TABS -----------------
tabs = st.tabs(["Universe", "Fuzzy Sets", "Fuzzy Operations", "Fuzzy Implications", "Defuzzification"])

# Store sets
if "sets" not in st.session_state:
    st.session_state["sets"] = {}

 # TAB 1: DEFINE UNIVERSE
 
with tabs[0]:
    st.header("Define Universe of Discourse")

    universe_name = st.text_input(
        "Enter Universe Variable (e.g., Temperature, Speed, Pressure)",
        value=st.session_state.get("universe_name", "Temperature"),
        key="universe_name"
    )

    col1, col2 = st.columns(2)
    with col1:
        umin = st.number_input("Minimum Value", value=st.session_state.get("umin", 0.0))
    with col2:
        umax = st.number_input("Maximum Value", value=st.session_state.get("umax", 10.0))

    points = st.slider("Number of Points (Resolution)", 50, 1000, st.session_state.get("points", 200))

    universe = np.linspace(umin, umax, points)

    # Save universe
    st.session_state["universe"] = universe
    st.session_state["umin"] = umin
    st.session_state["umax"] = umax
    st.session_state["points"] = points

    st.success(f"Universe '{universe_name}' created from {umin} to {umax} with {points} points.")



 # TAB 2: FUZZY SETS
#  
with tabs[1]:
    st.header("Define Fuzzy Sets")

    if "universe" not in st.session_state:
        st.error("Please define the universe first in Tab 1.")
        st.stop()

    universe = st.session_state["universe"]
    umin, umax = float(universe.min()), float(universe.max())

    name = st.text_input("Set Name", key="set_name")

    mtype = st.selectbox("Membership Type",
                         ["Triangular", "Trapezoidal", "Gaussian", "Bell", "Sigmoid", "Manual"],
                         key="mtype")

    fs = None  # default


    # ---------------- TRIANGULAR ----------------
    if mtype == "Triangular":
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.slider("a", umin, umax, umin + (umax-umin)*0.2)
        with col2:
            b = st.slider("b", umin, umax, umin + (umax-umin)*0.5)
        with col3:
            c = st.slider("c", umin, umax, umin + (umax-umin)*0.8)

        fs = FuzzySet.triangular(universe, a, b, c)


    # ---------------- TRAPEZOIDAL ----------------
    elif mtype == "Trapezoidal":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            a = st.slider("a", umin, umax, umin + (umax-umin)*0.1)
        with col2:
            b = st.slider("b", umin, umax, umin + (umax-umin)*0.3)
        with col3:
            c = st.slider("c", umin, umax, umin + (umax-umin)*0.6)
        with col4:
            d = st.slider("d", umin, umax, umin + (umax-umin)*0.9)

        fs = FuzzySet.trapezoidal(universe, a, b, c, d)


    # ---------------- GAUSSIAN ----------------
    elif mtype == "Gaussian":
        center, sigma = st.slider(
            "Parameters (center c, sigma)",
            umin, umax, ( (umin+umax)/2, (umax-umin)/10 )
        )
        fs = FuzzySet.gaussian(universe, center, sigma)


    # ---------------- BELL ----------------
    elif mtype == "Bell":
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.slider("a (width)", 0.1, (umax-umin), (umax-umin)/5)
        with col2:
            b = st.slider("b (slope)", 0.1, 10.0, 2.0)
        with col3:
            c = st.slider("c (center)", umin, umax, (umin+umax)/2)

        fs = FuzzySet.bell(universe, a, b, c)


    # ---------------- SIGMOID ----------------
    elif mtype == "Sigmoid":
        a = st.slider("a (slope)", -10.0, 10.0, 1.0)
        c = st.slider("c (center)", umin, umax, (umin+umax)/2)
        fs = FuzzySet.sigmoid(universe, a, c)


    # ---------------- MANUAL ----------------
    elif mtype == "Manual":
        values = st.text_area("Enter membership values (comma-separated)",
                              key="manual_values")

        if values.strip():
            try:
                parts = [v.strip() for v in values.split(",") if v.strip()]
                raw = np.array([float(v) for v in parts])

                original_x = np.linspace(umin, umax, len(raw))
                resampled = np.interp(universe, original_x, raw)

                fs = FuzzySet.manual(universe, resampled)

            except:
                st.error("Invalid values! Use only numbers separated by commas.")
                fs = None


    # ---------- ADD SET ----------
    if st.button("Add Fuzzy Set"):
        if fs is not None and name.strip():
            st.session_state["sets"][name] = fs
            st.success(f"Added Fuzzy Set: {name}")
        else:
            st.error("Invalid set or empty name.")


    # ---------- PLOT ----------
    if fs is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fs.universe, y=fs.membership,
                                 mode="lines", name=name))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Define parameters to show the membership function.")
