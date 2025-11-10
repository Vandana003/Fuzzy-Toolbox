# streamlit_toolbox.py

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
    mtype = st.selectbox("Membership Type", ["Triangular", "Trapezoidal", "Gaussian", "Bell", "Sigmoid", "Manual"])

    if mtype == "Triangular":
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.slider("a", 0.0, 10.0, 2.0)
        with col2:
            b = st.slider("b", 0.0, 10.0, 5.0)
        with col3:
            c = st.slider("c", 0.0, 10.0, 8.0)

        fs = FuzzySet.triangular(universe,a,b,c)
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
        
        fs = FuzzySet.trapezoidal(universe,a,b,c,d)
    elif mtype == "Gaussian":
        c,sigma = st.slider("Parameters (c,sigma)", 0.0,10.0,(5.0,1.0))
        fs = FuzzySet.gaussian(universe,c,sigma)
    elif mtype == "Bell":
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.slider("a (width)", 0.1, 10.0, 2.0)
        with col2:
            b = st.slider("b (slope)", 0.1, 10.0, 2.0)
        with col3:
            c = st.slider("c (center)", 0.1, 10.0, 5.0)
        fs = FuzzySet.bell(universe, a, b, c)
    elif mtype == "Sigmoid":
        a,c = st.slider("Parameters (a,c)", -10.0,10.0,(1.0,5.0))
        fs = FuzzySet.sigmoid(universe,a,c)

    elif mtype == "Manual":

      values = st.text_area(
        "Enter membership values (comma-separated)",
        key="manual_membership_input"
    )

    # Everything related to 'values' MUST be inside this block!
      if values.strip() != "":
        try:
            parts = [v.strip() for v in values.split(",") if v.strip() != ""]
            mua = np.array([float(v) for v in parts])

            fs = FuzzySet.manual(universe[:len(mua)], mua)

        except ValueError:
            st.error("Invalid input! Please enter only numbers separated by commas.")
            fs = None
      else:
        fs = None

    if st.button("Add Fuzzy Set"):
        st.session_state["sets"][name] = fs
        st.success(f"Added Fuzzy Set: {name}")

if fs is not None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fs.universe, y=fs.membership, mode="lines", name=name))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter valid parameters to display the membership function.")


# --- Tab 2: Operations ---
with tabs[1]:
    st.header("Fuzzy Set Operations")
    sets = list(st.session_state["sets"].keys())
    if len(sets) < 2:
        st.warning("Define at least two sets first.")
    else:
        A = st.selectbox("Select Set A", sets)
        B = st.selectbox("Select Set B", sets)
        op = st.selectbox("Select Operation", [
            "Equality","Complement (A)","Intersection","Union","Algebraic Product",
            "Multiplication by Crisp","Power of Fuzzy Set","Algebraic Sum",
            "Algebraic Difference","Bounded Sum","Bounded Difference"
        ])
        fsA, fsB = st.session_state["sets"][A], st.session_state["sets"][B]
        result = None

        if op=="Equality": result = fsA == fsB
        elif op=="Complement (A)": result = fsA.complement()
        elif op=="Intersection": result = fsA.intersection(fsB)
        elif op=="Union": result = fsA.union(fsB)
        elif op=="Algebraic Product": result = fsA.algebraic_product(fsB)
        elif op=="Multiplication by Crisp": result = fsA.multiply_by_crisp(st.slider("k",0.0,2.0,1.0))
        elif op=="Power of Fuzzy Set": result = fsA.power(st.slider("p",0.1,5.0,2.0))
        elif op=="Algebraic Sum": result = fsA.algebraic_sum(fsB)
        elif op=="Algebraic Difference": result = fsA.algebraic_difference(fsB)
        elif op=="Bounded Sum": result = fsA.bounded_sum(fsB)
        elif op=="Bounded Difference": result = fsA.bounded_difference(fsB)

        if result is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fsA.universe, y=fsA.membership, name=A))
            fig.add_trace(go.Scatter(x=fsB.universe, y=fsB.membership, name=B))
            if isinstance(result, FuzzySet):
                fig.add_trace(go.Scatter(x=result.universe, y=result.membership, name="Result", line=dict(width=4)))
            st.plotly_chart(fig, use_container_width=True)

 # --- Tab 3: Implications ---
with tabs[2]:
    st.header("Fuzzy Implications with IF-THEN Conditions")

    if len(st.session_state["sets"]) < 2:
        st.warning("Define at least two sets first.")
    else:

        st.subheader("Define Condition")

        # Condition: IF A THEN B
        A = st.selectbox("IF (Antecedent) – choose fuzzy set A", sets, key="condA")
        B = st.selectbox("THEN (Consequent) – choose fuzzy set B", sets, key="condB")

        method = st.selectbox("Choose Implication Method", ["Zadeh", "Mamdani", "Larsen"], key="implMethod")

        fsA = st.session_state["sets"][A]
        fsB = st.session_state["sets"][B]

        st.subheader("Choose Input Value for Antecedent A")

        # Choose a crisp input value x
        x_val = st.slider(
            f"Select crisp value for A (x)",
            float(fsA.universe.min()),
            float(fsA.universe.max()),
            float(fsA.universe.mean()),
            key="input_x"
        )

        # Calculate μA(x)
        muA = np.interp(x_val, fsA.universe, fsA.membership)

        st.info(f"Membership value:  **μA({x_val:.2f}) = {muA:.4f}**")

        st.subheader("Implication Result")

        # Compute implication output
        if method == "Zadeh":
            # IF A THEN B = max(1 – μA, μB)
            implication_curve = fsA.zadeh_implication(fsB)
            implication_output = max(1 - muA, muA)  # evaluation at that point
        elif method == "Mamdani":
            # IF A THEN B = min(μA, μB)
            implication_curve = fsA.mamdani_implication(fsB)
            implication_output = muA
        else:  # Larsen
            # IF A THEN B = μA * μB
            implication_curve = fsA.larsen_implication(fsB)
            implication_output = muA  # multiplier will be applied on the curve

        st.success(f"**IF {A}({x_val:.2f}) THEN {B} = {implication_output:.4f}**")

        st.subheader("Implication Curve Visualization")

        # Plot result curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=implication_curve.universe,
            y=implication_curve.membership,
            mode="lines",
            name=f"{method} Implication"
        ))

        # Mark the point (x_val, implication_output)
        fig.add_trace(go.Scatter(
            x=[x_val],
            y=[implication_output],
            mode="markers+text",
            text=[f"{implication_output:.3f}"],
            textposition="top center",
            marker=dict(size=10)
        ))

        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Defuzzification ---
# --- Tab 4: Defuzzification ---
with tabs[3]:
    st.header("Defuzzification Methods")

    if len(st.session_state["sets"]) == 0:
        st.warning("Define a fuzzy set first.")
    else:
        name = st.selectbox("Select Fuzzy Set", sets)
        fs = st.session_state["sets"][name]

        method = st.selectbox("Method", [
            "Centroid","Bisector","Mean of Maximum","Smallest of Maximum","Largest of Maximum",
            "Lambda-cut","Weighted Average","Height Method","Center of Sums","Center of Area"
        ])

        # Compute result
        result = getattr(fs, method.lower().replace(" ","_").replace("-","_"))()

        if isinstance(result, np.ndarray):
            if result.size == 1:
                result = float(result)
            else:
                # You can choose: mean, min, max
                result = float(np.mean(result))  # average of possible values

        st.success(f"Defuzzified Value: {result:.4f}")

        # Plot set
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fs.universe, y=fs.membership, mode="lines", name=name))
        fig.add_vline(x=result, line_color="red")
        st.plotly_chart(fig, use_container_width=True)
