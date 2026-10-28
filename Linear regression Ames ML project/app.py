import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="wide")

NUM = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BIN = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
FEATS = NUM + BIN + ["furnished"]          # same features as lr_4 in the notebook


@st.cache_data
def train(raw: pd.DataFrame):
    """Repeat the notebook steps: encode -> split -> MinMax scale -> OLS (lr_4)."""
    df = raw.copy()
    if "furnishingstatus" in df.columns:        # original Housing.csv
        df[BIN] = df[BIN].apply(lambda s: s.map({"yes": 1, "no": 0}))
        df["furnished"] = (df["furnishingstatus"] == "furnished").astype(int)
        df = df.drop(columns="furnishingstatus")
    else:                                       # housing_clean.csv (already encoded)
        df["furnished"] = df["furnished"].astype(int)
        df = df[["price"] + FEATS]

    tr, te = train_test_split(df, train_size=0.7, test_size=0.3, random_state=100)
    tr, te = tr.copy(), te.copy()

    cols = NUM + ["price"]
    sc = MinMaxScaler()
    tr[cols] = sc.fit_transform(tr[cols])
    te[cols] = sc.transform(te[cols])

    model = sm.OLS(tr["price"], sm.add_constant(tr[FEATS])).fit()
    b = model.params

    # Convert scaled coefficients -> raw units, so: price = c + a1*x1 + a2*x2 ...
    lo = dict(zip(cols, sc.data_min_))
    rng = dict(zip(cols, sc.data_range_))
    y_lo, y_rng = lo["price"], rng["price"]
    raw_coef = {f: b[f] * y_rng / rng[f] if f in NUM else b[f] * y_rng for f in FEATS}
    raw_int = y_lo + y_rng * (b["const"] - sum(b[f] * lo[f] / rng[f] for f in NUM))

    pred_te = model.predict(sm.add_constant(te[FEATS], has_constant="add"))
    r2_test = 1 - ((te["price"] - pred_te) ** 2).sum() / ((te["price"] - te["price"].mean()) ** 2).sum()

    return dict(
        scaled=b, raw_coef=raw_coef, raw_int=raw_int,
        r2_train=model.rsquared, r2_adj=model.rsquared_adj, r2_test=r2_test,
        pvalues=model.pvalues, ranges={f: (lo[f], lo[f] + rng[f]) for f in NUM},
    )


# ---------- Data ----------
st.title("🏠 Housing Price Predictor")
st.caption("Linear regression (lr_4 from the notebook): price = c + a·x + b·z + ...")

up = st.sidebar.file_uploader("Upload Housing.csv", type="csv")
currency = st.sidebar.text_input("Currency label", "USD")
try:
    raw = pd.read_csv(up) if up else pd.read_csv("Housing.csv")
except FileNotFoundError:
    st.info("Upload **Housing.csv** in the sidebar (or put it next to app.py).")
    st.stop()

m = train(raw)
c1, c2, c3 = st.columns(3)
c1.metric("R² (train)", f"{m['r2_train']:.3f}")
c2.metric("Adj. R² (train)", f"{m['r2_adj']:.3f}")
c3.metric("R² (test)", f"{m['r2_test']:.3f}")

tab_pred, tab_coef = st.tabs(["💰 Predict price", "📊 Coefficients"])

# ---------- Prediction ----------
with tab_pred:
    st.subheader("Change the inputs")
    left, right = st.columns(2)
    x = {}
    with left:
        for f in NUM:
            lo, hi = m["ranges"][f]
            x[f] = st.number_input(f.title(), float(lo), float(hi) * 2, float(np.median(raw[f])), step=1.0)
    with right:
        for f in BIN:
            x[f] = 1 if st.selectbox(f.title(), ["no", "yes"]) == "yes" else 0
        x["furnished"] = 1 if st.selectbox("Furnishing", ["not furnished", "furnished"]) == "furnished" else 0

    rows = [{"Feature": f, "Value (x)": x[f], "Coefficient": m["raw_coef"][f],
             "Contribution": m["raw_coef"][f] * x[f]} for f in FEATS]
    rows.append({"Feature": "Intercept (c)", "Value (x)": 1, "Coefficient": m["raw_int"], "Contribution": m["raw_int"]})
    calc = pd.DataFrame(rows)
    price = calc["Contribution"].sum()

    st.metric("Predicted price", f"{price:,.2f} {currency}")

    st.markdown("**Equation**")
    terms = " + ".join(f"({m['raw_coef'][f]:.4g})({x[f]:g})" for f in FEATS)
    st.code(f"price = {terms} + ({m['raw_int']:.4g})\nprice = {price:,.2f} {currency}", language="text")

    st.dataframe(calc.style.format({"Value (x)": "{:g}", "Coefficient": "{:,.4f}", "Contribution": "{:,.2f}"}),
                 width="stretch", hide_index=True)

# ---------- Coefficients ----------
with tab_coef:
    st.subheader("Model coefficients")
    coef = pd.DataFrame({
        "Feature": ["Intercept (c)"] + FEATS,
        "Scaled coef (0–1 data)": [m["scaled"]["const"]] + [m["scaled"][f] for f in FEATS],
        "Raw coef (real units)": [m["raw_int"]] + [m["raw_coef"][f] for f in FEATS],
        "p-value": [m["pvalues"]["const"]] + [m["pvalues"][f] for f in FEATS],
    })
    st.dataframe(coef.style.format({"Scaled coef (0–1 data)": "{:.4f}", "Raw coef (real units)": "{:,.4f}",
                                    "p-value": "{:.4f}"}), width="stretch", hide_index=True)
    st.bar_chart(coef[coef["Feature"] != "Intercept (c)"].set_index("Feature")["Scaled coef (0–1 data)"])
    st.caption("Scaled coefs are comparable to each other (all features on 0–1). "
               "Raw coefs are what the prediction equation uses.")
