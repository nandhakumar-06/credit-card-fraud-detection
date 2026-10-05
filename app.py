from flask import Flask, render_template, request
import pandas as pd
import joblib
from xgboost import XGBClassifier

app = Flask(__name__)

# ============================================================
# MODEL FILES
# ============================================================

XGBOOST_FILE = "xgboost_model.json"
ISOLATION_FILE = "isolation_forest_model.pkl"


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier()
xgb_model.load_model(XGBOOST_FILE)


# ============================================================
# LOAD ISOLATION FOREST MODEL
# ============================================================

iso_model = joblib.load(ISOLATION_FILE)


# ============================================================
# CREATE INPUT DATA
# ============================================================

def create_input(time_value, amount_value):

    data = {
        "Time": float(time_value)
    }

    # V1 to V28
    for i in range(1, 29):
        data[f"V{i}"] = 0.0

    data["Amount"] = float(amount_value)

    return pd.DataFrame([data])


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# FRAUD PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Get values from HTML
        # ----------------------------------------------------

        time_text = request.form.get("Time", "").strip()
        amount_text = request.form.get("Amount", "").strip()

        # ----------------------------------------------------
        # Do NOT allow empty inputs
        # ----------------------------------------------------

        if time_text == "" or amount_text == "":
            return render_template(
                "index.html",
                error="Please enter both Transaction Time and Transaction Amount."
            )

        # ----------------------------------------------------
        # Convert to numbers
        # ----------------------------------------------------

        time_value = float(time_text)
        amount_value = float(amount_text)

        # ----------------------------------------------------
        # Check negative values
        # ----------------------------------------------------

        if time_value < 0:
            return render_template(
                "index.html",
                error="Transaction Time cannot be negative."
            )

        if amount_value < 0:
            return render_template(
                "index.html",
                error="Transaction Amount cannot be negative."
            )

        # ----------------------------------------------------
        # Create model input
        # ----------------------------------------------------

        input_data = create_input(
            time_value,
            amount_value
        )

        # ----------------------------------------------------
        # XGBoost Prediction
        # ----------------------------------------------------

        fraud_probability = xgb_model.predict_proba(
            input_data
        )[0][1]

        # ----------------------------------------------------
        # XGBoost decision
        # ----------------------------------------------------

        if fraud_probability >= 0.50:
            result = "fraud"
        else:
            result = "genuine"

        # ----------------------------------------------------
        # Isolation Forest
        # ----------------------------------------------------

        try:

            iso_prediction = iso_model.predict(
                input_data
            )[0]

            if iso_prediction == -1:
                isolation_result = "Anomaly detected"
            else:
                isolation_result = "Normal transaction"

        except Exception:

            isolation_result = "Not available"

        # ----------------------------------------------------
        # Convert probability to percentage
        # ----------------------------------------------------

        probability_percent = round(
            fraud_probability * 100,
            2
        )

        # ----------------------------------------------------
        # Send result to HTML
        # ----------------------------------------------------

        return render_template(
            "index.html",

            result=result,

            probability=probability_percent,

            isolation_result=isolation_result,

            time_value=time_value,

            amount_value=amount_value
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except ValueError:

        return render_template(
            "index.html",
            error="Please enter valid numeric values."
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("     CREDIT CARD FRAUD DETECTION SYSTEM")
    print("==============================================")
    print("     XGBoost + Isolation Forest")
    print("==============================================")
    print()
    print("Open this URL in your browser:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )