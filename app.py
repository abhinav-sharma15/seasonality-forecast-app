
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")

st.title("📈 Seasonality-Adjusted Unique Visitor Forecast")

st.markdown("""
This app forecasts 2025 unique visitors by adjusting for seasonality based on 2024 trends.
You can upload your own CSV file or use the default data provided below.

**CSV Format Required**: month, last_year_data, this_year_data
""")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

try:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_columns = {"month", "last_year_data", "this_year_data"}
        if not required_columns.issubset(df.columns):
            st.error("❌ Invalid CSV format. Required columns: month, last_year_data, this_year_data")
            st.stop()
    else:
        # Default monthly data
        last_year_data = [853892, 753225, 1012344, 818551, 758921, 784341, 689904, 637453, 729752, 722105, 747647, 703750]
        this_year_data = [564649, 501947, 612701, 459772, 452334, 0, 0, 0, 0, 0, 0, 0]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df = pd.DataFrame({"month": months, "last_year_data": last_year_data, "this_year_data": this_year_data})

    # Step 1: Calculate seasonal index
    average_last_year = df["last_year_data"].mean()
    df["seasonal_index"] = df["last_year_data"] / average_last_year

    # Step 2: Deseasonalize known this_year values
    df["this_year_deseasonalized"] = df["this_year_data"] / df["seasonal_index"]

    # Step 3: Train model
    train_df = df[df["this_year_data"] > 0]
    X_train = train_df["last_year_data"].values.reshape(-1, 1)
    y_train = train_df["this_year_deseasonalized"].values
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Predict for unknown months
    predict_df = df[df["this_year_data"] == 0]
    X_predict = predict_df["last_year_data"].values.reshape(-1, 1)
    y_pred_deseasonalized = model.predict(X_predict)

    # Step 5: Reseasonalize
    reseasonalized = y_pred_deseasonalized * predict_df["seasonal_index"].values
    df.loc[df["this_year_data"] == 0, "this_year_data_adjusted"] = reseasonalized.astype(int)
    df["this_year_data_adjusted"] = df["this_year_data_adjusted"].fillna(df["this_year_data"])

    # Display forecast table
    st.subheader("Forecast Table")
    st.dataframe(df[["month", "last_year_data", "this_year_data", "this_year_data_adjusted"]].set_index("month"))

    # Plot
    st.subheader("Forecast Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["month"], df["last_year_data"], label="Last Year", marker='o')
    ax.plot(df["month"], df["this_year_data_adjusted"], label="This Year (Adjusted)", marker='s')
    ax.set_title("Seasonality-Adjusted Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Unique Visitors")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Downloadable forecast CSV
    st.subheader("Download Adjusted Forecast")
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="forecast_adjusted.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"❌ An error occurred: {e}")
