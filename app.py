import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder


# Streamlit UI
st.title("📊 Business Intelligence")

# Upload File
uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")

if uploaded_file is not None:
    df_amazon = pd.read_csv(uploaded_file)

    # --- Data Cleaning ---
    st.header("🧹 Data Cleaning")

    # Convert rating_count to numeric after removing commas
    df_amazon['rating_count'] = df_amazon['rating_count'].astype(str).str.replace(',', '', regex=False)
    df_amazon['rating_count'] = pd.to_numeric(df_amazon['rating_count'], errors='coerce')
    df_amazon['rating_count'].fillna(df_amazon['rating_count'].median(), inplace=True)
    df_amazon['rating_count'] = df_amazon['rating_count'].astype(int)

    # Convert currency and percentage columns to numeric
    for col in ['discounted_price', 'actual_price', 'discount_percentage', 'rating']:
        df_amazon[col] = df_amazon[col].astype(str).str.replace(r'[₹,%]', '', regex=True)
        df_amazon[col] = pd.to_numeric(df_amazon[col], errors='coerce')

    # Drop rows with missing values
    df_amazon.dropna(subset=['discounted_price', 'actual_price', 'discount_percentage', 'rating'], inplace=True)
    df_amazon.drop_duplicates(inplace=True)

    # Debugging: Show cleaned data
    st.write("✅ **Cleaned Data Sample:**")
    st.dataframe(df_amazon.head())

    # --- Extract Country from Text Fields ---
    country_mapping = {
        'India': 'India', 'USA': 'USA', 'US': 'USA', 'UK': 'UK', 
        'United Kingdom': 'UK', 'China': 'China', 'Germany': 'Germany', 
        'Japan': 'Japan', 'Australia': 'Australia', 'Canada': 'Canada'
    }

    def extract_country(text):
        if isinstance(text, str):
            for keyword, country in country_mapping.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    return country
        return None

    df_amazon['country'] = (
        df_amazon['review_content'].apply(extract_country)
        .combine_first(df_amazon['review_title'].apply(extract_country))
        .combine_first(df_amazon['product_name'].apply(extract_country))
        .combine_first(df_amazon['about_product'].apply(extract_country))
    )

    df_amazon.dropna(subset=['country'], inplace=True)

    # Debugging: Show unique countries
    st.write("🌍 **Unique Countries in Data:**", df_amazon['country'].unique())

    # --- Sales & Profit Calculation ---
    df_amazon['profit'] = df_amazon['discounted_price'] * 0.1  # 10% profit assumption

    grouped_data = df_amazon.groupby(['country', 'product_name']).agg(
        total_sales=('discounted_price', 'sum'),
        total_profit=('profit', 'sum')
    ).reset_index()

    # Debugging: Show grouped data
    st.write("📊 **Grouped Data Sample:**")
    st.dataframe(grouped_data.head())

    # --- Data Visualization ---
    st.header("📈 Data Visualization")

    if not grouped_data.empty:
        # Sales per Country
        st.write("## Total Sales per Country")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(grouped_data['country'], grouped_data['total_sales'], color='skyblue')
        ax.set_xlabel('Country')
        ax.set_ylabel('Total Sales')
        ax.set_title('Total Sales per Country')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        # Profit per Country
        st.write("## Total Profit per Country")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(grouped_data['country'], grouped_data['total_profit'], color='lightcoral')
        ax.set_xlabel('Country')
        ax.set_ylabel('Total Profit')
        ax.set_title('Total Profit per Country')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        # Top 10 most sold products
        top_10_products = grouped_data.nlargest(10, 'total_sales')
        st.write("## Top 10 Most Sold Products")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_10_products['product_name'], top_10_products['total_sales'], color='mediumseagreen')
        ax.set_xlabel('Total Sales')
        ax.set_ylabel('Product Name')
        ax.set_title('Top 10 Most Sold Products')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("⚠️ No data available for visualization. Check your dataset.")

    # --- Model Training ---
    st.header("🤖 Model Training and Evaluation")

    X = grouped_data[['country', 'product_name', 'total_profit']]
    y = grouped_data['total_sales']

    if not X.empty:
        # One-Hot Encoding
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = encoder.fit_transform(X[['country', 'product_name']])
        feature_names = encoder.get_feature_names_out(['country', 'product_name'])

        encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
        X = pd.concat([X.drop(columns=['country', 'product_name']), encoded_df], axis=1)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Model Evaluation
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.write(f"📊 **R-squared:** {r2:.4f}")
        st.write(f"📉 **Mean Absolute Error:** {mae:.2f}")
        st.write(f"📈 **Root Mean Squared Error:** {rmse:.2f}")

    else:
        st.write("⚠️ Not enough data for training the model. Please upload a valid dataset.")

    # --- Insights ---
    st.header("📢 Insights")

    insights = [
        f"🔹 The top-performing countries in terms of sales are: **{', '.join(grouped_data.groupby('country')['total_sales'].sum().nlargest(3).index)}**",
        f"🔹 The most profitable countries are: **{', '.join(grouped_data.groupby('country')['total_profit'].sum().nlargest(3).index)}**",
        f"🔹 The top-selling products are: **{', '.join(top_10_products['product_name'].head(3))}**",
        f"🔹 The linear regression model achieved an R-squared of **{r2:.4f}**, indicating its predictive strength.",
        "🔹 Further analysis could explore seasonal trends and demand forecasting."
    ]

    for insight in insights:
        st.write(f"- {insight}")
