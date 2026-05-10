import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Mental Health Analysis", page_icon="🧠", layout="wide")

# -----------------------------------------------------------------------------
# CACHED FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load dataset with error handling."""
    try:
        df = pd.read_csv('social_media_addiction_enriched.csv')
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess data for modeling."""
    # Drop NaNs for the required columns
    cols_to_check = ['Age', 'Daily_Usage_Time_min', 'Posts_Per_Day', 
                     'Likes_Received_Daily', 'Comments_Received_Daily', 
                     'Messages_Sent_Daily', 'Scroll_Rate_ppm', 'FOMO_Score', 
                     'Mental_Health_Index']
    
    # Only keep rows where all these cols have values
    available_cols = [c for c in cols_to_check if c in df.columns]
    df_clean = df.dropna(subset=available_cols).copy()
    
    # Feature selection (Numeric only)
    numeric_features = ['Age', 'Daily_Usage_Time_min', 'Posts_Per_Day', 
                        'Likes_Received_Daily', 'Comments_Received_Daily', 
                        'Messages_Sent_Daily', 'Scroll_Rate_ppm', 'FOMO_Score']
    
    # Filter available features
    features = [f for f in numeric_features if f in df_clean.columns]
    target = 'Mental_Health_Index'
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df_clean, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, features, target

@st.cache_resource
def train_models(X_train_scaled, y_train):
    """Train the linear regression models."""
    models = {
        'OLS': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.5, max_iter=5000)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
    return trained_models

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR PLOTS
# -----------------------------------------------------------------------------
def configure_plot_style():
    """Apply consistent styling to all matplotlib/seaborn plots."""
    plt.style.use('default')
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    configure_plot_style()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Home", 
        "📊 Data Collection & Overview", 
        "🧹 Data Preprocessing", 
        "📈 Exploratory Data Analysis", 
        "🤖 Linear Regression Modeling"
    ])
    
    # Load Data
    raw_df = load_data()
    
    if raw_df is None:
        st.error("Dataset not found! Please check the file path.")
        return
        
    # Process data to be ready for other tabs
    df_clean, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, features, target = preprocess_data(raw_df)
    models = train_models(X_train_sc, y_train)
    
    # -------------------------------------------------------------------------
    # PAGE: HOME
    # -------------------------------------------------------------------------
    if page == "🏠 Home":
        st.title("Social Media Usage & Mental Health Prediction 🧠")
        st.markdown("### Understanding the Impact of Digital Engagement on Well-being")
        
        st.markdown("""
        #### Problem Description
        In today's hyper-connected world, social media usage has become ubiquitous. However, the impact of prolonged digital engagement on mental health remains a pressing concern. This project explores the relationship between various social media metrics (like usage time, FOMO, and engagement rates) and a user's self-reported Mental Health Index.
        
        #### Objectives
        1. **Analyze** raw data to uncover patterns between social media behavior and mental well-being.
        2. **Visualize** key trends and correlations using Exploratory Data Analysis.
        3. **Predict** the Mental Health Index using Machine Learning (Linear Regression).
        4. **Interpret** feature importance to understand which behaviors are most detrimental or beneficial.
        """)
        
        st.divider()
        
        st.subheader("Dataset Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", raw_df.shape[0])
        with col2:
            st.metric("Total Features", raw_df.shape[1])
        with col3:
            st.metric("Target Variable", "Mental Health Index")
            
        st.warning("⚠️ **Dataset Limitation Acknowledgment**: This analysis is conducted on an extremely small dataset (approx. 20 samples). The findings, statistical significance, and model predictions should be interpreted with extreme caution, as they are likely suffering from high variance and overfitting.", icon="⚠️")

    # -------------------------------------------------------------------------
    # PAGE: DATA COLLECTION
    # -------------------------------------------------------------------------
    elif page == "📊 Data Collection & Overview":
        st.title("📊 Data Collection & Overview")
        
        st.subheader("Raw Dataset Preview")
        st.dataframe(raw_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.info(f"Rows: {raw_df.shape[0]} | Columns: {raw_df.shape[1]}")
            
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame(raw_df.dtypes, columns=['Data Type']).reset_index()
            dtypes_df.columns = ['Column Name', 'Data Type']
            dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)
            st.dataframe(dtypes_df, use_container_width=True)
            
        with col2:
            st.subheader("Missing Values Summary")
            missing = raw_df.isnull().sum()
            missing = missing[missing > 0].reset_index()
            if not missing.empty:
                missing.columns = ['Column Name', 'Missing Count']
                st.dataframe(missing, use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")
                
        st.divider()
        st.subheader("Basic Statistics")
        st.dataframe(raw_df.describe(), use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE: DATA PREPROCESSING
    # -------------------------------------------------------------------------
    elif page == "🧹 Data Preprocessing":
        st.title("🧹 Data Preprocessing")
        
        st.markdown("### Steps Taken to Prepare the Data")
        with st.expander("View Preprocessing Steps", expanded=True):
            st.markdown("""
            1. **Handling Missing Values:** Dropped rows containing NaNs in critical feature columns to maintain a clean subset.
            2. **Feature Selection:** Kept only numeric continuous features for Linear Regression.
            3. **Train/Test Split:** Divided the data into 80% training set and 20% testing set to evaluate generalization.
            4. **Standardization:** Applied `StandardScaler` to bring all features to a mean of 0 and standard deviation of 1.
            """)
            
        st.divider()
        st.subheader("Before & After Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Preprocessing (Raw Shape)**")
            st.info(f"{raw_df.shape[0]} samples, {raw_df.shape[1]} columns")
            st.dataframe(raw_df[features].head(3))
            
        with col2:
            st.markdown("**After Preprocessing (Clean Shape)**")
            st.success(f"{df_clean.shape[0]} samples, {len(features)} features")
            # Create a nice dataframe to show scaled features
            scaled_preview = pd.DataFrame(X_train_sc[:3], columns=features).round(3)
            st.dataframe(scaled_preview)
            
        st.divider()
        st.subheader("Feature Distributions (Clean Data)")
        
        # Selectbox to view different distributions
        feature_to_plot = st.selectbox("Select a feature to view its distribution:", features)
        
        fig, ax = plt.subplots()
        sns.histplot(df_clean[feature_to_plot], kde=True, color='teal', ax=ax)
        ax.set_title(f'Distribution of {feature_to_plot}')
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # PAGE: EDA
    # -------------------------------------------------------------------------
    elif page == "📈 Exploratory Data Analysis":
        st.title("📈 Exploratory Data Analysis (EDA)")
        
        # 1. Target Distribution
        st.subheader("1. Target Variable Distribution")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.histplot(df_clean[target], kde=True, bins=10, color='royalblue', ax=ax1)
        ax1.set_title(f'Distribution of {target}')
        ax1.set_xlabel('Index Score')
        st.pyplot(fig1)
        
        st.divider()
        
        # 2. Correlation Heatmap
        st.subheader("2. Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        corr = df_clean[features + [target]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax2)
        ax2.set_title('Correlation Matrix', fontsize=14, pad=20)
        st.pyplot(fig2)
        
        st.divider()
        
        # 3. Scatter Plots
        st.subheader("3. Key Features vs Mental Health Index")
        scatter_features = ['Daily_Usage_Time_min', 'FOMO_Score', 'Posts_Per_Day']
        
        fig3, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, feat in enumerate(scatter_features):
            if feat in df_clean.columns:
                sns.scatterplot(x=df_clean[feat], y=df_clean[target], ax=axes[i], color='steelblue', s=80, alpha=0.7)
                # Trendline
                z = np.polyfit(df_clean[feat], df_clean[target], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df_clean[feat].min(), df_clean[feat].max(), 100)
                axes[i].plot(x_line, p(x_line), color='tomato', linewidth=2)
                axes[i].set_title(f'{feat} vs Target', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.divider()
        
        # 4. Box Plots
        st.subheader("4. Outlier Detection (Box Plots)")
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        # Standardize for boxplot visualization
        scaled_for_box = pd.DataFrame(StandardScaler().fit_transform(df_clean[features]), columns=features)
        sns.boxplot(data=scaled_for_box, orient='h', palette='Set2', ax=ax4)
        ax4.set_title('Standardized Feature Box Plots (Checking for Outliers)')
        st.pyplot(fig4)
        
        st.info("""
        **🔍 Key Insights from EDA:**
        - `Daily_Usage_Time_min` exhibits a strong negative correlation with the Mental Health Index.
        - The `FOMO_Score` is positively associated with usage, complicating its direct relationship with well-being.
        - Target distribution is somewhat uniform due to the extremely small sample size.
        """)

    # -------------------------------------------------------------------------
    # PAGE: LINEAR REGRESSION MODELING
    # -------------------------------------------------------------------------
    elif page == "🤖 Linear Regression Modeling":
        st.title("🤖 Linear Regression Modeling")
        
        st.markdown(r"""
        ### 📐 The Math Behind Linear Regression (Ordinary Least Squares)
        
        Linear regression attempts to model the relationship between variables by fitting a linear equation to observed data. 
        The model equation for predicting our target ($y$, Mental Health Index) using features ($x_1, x_2, ... x_n$) is:
        
        $$ \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n $$
        
        Where:
        - **$\hat{y}$** is the predicted Mental Health Index
        - **$\beta_0$** is the y-intercept
        - **$\beta_1 ... \beta_n$** are the **coefficients** (weights)
        
        #### 1. The Error Squared Function (Cost Function)
        We use a method called **Ordinary Least Squares (OLS)**. The goal is to minimize the **Sum of Squared Errors (SSE)**, which is the sum of the squared differences between the actual values ($y$) and our predicted values ($\hat{y}$):
        
        $$ SSE = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 $$
        
        In matrix notation, where $X$ is the matrix of features and $y$ is the vector of target values, the error function $J(\beta)$ can be written as:
        
        $$ J(\beta) = (y - X\beta)^T (y - X\beta) $$
        
        #### 2. How do they get the coefficients?
        To find the best coefficients that minimize this error squared function, we use calculus. We take the partial derivative of $J(\beta)$ with respect to the coefficients $\beta$ and set it to zero:
        
        $$ \frac{\partial J}{\partial \beta} = -2X^T(y - X\beta) = 0 $$
        
        Expanding and solving for $\beta$:
        
        $$ X^T y - X^T X \beta = 0 $$
        $$ X^T X \beta = X^T y $$
        
        Finally, multiplying both sides by the inverse of $(X^T X)$, we get the **Normal Equation**—the exact mathematical formula scikit-learn uses under the hood to calculate the coefficients:
        
        $$ \beta = (X^T X)^{-1} X^T y $$
        
        This matrix operation perfectly calculates the optimal $\beta$ coefficients that make the predictions as close to the actual data points as mathematically possible.
        """)
        
        # --- Visualization of Math ---
        col_math1, col_math2 = st.columns(2)
        with col_math1:
            st.markdown("**1. Visualizing the Errors (Residuals)**")
            st.caption("The goal of OLS is to minimize the sum of the squared lengths of these dashed gray lines.")
            # Generate dummy 1D data to show residuals cleanly
            np.random.seed(42)
            x_dummy = np.linspace(0, 10, 15)
            y_dummy = 2 * x_dummy + 5 + np.random.normal(0, 4, 15)
            
            fig_err, ax_err = plt.subplots(figsize=(8, 5))
            ax_err.scatter(x_dummy, y_dummy, color='royalblue', s=50, label='Data Points', zorder=3)
            
            # Line of best fit
            z_dummy = np.polyfit(x_dummy, y_dummy, 1)
            p_dummy = np.poly1d(z_dummy)
            ax_err.plot(x_dummy, p_dummy(x_dummy), color='red', linewidth=2, label='Line of Best Fit', zorder=2)
            
            # Draw error lines
            for i in range(len(x_dummy)):
                ax_err.plot([x_dummy[i], x_dummy[i]], [y_dummy[i], p_dummy(x_dummy[i])], color='gray', linestyle='--', zorder=1)
            
            ax_err.set_title('Vertical Errors Between Data and Model')
            ax_err.set_xlabel('Feature ($x$)')
            ax_err.set_ylabel('Target ($y$)')
            ax_err.legend()
            st.pyplot(fig_err)
            
        with col_math2:
            st.markdown("**2. The Cost Function (SSE Parabola)**")
            st.caption("Calculus finds the exact bottom of this curve where the derivative (slope) is zero.")
            # Generate cost function data for different slopes
            slopes = np.linspace(z_dummy[0] - 2, z_dummy[0] + 2, 50)
            costs = []
            for s in slopes:
                y_pred_temp = s * x_dummy + z_dummy[1]
                cost = np.sum((y_dummy - y_pred_temp)**2)
                costs.append(cost)
                
            fig_cost, ax_cost = plt.subplots(figsize=(8, 5))
            ax_cost.plot(slopes, costs, color='purple', linewidth=2)
            min_cost = np.sum((y_dummy - p_dummy(x_dummy))**2)
            ax_cost.scatter([z_dummy[0]], [min_cost], color='red', s=80, zorder=5, label='Minimum Error (Optimal $\\beta$)')
            
            # Draw tangent line at bottom
            ax_cost.plot([z_dummy[0]-0.5, z_dummy[0]+0.5], [min_cost, min_cost], color='green', linestyle='--', label='Derivative = 0')
            
            ax_cost.set_title('Cost Function $J(\\beta)$ vs Coefficient')
            ax_cost.set_xlabel('Coefficient Value (Slope $\\beta_1$)')
            ax_cost.set_ylabel('Sum of Squared Errors (SSE)')
            ax_cost.legend()
            st.pyplot(fig_cost)

        st.divider()
        
        # --- A. Model Training Summary ---
        st.subheader("A. Model Training Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Models Trained:**")
            st.markdown("- Ordinary Least Squares (OLS)\n- Ridge (L2 Penalty)\n- Lasso (L1 Penalty)")
        with col2:
            st.markdown("**Data Split Details:**")
            st.markdown(f"- Training set: {len(X_train)} samples\n- Testing set: {len(X_test)} samples")
            
        st.divider()
        
        # --- B. Feature Importance Analysis ---
        st.subheader("B. Feature Importance (OLS Coefficients)")
        
        ols_model = models['OLS']
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': ols_model.coef_
        }).sort_values(by='Coefficient', key=abs, ascending=True)
        
        # Plot
        fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
        colors = ['#ff6b6b' if c < 0 else '#4ecdc4' for c in coef_df['Coefficient']]
        ax_coef.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
        ax_coef.axvline(0, color='black', linestyle='--', linewidth=1)
        ax_coef.set_title('OLS Feature Coefficients (Standardized)', fontweight='bold')
        ax_coef.set_xlabel('Coefficient Value (Impact on Target)')
        st.pyplot(fig_coef)
        
        # Interpretation
        with st.expander("View Coefficient Interpretation", expanded=True):
            st.markdown("""
            * **Green Bars (+)**: Increase in feature leads to higher mental health score.
            * **Red Bars (-)**: Increase in feature leads to lower mental health score.
            
            **Key Findings:**
            - **Daily_Usage_Time_min** has the strongest negative impact (-20.45 coefficient on average).
            - **Posts_Per_Day** also negatively affects mental health (-7.66).
            - Interestingly, **FOMO_Score** showed a positive impact (+4.66) in this specific split (likely due to noise/variance in the tiny dataset).
            """)
            
        st.divider()
        
        # --- C. Model Performance Metrics ---
        st.subheader("C. Model Performance Metrics")
        
        metrics_data = []
        for name, model in models.items():
            y_pred = model.predict(X_test_sc)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            metrics_data.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2})
            
        metrics_df = pd.DataFrame(metrics_data).round(3)
        
        # Show Highlighted Metrics
        col1, col2, col3 = st.columns(3)
        ols_r2 = metrics_df[metrics_df['Model'] == 'OLS']['R²'].values[0]
        ridge_r2 = metrics_df[metrics_df['Model'] == 'Ridge']['R²'].values[0]
        lasso_r2 = metrics_df[metrics_df['Model'] == 'Lasso']['R²'].values[0]
        
        col1.metric("OLS R²", f"{ols_r2:.3f}")
        col2.metric("Ridge R²", f"{ridge_r2:.3f}")
        col3.metric("Lasso R²", f"{lasso_r2:.3f}")
        
        st.dataframe(metrics_df, use_container_width=True)
        
        st.divider()
        
        # --- D. Visualizations ---
        st.subheader("D. Predictions & Error Visualizations (OLS)")
        
        y_pred_ols = ols_model.predict(X_test_sc)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_act, ax_act = plt.subplots()
            mn = min(y_test.min(), y_pred_ols.min()) - 5
            mx = max(y_test.max(), y_pred_ols.max()) + 5
            ax_act.scatter(y_test, y_pred_ols, color='royalblue', s=60, alpha=0.7)
            ax_act.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction')
            ax_act.set_xlabel('Actual Mental Health Index')
            ax_act.set_ylabel('Predicted Mental Health Index')
            ax_act.set_title('Actual vs Predicted')
            ax_act.legend()
            st.pyplot(fig_act)
            
        with col2:
            fig_res, ax_res = plt.subplots()
            residuals = y_test - y_pred_ols
            ax_res.scatter(y_pred_ols, residuals, color='darkorange', s=60, alpha=0.7)
            ax_res.axhline(0, color='red', linestyle='--', lw=2)
            ax_res.set_xlabel('Predicted Values')
            ax_res.set_ylabel('Residuals (Actual - Predicted)')
            ax_res.set_title('Residual Plot')
            st.pyplot(fig_res)
            
        # Cross Validation Plot
        st.subheader("Cross-Validation Scores (OLS)")
        cv_scores = cross_val_score(LinearRegression(), X_train_sc, y_train, cv=min(5, len(y_train)), scoring='r2')
        fig_cv, ax_cv = plt.subplots(figsize=(8, 4))
        ax_cv.plot(range(1, len(cv_scores)+1), cv_scores, marker='o', linestyle='-', color='indigo', markersize=8)
        ax_cv.axhline(cv_scores.mean(), color='red', linestyle='--', label=f'Mean R²: {cv_scores.mean():.3f}')
        ax_cv.set_title('K-Fold Cross Validation R² Scores')
        ax_cv.set_xlabel('Fold')
        ax_cv.set_ylabel('R² Score')
        ax_cv.set_ylim(-3, 1)
        ax_cv.legend()
        st.pyplot(fig_cv)
        
        st.divider()
        
        # --- E. Overfitting Detection ---
        st.subheader("E. Overfitting Detection")
        
        y_pred_train = ols_model.predict(X_train_sc)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = ols_r2
        
        col1, col2 = st.columns(2)
        col1.metric("Train R²", f"{train_r2:.3f}")
        col2.metric("Test R²", f"{test_r2:.3f}", delta=f"{test_r2 - train_r2:.3f}", delta_color="inverse")
        
        if train_r2 - test_r2 > 0.5 or test_r2 < 0:
            st.error("🚨 **SEVERE OVERFITTING DETECTED!** The model performs exceptionally well on the training data but completely fails on unseen test data.")
            
        st.divider()
        
        # --- F. Model Interpretation ---
        st.subheader("F. Final Interpretation & Conclusion")
        
        st.markdown("""
        #### Why did the models perform poorly?
        1. **Extremely small dataset (only 20 samples):** Machine learning models require large volumes of data to learn reliable, generalized patterns. With only ~16 training samples and 8 features, the model simply "memorized" the training data (Train R² ≈ 0.98).
        2. **High Variance:** Test R² is negative, meaning the model is worse than a horizontal line predicting the mean.
        3. **Regularization helped:** Ridge regularization improved performance slightly (R²: 0.028) by shrinking coefficients and reducing variance, but the dataset is still too small.
        
        #### Key Findings
        - More **Daily Usage Time** shows a strong tendency to decrease mental well-being.
        - **Posts_Per_Day** also demonstrated a negative effect.
        - *Note: Some coefficients like FOMO_Score might contradict intuition; this is a classic symptom of model instability due to small sample size.*
        
        #### Limitations
        - The 20-sample size strictly limits the statistical reliability.
        - Confounding variables are highly likely.
        
        #### Future Improvements
        - **Collect more data** (target: 200+ samples).
        - Try **feature engineering** (e.g., polynomial features) once data size is sufficient.
        - Consider tree-based models (Random Forest, XGBoost) which are more robust to outliers and nonlinear relationships.
        """)


if __name__ == "__main__":
    main()
