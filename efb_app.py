import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm


st.set_page_config(
    page_title="Esophageal foreign bodies analysis",
    layout="wide"
)

#create a title and description
st.title("Esophageal Foreign Bodies among Pediatric patient at RMH")
st.markdown(
    """
    This research is therefore essential to quantify the burden of esophageal FB ingestion among children at RMRTH, delineate the patterns and types of esophageal FB encountered and inform both clinical and policymaker.
    """
)

#set the sidebar 
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose Section:",[
        "1. Exploratory Data Analysis",
        "2. Model Training and Evaluation"
    ]
)


#load the sample dataset 
@st.cache_data

def clean_data(filepath):
    df = pd.read_excel(filepath)

    #clean the data
    #remove trails of extraspaces in the variable names 
    df.columns = df.columns.str.strip()
    #select columns of interest 
    var_interest = [
    'Age(years)',
    'Gender',
    'Health insurance',
    'Presented signs and symptoms /Dysphagia',
    'Presented signs and symptoms /Drooling',
    'Presented signs and symptoms /Odynophagia',
    'Presented signs and symptoms /Regurgitation',
    'Presented signs and symptoms /Gagging and choking',
    'Presented signs and symptoms /Chest pain',
    'Presented signs and symptoms /Shortness of breath',
    'Presented signs and symptoms /Coughing',
    'Foreign body ingested',
    'Ingestion Circumstance',
    'Predisposing Conditions',
    'Was the child supervised at the time',
    'Imaging modality used in diagnosis',
    'Location of foreign body',
    'Method of removal',
    'Complications',
    'Length of hospital stay (days)',
    'Treatment outcome']

    #df subset 
    df_subset = df[var_interest]
    #replace Nan Values in predisposing conditions 
    df_subset["Predisposing Conditions"] = df_subset['Predisposing Conditions'].fillna("None")
    #replace the Nan Values in Length of Hospital stay
    mean = df_subset["Length of hospital stay (days)"].mean()
    df_subset["Length of hospital stay (days)"] = df_subset['Length of hospital stay (days)'].fillna(mean)

    #lets do some individual clean ups 
    df_subset["Imaging modality used in diagnosis"] = df_subset["Imaging modality used in diagnosis"].replace("Endoscopy ultrasound", "Endoscopy")
    df_subset["Imaging modality used in diagnosis"] = df_subset["Imaging modality used in diagnosis"].replace("x-ray ultrasound", "x-ray")
    df_subset["Imaging modality used in diagnosis"] = df_subset["Imaging modality used in diagnosis"].replace("x-ray CT-scan", "CT-scan")
        
    df_subset["Method of removal"] = df_subset["Method of removal"].replace("Others(specify)", "conservative")
    df_subset["Method of removal"] = df_subset["Method of removal"].replace("Surgery", "conservative")

    df_subset["Complications"] = df_subset["Complications"].replace("Others(specify)","none" )
    df_subset["Complications"] = df_subset["Complications"].replace("Others(specify) none", "none")
    df_subset["Complications"] = df_subset["Complications"].replace("Perforation none", "Perforation")
    df_subset["Complications"] = df_subset["Complications"].replace("Perforation Infection", "Infection")

    # lets replace all missing values with the most common word 
    for col in df_subset.columns:
        if df_subset[col].dtype == "object":
            var_counts= df_subset[col].value_counts()
            most_common_word = var_counts.idxmax()
            df_subset[col] = df_subset[col].fillna(most_common_word)
        elif df_subset[col].dtype == "float64":
            # For numeric columns → replace NaN with most common value (mode)
            most_common_value = df_subset[col].mode()[0]
            df_subset[col] = df_subset[col].fillna(most_common_value)

    df_filtered = df_subset[(df_subset["Length of hospital stay (days)"] < 20) & (df_subset["Age(years)"] < 16)]
        

        

    return df_filtered

# Data Exploration 
if section == "1. Exploratory Data Analysis":
    #create a header and layout 
    st.header("Exploratory data Analysis")
    filepath = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])


    #create two columns
    

    #lets load the dataframe in the first column 
    if filepath is None:
        st.error("No selected File")
    else:

    
        st.subheader("1. The Dataset preview")

        #displaye the dataset info
        df = clean_data(filepath)
        st.dataframe(df.head())

        #show the dataset statistics 
        st.subheader("2. Dataset Information")
        st.write(f"shape: {df.shape}")
        st.write(f"Number of Features: {len(df.columns)}")
        st.write(f"Number of Samples: {len(df)}")

        #show if there are any missing values 
        st.metric("Missing values", df.isnull().sum().sum())

   

        #function to plot bar chart
        def plot_bar(cat_name):
            var_count = df[cat_name].value_counts()
            fig = px.bar(
                x = var_count,
                y = var_count.index,
                title = f"Distribution of {cat_name}"

            )
            fig.update_layout(
                xaxis_title=f"Frequency",
                yaxis_title = f"{cat_name}"
            )

            return fig
    
        #function to plot distribution 
        def plot_hist(cat_name):
            fig = px.histogram(
                df,
                x=cat_name,
                nbins=10,
                marginal="violin",
                
                title="Distribution of Length of Hospital in days")
            
            fig.update_layout(
                xaxis_title =f"{cat_name}",
                yaxis_title="Frequency"
            )
            return fig
        #we are going to create a visualisation for each variables 
        st.subheader("3. Data Exploration:")
        cat_name = st.selectbox(
            "Choose a variable:",
            df.columns
        )

        if cat_name != "Length of hospital stay (days)" and df[cat_name].dtype == "object":
            fig = plot_bar(cat_name)
            st.write(f"Data type: {df[cat_name].dtype}")
            st.plotly_chart(fig, use_container_width=True)

        elif cat_name != "Length of hospital stay (days)" and df[cat_name].dtype == "float64":
            st.write(f"Data type: {df[cat_name].dtype}")
            
            def plot_bar_2(cat_name):
                df[cat_name] = df[cat_name].astype(bool)

                fig = px.bar(
                    x= df[cat_name].value_counts().index,
                    y= df[cat_name].value_counts(),
                    title=f"Frequency of {cat_name}"
                )
                fig.update_layout(
                    xaxis_title=f"{cat_name}",
                    yaxis_title="Frequency"
                )
                return fig
            fig = plot_bar_2(cat_name)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"Data type: {df[cat_name].dtype}")
            fig = plot_hist(cat_name)
            st.plotly_chart(fig,use_container_width=True)

        #lets create Tables for Pvalues evaluation and between categorical variables 

        st.subheader("4. Relationship Analysis")
        st.write("a. Categorical relationship analysis")
        
        feat1 = st.selectbox(
            "Choose first variable:",
            df.columns
        )
        feat2 = st.selectbox(
            "Choose second variable:",
            df.columns
        )

        def chi2_test( feat1, feat2):
        # Cross-tabulation
            ct = pd.crosstab(df[feat1], df[feat2])

            # Run chi-square test
            chi2, p, dof, expected = chi2_contingency(ct)

            # Expected frequency table
            expected_df = pd.DataFrame(
                expected,
                index=ct.index,
                columns=ct.columns
            )

            # Standardized residuals (direction of relationship)
            residuals = (ct - expected) / np.sqrt(expected)

            # Flatten into a detailed result table
            result_table = pd.DataFrame({
                "Observed": ct.stack(),
                "Expected": expected_df.stack().round(2),
                "Residual": residuals.stack().round(2)
            })

            # Direction interpretation
            result_table["Direction"] = np.where(
                result_table["Residual"] > 2, "Over-represented",
                np.where(result_table["Residual"] < -2, "Under-represented", "As expected")
            )

            # Reset index for readability
            result_table = result_table.reset_index().rename(columns={"level_0": feat1, "level_1": feat2})

            # Cramér’s V (strength of association)
            n = ct.values.sum()
            k = min(ct.shape)
            cramers_v = np.sqrt(chi2 / (n * (k-1)))

            # Test summary
            summary = pd.DataFrame({
                "Chi2": [chi2],
                "p_value": [p],
                "dof": [dof],
                "Cramers_V": [cramers_v],
                "Significance": ["***" if p < 0.001 else 
                                "**" if p < 0.01 else 
                                "*" if p < 0.05 else 
                                "ns"]
            })

            return summary, result_table

        if df[feat1].dtype == "object" and df[feat2].dtype == "object":
            chi_summary, result_table = chi2_test(feat1, feat2)
            st.write("chi2 test and its p values:")
            st.dataframe(chi_summary)
            
            # Title
            st.write("## Quick Guide: Interpreting Chi-Square & Cramér's V")

            # Chi-square guide
            st.write("""
            **1.Chi-square statistic (χ²)**  
            - Tests if there is an association between two categorical variables.  
            - Large χ² → bigger difference between observed and expected counts.  

            **2. Degrees of Freedom (DoF)**  
            - `(rows-1)*(columns-1)`  
            - Used to determine significance.

            **3. p-value**  
            - `p < 0.05` → significant association, reject null hypothesis.  
            - `p ≥ 0.05` → no significant association, fail to reject null hypothesis.

            **4. Cramér's V**  
            - Measures **strength of association** (0 to 1).  
            - 0 → no association  
            - 0.1 → weak  
            - 0.3 → moderate  
            - 0.5+ → strong  
            """)

            # Small note
            st.write("*💡 Chi-square shows if variables are related; Cramér’s V shows how strong that relationship is.*")

            st.write("b. Expected frequency table analysis")
            st.dataframe(result_table)

        




            # Title
            st.write("## Guide: Interpreting Observed vs Expected Table")

            # Guide explanation
            st.write("""
            This table compares **Observed counts** to **Expected counts** for each combination of complications and treatment outcomes. Here's how to read it:

            **1. Observed vs Expected**
            - **Observed** = actual number of cases in that category  
            - **Expected** = number of cases predicted under independence (no association)  
            - Large differences suggest association.

            **2. Residuals**
            - Residual = (Observed - Expected) / √Expected (standardized difference)  
            - **Positive residual** → over-represented category (more cases than expected)  
            - **Negative residual** → under-represented category (fewer cases than expected)  

            **3. Direction**
            - Tells you if the category is over-represented or as expected.  
                    
            **Tip:** Focus on **high residuals** (absolute value > 2) to identify which categories drive the association.  
            """)

            # Optional note
            st.write("* This helps you quickly see which combinations occur more or less often than expected under independence.*")


        else:
            st.warning("Both variables must be categorical")

        st.subheader("4. Inferential Analysis")
        st.write("•	To identify underlying conditions and risk factors associated with Treatment outcomes in esophageal foreign body impaction in pediatric patients")

        # Create a copy of the dataframe to work with
        df_encoded = df.copy()

        # Initialize label encoders for each categorical variable
        le_gender = LabelEncoder()
        le_foreign_body = LabelEncoder()
        le_conditions = LabelEncoder()
        le_outcome = LabelEncoder()

        # Encode categorical variables
        df_encoded['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
        df_encoded['Foreign_body_encoded'] = le_foreign_body.fit_transform(df['Foreign body ingested'])
        df_encoded['Conditions_encoded'] = le_conditions.fit_transform(df['Predisposing Conditions'])

        # Encode outcome
        y_encoded = le_outcome.fit_transform(df['Treatment outcome'])



        st.write("## Encoding Guide")

        # Create 2 columns
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Gender")
            st.write("• Female → 0")  
            st.write("• Male → 1")  

            st.write("### Conditions")
            st.write("• Autism spectrum disorder → 0")  
            st.write("• Developmental delay → 1")  
            st.write("• None → 2")  
            st.write("• Previous ingestion → 3")  

        with col2:
            st.write("### Foreign Body")
            st.write("• Battery → 0")  
            st.write("• Bones → 1")  
            st.write("• Buttons → 2")  
            st.write("• Coin → 3")  
            st.write("• Food items → 4")  
            st.write("• Metal → 5")  
            st.write("• Pins → 6")  
            st.write("• Plastic pieces → 7")  
            st.write("• Ring → 8")  
            st.write("• Toy → 9")  

            st.write("### Outcome")
            st.write("• Full recovery → 0")  
            st.write("• Complication/sequelae → 1")  
            st.write("• Unknown/lost → 2") 

        # Prepare features - use encoded versions instead of dummies
        X = df_encoded[["Age(years)", "Gender_encoded", "Foreign_body_encoded", "Conditions_encoded"]]
        X = sm.add_constant(X)  # Add intercept

        print(f"Shape of X: {X.shape}")
        print(f"Unique outcome values: {np.unique(y_encoded)}")

        # Fit model
        model = sm.MNLogit(y_encoded, X)
        results = model.fit(maxiter=200, disp=True)  # Set disp=True to see optimization progress

        
        # Display in Streamlit with monospaced font
        st.write("## A. Full Regression Summary")
        st.code(results.summary())

        st.write(
        "The regression analysis results show that the model did not converge (converged=False) and has a low pseudo R-squared (0.10), suggesting weak explanatory power. "
        "Most predictors (age, gender, foreign body type, and conditions) are not statistically significant (p>0.05), meaning they do not strongly explain outcome differences. "
        "This indicates the current data  may not be sufficient to produce a stable model."
        "The issues lies in the size of the dataset with different categories."
        "To improve this, increase in data or other data manipulation techniques to tackle data imbalance."
    )


    #Model Training and evaluation metrics 

elif section == "2. Model Training and Evaluation":
    st.subheader("Model Training and selection")
    st.markdown("This session is to further evaluate the influence and ability of our variables to predict either the length of Hospital stay or Treatment Outcomes among Pediatric patient with foreign Bodies")
    st.info("The Data provided are insufficient to proceed with a predictive analysis. Data Manipulation techniques did not provide improvement for our model!!")


    #lets start with a regression analysis 


