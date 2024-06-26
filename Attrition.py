import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Warranty Contract Attrition Analysis")

st.write("""
This Streamlit app analyzes a dataset of warranty contracts to identify patterns of attrition.
Using the Apriori algorithm, we can find association rules that help us understand the relationships
between various factors influencing contract cancellations.
""")

# Use st.cache_data to cache data loading
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

raw_data = load_data('warranty-contract-attrition.csv')

st.subheader("Dataset Preview")
st.dataframe(raw_data.head())

# One-hot encode the data
one_hot_encoded_data = pd.get_dummies(raw_data)

# Apply Apriori algorithm
frequent_itemsets = apriori(one_hot_encoded_data, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# Convert sets to strings for display
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Explain the features
st.subheader("Explanation of Metrics in Association Rules")
st.write("""
- **Antecedents**: Items or conditions on the left-hand side of the rule.
- **Consequents**: Items or conditions on the right-hand side of the rule.
- **Antecedent Support**: Proportion of transactions that contain the antecedents.
- **Consequent Support**: Proportion of transactions that contain the consequents.
- **Support**: Proportion of transactions that contain both the antecedents and consequents.
- **Confidence**: Likelihood that the consequent is also in the transaction given that the antecedent is in it.
- **Lift**: Measure of how much more often the antecedent and consequent appear together than expected if they were independent.
- **Leverage**: Difference between the observed frequency of the antecedent and consequent appearing together and the frequency if they were independent.
- **Conviction**: Measure of the implication strength, the higher the value, the stronger the implication.
- **Zhang's Metric**: Combines measures of support, confidence, and lift, providing a more balanced metric for rule interestingness.
""")

# Display the rules
st.subheader("Association Rules")
st.dataframe(rules.head())

# Sidebar filters
st.sidebar.subheader("Filter Rules")
filter_by = st.sidebar.radio("Filter by:", ['antecedents', 'consequents'])

# Extract unique items for filter
unique_items = pd.unique(rules[filter_by].str.split(', ', expand=True).stack())
selected_item = st.sidebar.selectbox(f"Select {filter_by[:-1]} item:", sorted(unique_items))
min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.2)

# Use st.cache_data to cache the filtering function
@st.cache_data
def filter_rules(rules, filter_by, selected_item, min_confidence):
    filtered_rules = rules[rules[filter_by].str.contains(selected_item)]
    return filtered_rules[filtered_rules['confidence'] >= min_confidence]

filtered_rules = filter_rules(rules, filter_by, selected_item, min_confidence)

st.subheader(f"Filtered Association Rules ({filter_by} contains '{selected_item}')")
st.dataframe(filtered_rules)

# Convert to CSV
csv = filtered_rules.to_csv(index=False)
st.download_button(label="Download Filtered Rules as CSV", data=csv, file_name='filtered_rules.csv', mime='text/csv')
