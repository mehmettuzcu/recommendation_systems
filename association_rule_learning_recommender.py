#####  Data Understanding

# Importing Libraries
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


# Importing Data

df = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")


##### Descriptive Statistics

df.shape  # Dimension of dataframe

df.dtypes  # Data type of each variable

df.info  # Print a concise summary of a DataFrame

df.head()  # First 5 observations of dataframe

df.tail()  # Last 5 observations of dataframe



def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


##### Data Preparation
########################## Task-1 ##########################
df.dropna(inplace=True)  # Remove missing observations from the data set

df = df[~df["Invoice"].str.contains("C", na=False)]  # Delete operation if it starts with C in "Invoice".

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# Functions required to delete outliers
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.head()

####### Invoice-Product Matrix

########################## Task-2 ##########################

df_ge = df[df["Country"] == "Germany"]

df_ge.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"})


df_ge.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5].head()


# functionalization of transactions
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)



ge_inv_pro_df = create_invoice_product_df(df_ge, id=True)
ge_inv_pro_df.head()

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    return product_name



########################## Task-3 ##########################

# User 1 product id: 21987
# User 1 product id: 23235
# User 1 product id: 22747

l = [21987, 23235, 22747]
for index, product in enumerate(l):
    print("Product Name " + str(index) + ": " + str(check_id(df_ge, product)))




########################## Task-4 ##########################

############# Enforcement of Association Rules

# Possibilities of all possible product combinations
frequent_itemsets = apriori(ge_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()


# Enforcement of Association Rules:
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head(50)


# Product recommendation for users in the cart to be done
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

check_id(df, 21987)
arl_recommender(rules, 21987, 3)


########################## Task-5 ##########################

# Recommended products for all users according to their product id
for product in l:
    print("Product Name  " + str(product) + ": " + str(check_id(df_ge, product)) + " Recommended Product: " + str([check_id(df_ge, i) for i in arl_recommender(rules, product, 3)]))


# with list comprehension
["Product Name  " + str(product) + ": " + str(check_id(df_ge, product)) + " Recommended Product: " + str([check_id(df_ge, i) for i in arl_recommender(rules, product, 3)]) for product in l]