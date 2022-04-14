# Import
import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import datetime

import requests
import json
import os


# Initial state of website
st.set_page_config(
    page_title="Forecast for sales",
    page_icon="💰",
    layout="wide")
# css style
CSS = """
h1 {
    color: orange;
}

.stApp {
    background-color: #2C2E43;
}
"""

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
# -----------
# Request API
# -----------

# pickup_datetime = f"{date} {time}"

date_predict = '2016-01-01'
store_nbr_predict = 1
family_predict = 'BREAD/BAKERY' # BREAD%2FBAKERY
# family_predict = '' # BREAD%2FBAKERY

dict_predict_store = {
    'date': date_predict,
    'store_nbr': store_nbr_predict,
    'family': family_predict
}

# Example request API
# https://favorita-cquq2ssw6q-ew.a.run.app/predict?date=2016-01-01&store_nbr=1&family=BREAD%2FBAKERY

# Call API using `requests`
# Retrieve the prediction from the **JSON** returned by API...
# display the prediction
@st.cache
def get_predict():

    # my_url = 'https://docker-tfm-ipbs6r3hdq-ew.a.run.app/predict'
    # url_wagon = 'https://taxifare.lewagon.ai/predict'
    url_forecast = 'https://favorita-cquq2ssw6q-ew.a.run.app/predict'
    response = requests.get(url_forecast, params=dict_predict_store) # my_url
    sales_fare = response.json()
    return sales_fare

sales_fare = get_predict()
'''Here a prediction'''
# -----------
# Request API
# -----------

# top 10
# sales_fare['predicted_sales-per_item']['data']

sales_fare

predicted_sales_per_item = json.loads(sales_fare['predicted_sales-per_item'])
# st.write(predicted_sales_per_item)
confidence_int = json.loads(sales_fare['confidence_int']) # confidence_int
# st.write(confidence_int)
family_forecast = json.loads(sales_fare['family_forecast']) # forecast
# st.write(family_forecast)


# Generate DataFrame
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_cached_data():
    # return pd.read_csv('raw_data/train_all_table.csv', nrows=10000).drop(columns='Unnamed: 0')
    # work with 1 store
    # return pd.read_csv('forecasting_sales_front/data/preprocessed_sales_grouped_21.csv')
    return pd.read_csv('forecasting_sales_front/data/preprocessed_sales_grouped_1.csv')

df = get_cached_data()
df['date'] = pd.to_datetime(df['date'])

'''
# Prévisions des ventes

## Gestion d'inventaires
---
'''



# check actual month (vs previous month)
# compare sum 2 family_sales
# show with deficit or benefit


def inventory_unit(sb_month_unit, sb_year_unit):
    df_present = df.loc[(df['date'].dt.year == sb_year_unit) & (df['date'].dt.month == sb_month_unit)]
    result_actual = df_present['family_sales'].sum()

    if sb_month_unit == 1:
        df_past = df.loc[(df['date'].dt.year == sb_year_unit - 1) &
                        (df['date'].dt.month == 12)]
    else:
        df_past = df.loc[(df['date'].dt.year == sb_year_unit) &
                        (df['date'].dt.month == sb_month_unit - 1)]
    # result_past = result_actual - df_past['family_sales'].sum()
    result_past = df_past['family_sales'].sum()


    return result_actual, result_past


col1, col2 = st.columns(2)
with col1:
    date_unit = st.date_input('Chosir la période', datetime.date(2015, 8, 1))

    sb_month_unit = date_unit.month
    sb_year_unit = date_unit.year

    # sb_month_unit = st.selectbox('Month Unit', range(min(df['date'].dt.month),
    #                                                 max(df['date'].dt.month)))

# with col2:
    # sb_year_unit = st.selectbox('Year Unit', range(min(df['date'].dt.year),
    #                                                 max(df['date'].dt.year)))



with col2:
    su_actual_month, su_past_month = inventory_unit(sb_month_unit, sb_year_unit)
    su_past_month = -(100 - (su_actual_month * 100 / su_past_month))

    # Sales Units (With + or -) %
    try:
        st.metric("Unités de ventes", f"{round(su_actual_month)}", f"{round(su_past_month)}%")
    except:
        st.metric("Unités de ventes", f"{su_actual_month}", f"{su_past_month}%")
    # st.metric("Sales Units", f"{round(su_actual_month)}", f"{round(su_past_month)}%")

    # # In Progress - need stock
    # # Inventory Units (With + or -) %
    # st.metric("Inventory Units", "121.10", "0.46%")



col_left, col_right = st.columns(2)

expander_need_stock = col_left.expander(label='Produits nécessaires à restock')
expander_avail_stock = col_left.expander(label='Stocks disponibles (Sur 1 famille)')
expander_top_ten = col_right.expander(label='Top 10 des ventes (Sur 1 famille)')
expander_predict_sales = col_right.expander(label='Prévisions des ventes')

with col_left:
    '### Produits nécessaires à restock (Sur 1 boutique)'

    # color not functionnal
    # def _color_red_or_green(val):
    #     color = 'red' if val < 10 else 'green'
    #     return 'color: %s' % color

    # Select store - see after
    # option = st.selectbox('Select a line to filter', df['store_nbr'].unique())
    # df_store = df[df['store_nbr'] == option]
    df_store = df

    df_store = df_store[['family', 'family_sales']]\
                        .groupby(by='family').sum()\
                        .sort_values('family_sales', ascending=False)
    df_store['alert'] = (df_store['family_sales'] <= 0)

    # add colors
    # df_store.style.apply(_color_red_or_green,
    #                      subset='family_sales', axis=1)
    # By store
    st.dataframe(df_store[['alert', 'family_sales']])

    if st.button('Commander'):
        st.write('Commande effectué')
    else:
        st.write('Commande non effectué')


    # too weight
    # def stackbarplot(df):
    #     fig = px.bar(df, y="family", x="family_sales", color='item_nbr' ,title="Prod à restock")
    #     fig.update_layout(paper_bgcolor='#B2B1B9')
    #     return fig

    # st.plotly_chart(stackbarplot(df))


    '### Stocks disponibles (Sur 1 famille)'

    '''

    'Here, a selecbox of family'

    Graphe des stocks disponibles
    Need item_nbr if check 1 family...
    '''
    # sb_family_stock = st.selectbox('Family', df['family'].unique())

    # df_family = df[df['family'] == sb_family_stock]
    # df_family = df[df['family'] == sb_family_stock]
    # st.plotly_chart(px.bar(df_family,x='unit_sales' y='item_nbr'))


with col_right:
    '### Top 10 des ventes (Sur 1 famille)'
    # add date to choose
    sb_year_top_10 = st.selectbox('Year',
                            range(min(df['date'].dt.year),
                                max(df['date'].dt.year)))

    # Later to select meat, chicken, beef, etc.
    # st.multiselect(label="Sélectionner vos produits", options=df['item_nbr'].columns.tolist(), default=["alcohol","malic_acid"])

    def show_top_10(df, sb_year_top_10):
        df_in_date = df[df['date'].dt.year == sb_year_top_10]
        df_top_10 = df_in_date[['family', 'family_sales']].groupby(by='family')\
                            .sum()\
                            .sort_values('family_sales', ascending=False).head(10)
        return df_top_10


    st.dataframe(show_top_10(df, sb_year_top_10))

    def barplot_top10(df):
        # fig = px.bar(df, x='family_sales', y=df.index, color='item_nbr')
        fig = px.bar(df, x='family_sales', y=df.index)

        fig.update_layout(paper_bgcolor='#B2B1B9')
        return fig

    st.plotly_chart(barplot_top10(show_top_10(df, sb_year_top_10)))

    '### Prévisions des ventes'
    # Plot with confidence interval - start

    'Un graphe avec une intervalle de confiance'

    # -------------
    # No need maybe
    # -------------

    # # Forecast
    # forecast, std_err, confidence_int = arima.forecast(len(test), alpha=0.05)  # 95% confidence

    # -------------
    # No need maybe
    # -------------

    # forecast, std_err, confidence_int = {'predicted_sales': '{"columns":["item_nbr","forecast_product"],"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],"data":[[103665,118.1805681175],[153239,12.0317315362],[153395,45.4006592073],[153398,119.9685781917],[165718,63.6514990944],[215370,111.586895726],[253103,0.0],[265279,71.2477012761],[269084,59.2145852272],[302952,67.2229645917],[310644,634.6395670119],[310647,46.7342347702],[311994,874.0568623078],[312113,71.9110566686],[315473,0.0],[315474,111.4320607333],[359913,124.2214658545],[360313,98.6543577846],[360314,217.5643155294],[402299,0.0]]}', 'confidence_int': {'6602.960957236907': 15542.673115061367, '5626.846602575789': 14566.55876040025, '6348.378908161257': 15288.091065985718, '6250.10062934992': 15189.812787174382, '5641.551794978621': 14581.26395280308, '7109.3697105270085': 16049.08186835147, '6463.915887184812': 15403.62804500927, '1248.0240571079748': 10187.736214932436, '4936.72242802496': 13876.43458584942, '5176.752296871691': 14116.46445469607, '6850.997021333551': 15790.70917915793, '5552.547742131246': 14492.259899955625, '5399.764680278498': 15199.011466229273, '5381.514230917039': 15180.761016867813, '5753.044452900291': 15552.291238851065, '5366.474603953162': 15165.721389903936, '4633.559635709262': 14432.806421660036, '5797.710271910317': 15596.957057861091, '5840.027843573505': 15639.27462952428, '3276.965667413988': 13076.212453364762}, 'family_predictions': '{"columns":[0],"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],"data":[[11072.8170361491],[10096.702681488],[10818.2349870735],[10719.9567082622],[10111.4078738909],[11579.2257894392],[10933.771966097],[5717.8801360202],[9406.5785069372],[9646.6083757839],[11320.8531002457],[10022.4038210434],[10299.3880732539],[10281.1376238924],[10652.6678458757],[10266.0979969285],[9533.1830286846],[10697.3336648857],[10739.6512365489],[8176.5890603894]]}'}


    # Code copied in lesson
    def plot_forecast(fc, train, test, upper=None, lower=None):
        is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
        # Prepare plot series
        fc_series = pd.Series(fc, index=test.index)
        lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
        upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

        # Plot
        plt.figure(figsize=(10,4), dpi=100)
        plt.plot(train, label='prediction', color='black')
        plt.plot(test, label='reel', color='black', ls='--')
        plt.plot(fc_series, label='forecast', color='orange')
        if is_confidence_int:
            plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
        plt.title('Prédictions vs Réel')
        plt.legend(loc='upper left', fontsize=8);

    # create df_store_1 for test function predict for BREAD/BAKERY
    df_store_1 = pd.read_csv('raw_data/preprocessed_sales_grouped_1.csv')
    df_store_1 = df_store_1[df_store_1['family'] == 'BREAD/BAKERY']
    df_store_1['date'] = pd.to_datetime(df_store_1['date'])

    # test df about BREAD/BAKERY

    # Prepare train and test
    mask_train = (df['date'] >= '2013-01-01') & (df['date'] <= '2015-12-31')
    train_store_1 = df_store_1.loc[df_store_1['family'] == 'BREAD/BAKERY']\
                                .loc[mask_train] # 2013 -> 2015
    test_store_1 = df_store_1.loc[df_store_1['family'] == 'BREAD/BAKERY']\
                                .loc[df_store_1['date'].dt.year == 2016] # 2016 ->

    # confidence_int[:,0]
    # confidence_int[:,1]

    # Plot with confidence interval
    # plot_forecast(forecast, train_store_1, test_store_1, confidence_int[:,0], confidence_int[:,1])
    plot_forecast(family_forecast,
                  train_store_1, test_store_1,
                  [row[0] for row in confidence_int],
                  [row[1] for row in confidence_int])



expander_df = st.expander(label='DataFrame')
with expander_df:

    # df = df[df['date'].dt.year == 2017]
    option_head = st.slider('head : ', 1, 1000, 5)
    st.write(df.head(option_head))

st.write('Min date: ', min(df['date']))
st.write('Max date: ', max(df['date']))
