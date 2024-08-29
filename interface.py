import pandas as pd
import streamlit as st
import pickle
import requests
import numpy as np
from math import cos, sin
import plotly.express as px

st.header('Calcular Preço de Imóveis (Consulta Única)')

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_pickle(path):
    model = pickle.load(open(path, 'rb'))
    return model

df_data = load_data('./imoveis_cleaned.csv')

def convert_excel(df):
    return df.to_excel(index=False)

def get_lat_long(endereco):
    API_KEY = st.secrets['api_key']  # Google Maps API Key
    country = 'BR'          # country code for Brazil
    address = f'{endereco},+{country}'

    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={API_KEY}'

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            latitude = data['results'][0]['geometry']['location']['lat']
            longitude = data['results'][0]['geometry']['location']['lng']
            return latitude, longitude
        else:
            return None
    else:
        return None

def predict_price(endereco, area, quartos, vagas, banheiros, suites, is_apt, models, scaler):
    R = 6371
    latitude, longitude = get_lat_long(endereco)
    x_coord = R * cos(latitude) * cos(longitude)
    y_coord = R * cos(latitude) * sin(longitude)
    z_coord = R * sin(longitude)
    
    test = np.array([area, quartos, vagas, banheiros, suites, is_apt, x_coord, y_coord, z_coord]).reshape(1, -1)
    predicts = []
    for m in models:
        predicts.append(m.predict(scaler.transform(test))[0])
    
    return latitude, longitude,\
            {"media_algoritmos":np.mean(predicts), 
            "knn":predicts[0], 
            "xgboost":predicts[1]}

knn = load_pickle('./knn_model.pkl')
xgboost = load_pickle('./xgboost_model.pkl')
scaler = load_pickle('./scaler.pkl')


#Single place
st.text_input("Endereço do Imóvel (rua e número em Recife)", key="address")
st.number_input("Área do Imóvel (m²)", key="area", placeholder=0.0, step=0.1, min_value=0.0)
st.number_input("Número de Quartos", key="rooms", placeholder=0, min_value=0)
st.number_input("Número de Vagas de Garagem", key="parking", placeholder=0, min_value=0)
st.number_input("Número de Banheiros", key="bathrooms", placeholder=0, min_value=0)
st.number_input("Número de Suítes", key="suites", placeholder=0, min_value=0)
st.selectbox("Tipo do imóvel", ['Apartamento', 'Casa'], key="type_imovel")


button_calc = st.button('Calcular Preço')

if st.session_state.get('button') != True:
    st.session_state['button'] = button_calc

if st.session_state['button'] == True:
    address = st.session_state.address
    area = st.session_state.area
    rooms = st.session_state.rooms
    parking = st.session_state.parking
    bathrooms = st.session_state.bathrooms
    suites = st.session_state.suites
    is_apt = 1 if st.session_state.type_imovel == 'Apartamento' else 0

    latitude, longitude, price = predict_price(address, area, rooms, parking, bathrooms, suites, is_apt, [knn, xgboost], scaler)
    st.write(f'Preço estimado do imóvel:')
    for key, value in price.items():
        st.write(f'{key}: R$ {value:.2f}')
    
    st.selectbox("Atributo de Mapa de Calor", ['preco', 'area', 'quartos', 'vagas',
                                                'banheiros', 'suites', 'condominio'], key="type_mapa")
    if st.button('Gerar Mapa de Calor'):
    #st.map(pd.DataFrame({'latitude':[latitude], 'longitude':[longitude]}))
        fig1 = px.density_mapbox(df_data, lat='latitude', lon='longitude', z=st.session_state.type_mapa, radius=5,
                                    center=dict(lat=latitude, lon=longitude), zoom=12,
                                    mapbox_style='open-street-map', height= 800,
                                    color_continuous_scale='Viridis', title='Mapa de calor')
        st.plotly_chart(fig1)

        fig2 = px.scatter_mapbox(df_data, lat='latitude', lon='longitude', color=st.session_state.type_mapa,
             color_continuous_scale= [
                [0.0, "green"],
                [0.1, "green"],
                [0.1000001, "yellow"],
                [0.5, "yellow"],
                [0.50000001, "orange"],
                [0.9, "orange"],
                [0.9000001, "red"],
                [1, "red"]],
                        opacity = 0.5, center=dict(lat=latitude, lon=longitude), zoom=12,
                                    mapbox_style='open-street-map', height= 800
                        )
        st.plotly_chart(fig2)
        st.session_state['button'] == False
    
#Multiple places from csv with st.file_uploader
st.header('Calcular Preço de Imóveis em Lote a partir de arquivo Excel')
st.text('Excel com colunas: endereco, area, quartos, vagas, banheiros, suites, type_imovel ("Apartamento" ou "Casa")')


uploaded_file = st.file_uploader("Escolha um arquivo Excel", type=["xlsx", "xls"])
if st.button('Previsão em Lote'):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
        df.columns = ['endereco', 'area', 'quartos', 'vagas', 'banheiros', 'suites', 'tipo_imovel']
        df['latitude'] = 0
        df['longitude'] = 0
        df['preco_medio'] = 0
        df['preco_knn'] = 0
        df['preco_xgboost'] = 0
        for i, row in df.iterrows():
            address = row.iloc[0]
            area = row.iloc[1]
            rooms = row.iloc[2]
            parking = row.iloc[3]
            bathrooms = row.iloc[4]
            suites = row.iloc[5]
            is_apt = 1 if row.iloc[6] == 'Apartamento' else 0
            latitude, longitude, price = predict_price(address, area, rooms, parking, bathrooms, suites, is_apt, [knn, xgboost], scaler)
            df.loc[i, 'latitude'] = latitude
            df.loc[i, 'longitude'] = longitude
            df.loc[i, 'preco_medio'] = price['media_algoritmos']
            df.loc[i, 'preco_knn'] = price['knn']
            df.loc[i, 'preco_xgboost'] = price['xgboost']
            
        st.write(df)