import pandas as pd
import numpy as np
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Gera dados de vendas simulando maior volume em dezembro
def gerar_dados_vendas():
    datas = pd.date_range(end=datetime.today(), periods=1460)
    vendas = []

    for data in datas:
        base = 100 + np.random.normal(0, 10)
        if data.month == 5:
            base *= 1.4  # aumento sazonal em dezembro
        vendas.append(max(0, base))  # evitar vendas negativas

    df = pd.DataFrame({'data': datas, 'vendas': vendas})
    df.to_csv("vendas.csv", index=False)
    return df

# Gera dados se não existir
try:
    df = pd.read_csv("vendas.csv")
except FileNotFoundError:
    df = gerar_dados_vendas()

# Streamlit interface
st.title("📦 Previsão de Demanda com Sazonalidade")
st.write("Sistema de Supply Chain com previsão baseada em dados históricos")

# Mostra dados
st.subheader("📊 Dados de Vendas")
st.line_chart(df.set_index("data")["vendas"])

# Preparar para o Prophet
df_prophet = df.rename(columns={"data": "ds", "vendas": "y"})

# Criar e treinar modelo
modelo = Prophet()
modelo.fit(df_prophet)

# Prever próximos 30 dias
futuro = modelo.make_future_dataframe(periods=30)
previsao = modelo.predict(futuro)

# Gráfico da previsão
st.subheader("📈 Previsão de Vendas (próximos 30 dias)")
fig = modelo.plot(previsao)
st.pyplot(fig)

# Tabela com a previsão
st.subheader("📋 Tabela de Previsão")
st.dataframe(previsao[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30).round(2))
