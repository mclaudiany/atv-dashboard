import streamlit as st
import pandas as pd

from secoes_eda.secao_geral import secao_introducao_visao_geral, secao_volumetria, secao_qualidade_dados, secao_balanceamento, secao_justificativa_metodologica
from secoes_eda.secao_pp1 import secao_analise_densidade_idade_pp1, secao_analise_distibuicao_genero_pp1, secao_analise_atributos_nodulo_pp1
from secoes_eda.secao_pp2 import   secao_analise_distribuicao_idade_subtipo_pp2, secao_analise_distribuicao_genero_subtipo_pp2, secao_analise_correlacao_atributos_quantitativos_pp2, secao_identificacao_outliers_dbscan_pp2

st.set_page_config(
    page_title="Dashboard - Análise Exploratória de Dados (EDA)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Análise Exploratória de Dados (EDA)")

@st.cache_data
def carregar_dataframe(path):
    df = pd.read_csv(path)
    return df

if "df" not in st.session_state:
    st.session_state.df = None

df_treino = carregar_dataframe('./data/train.csv')
df_teste = carregar_dataframe('./data/test.csv')

if df_treino is not None:
    st.session_state.df = df_treino
  
tab_geral, tab_pp1, tab_pp2 = st.tabs([
    "1. Visão Geral e Balanceamento", 
    "2. Pergunta 1: Triagem Inicial", 
    "3. Pergunta 2: Subtipagem Tumoral"
])

with tab_geral:   
    secao_introducao_visao_geral()
    
    secao_volumetria(df_treino, df_teste)  
    
    secao_qualidade_dados(df_treino, df_teste)
    
    secao_balanceamento(df_treino, df_teste)
    
    secao_justificativa_metodologica(df_treino, df_teste)

with tab_pp1:
    st.markdown("## Pergunta 1: Como identificar na triagem inicial a existência de câncer?")
    
    secao_analise_densidade_idade_pp1(df_treino, df_teste)
    
    secao_analise_distibuicao_genero_pp1(df_treino, df_teste)
    
    secao_analise_atributos_nodulo_pp1(df_treino, df_teste)

with tab_pp2:
    st.markdown("## Pergunta 2: Uma vez identificada a patologia, como mapear o seu subtipo?")
    
    secao_analise_distribuicao_idade_subtipo_pp2(df_treino, df_teste)
    
    secao_analise_distribuicao_genero_subtipo_pp2(df_treino, df_teste)
    
    secao_analise_correlacao_atributos_quantitativos_pp2(df_treino, df_teste)
    
    secao_identificacao_outliers_dbscan_pp2(df_treino, df_teste)
    

        
        
