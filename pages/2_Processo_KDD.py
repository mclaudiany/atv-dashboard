import streamlit as st
import pandas as pd

from etapas_kdd.KDD_mapping_features import etapa_selecao_mapeamento
from etapas_kdd.KDD_pre_processing import etapa_pre_processamento
from etapas_kdd.KDD_data_transformation import etapa_transformacao_dados
from etapas_kdd.KDD_data_mining import etapa_mineracao_dados
from etapas_kdd.KDD_result_metrics import etapa_resultado_metricas

st.set_page_config(
    page_title="Dashboard - KDD Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Pipeline KDD: Detecção e Subtipagem de Câncer de Pulmão")

st.sidebar.title("Etapas do Processo KDD")
etapa_selecionada = st.sidebar.radio(
    "Selecione a fase atual:",
    [
        "1. Seleção e Mapeamento", 
        "2. Pré-processamento e Limpeza", 
        "3. Transformação de Dados", 
        "4. Mineração de Dados", 
        "5. Avaliação de Métricas"
    ]
)

if "df" not in st.session_state:
    st.session_state.df = None
if "df_test" not in st.session_state:
    st.session_state.df_test = None
if "df_train" not in st.session_state:
    st.session_state.df_train = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "X_train_m1" not in st.session_state:
    st.session_state.X_train_m1 = None
if "X_test_m1" not in st.session_state:
    st.session_state.X_test_m1 = None
if "y_train_m1" not in st.session_state:
    st.session_state.y_train_m1 = None
if "y_test_m1" not in st.session_state:
    st.session_state.y_test_m1 = None
if "X_train_m2" not in st.session_state:
    st.session_state.X_train_m2 = None
if "X_test_m2" not in st.session_state:
    st.session_state.X_test_m2 = None
if "y_train_m2" not in st.session_state:
    st.session_state.y_train_m2 = None
if "y_test_m2" not in st.session_state:
    st.session_state.y_test_m2 = None
if "epochs" not in st.session_state:
    st.session_state.epochs = None
if "loss_function_mlp" not in st.session_state:
    st.session_state.loss_function_mlp = None
if "loss_function_kmeans" not in st.session_state:
    st.session_state.loss_function_kmeans = None
if "y_true_m1" not in st.session_state:    
    st.session_state.y_true_m1 = None
if "y_pred_m1" not in st.session_state:
    st.session_state.y_pred_m1 = None
if "y_true_m2" not in st.session_state:
    st.session_state.y_true_m2 = None
if "y_pred_m2" not in st.session_state:
    st.session_state.y_pred_m2 = None
if "dbscan_X_scaled" not in st.session_state:
    st.session_state.dbscan_X_scaled = None
if "dbscan_labels" not in st.session_state:
    st.session_state.dbscan_labels = None


if "1. Seleção e Mapeamento" in etapa_selecionada:
    df_test, df_train = etapa_selecao_mapeamento()
        
    if df_test is not None:
        st.session_state.df_test = df_test
    
    if df_train is not None:
        st.session_state.df_train = df_train
    
elif "2. Pré-processamento e Limpeza" in etapa_selecionada:
    if st.session_state.df_train is None or st.session_state.df_test is None:
        st.warning("⚠️ Dados não carregados. Por favor, execute a Etapa 1 (Seleção e Mapeamento) primeiro.")
        st.stop()
    X_train, X_test, y_train, y_test  = etapa_pre_processamento(st.session_state.df_train, st.session_state.df_test)
    
    if y_train is not None:
        st.session_state.y_train = y_train
        
    if y_test is not None:
        st.session_state.y_test = y_test
        
    if X_train is not None:
        st.session_state.df_train = pd.concat([X_train, y_train], axis=1)
        
    if y_test is not None:
        st.session_state.df_test = pd.concat([X_test, y_test], axis=1)
    
elif "3. Transformação de Dados" in etapa_selecionada:
    if st.session_state.df_train is None or st.session_state.df_test is None:
        st.warning("⚠️ Dados não carregados. Por favor, execute as Etapas 1 e 2 primeiro.")
        st.stop()
    etapa_transformacao_dados(st.session_state.df_train, st.session_state.df_test)
            
elif "4. Mineração de Dados" in etapa_selecionada:
    etapa_mineracao_dados()
    
elif "5. Avaliação de Métricas" in etapa_selecionada:
    etapa_resultado_metricas()
        
else:
    st.info("Esta etapa ainda não foi ativada. Prossiga o desenvolvimento no script.")