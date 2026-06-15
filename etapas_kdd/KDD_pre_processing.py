import pandas as pd
import streamlit as st

def etapa_pre_processamento(df_train, df_test):
    st.header("Pré-processamento e Limpeza de Dados")
    st.subheader("Diagnóstico de Qualidade e Tratamento de Ruídos")

    st.markdown("Foi realizado o tratamento dos dados para detecção de inconsistências (Dados Brutos)")
    
    X_train, X_test, y_train, y_test, df_filtrado_nulos = tratar_dados_nulos(df_train, df_test)

    montar_kpis(X_train, X_test)

    if df_filtrado_nulos.empty:
        st.info("Mapeamento das features realizado. As features monitoradas estão 100% preenchidas no dataset.")
    
    return X_train, X_test, y_train, y_test 

def montar_kpis(X_train, X_test):
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(label="Amostras no Treino", value=f"{X_train.shape[0]}")
    with col_kpi2:
        st.metric(label="Amostras no Teste", value=f"{X_test.shape[0]}")
    with col_kpi3:
        st.metric(label="Atributos Monitorados ", value=f"{X_train.shape[1]} features")

def tratar_dados_nulos(df_train, df_test):
    target_cols = ["cancer_presence", "cancer_subtype"]
    
    X_train = df_train.drop(columns=[col for col in target_cols if col in df_train.columns])
    X_test = df_test.drop(columns=[col for col in target_cols if col in df_test.columns])

    y_train = df_train[[col for col in target_cols if col in df_train.columns]]
    y_test = df_test[[col for col in target_cols if col in df_test.columns]]

    nulos_treino = X_train.isnull().sum()
    nulos_teste = X_test.isnull().sum()
    
    df_diagnostico = pd.DataFrame({
        "Nulos no Treino": nulos_treino,
        "% Nulos Treino": (nulos_treino / len(X_train)) * 100 if len(X_train) > 0 else 0,
        "Nulos no Teste": nulos_teste,
        "% Nulos Teste": (nulos_teste / len(X_test)) * 100 if len(X_test) > 0 else 0
    }).sort_values(by="Nulos no Treino", ascending=False)
    
    df_filtrado_nulos = df_diagnostico[
        (df_diagnostico["Nulos no Treino"] > 0) | 
        (df_diagnostico["Nulos no Teste"] > 0)
    ]

    total_nulos_encontrados = df_train.isnull().sum().sum() + df_test.isnull().sum().sum()
    
    if total_nulos_encontrados > 0:
        st.warning(f"Foram detectados **{total_nulos_encontrados}** campos nulos/ausentes no dataset.")
    
    return X_train,X_test,y_train,y_test,df_filtrado_nulos   
        