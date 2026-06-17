import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

def etapa_transformacao_dados(df_train):
    st.header("Transformação de Dados")
    st.subheader("Encoding e Padronização Estrutural")
    st.markdown("A transformação dos dados mapeia as variáveis qualitativas para formatos numéricos e reescalona os atributos contínuos.")

    st.subheader("Mapeamento de Categorias e Encodings")
    mapear_categorias_encodings()

    st.subheader("Escalonamento por Padronização Z-score")
    escalonar_por_zcore(df_train)
    
    st.subheader("Estruturação das Features")
    estruturar_features()

def estruturar_features():
    if "X_train_m1" in st.session_state and st.session_state.X_train_m1 is not None:
        st.markdown("Estrutura das features de treino prontas para alimentar os tensores da rede neural:")
        st.dataframe(st.session_state.X_train_m1.head(5), width='stretch')
    else:
        X_train_m1, X_test_m1, y_train_m1, y_test_m1 = executar_transformacao_kdd(
            st.session_state.df_train, 
            st.session_state.df_test, 
            escopo='presenca'
        )
        
        st.session_state.X_train_m1 = X_train_m1
        st.session_state.X_test_m1 = X_test_m1
        st.session_state.y_train_m1 = y_train_m1
        st.session_state.y_test_m1 = y_test_m1
        
        st.success("O pipeline KDD foi executado! Matrizes geradas e escalonadas via Z-score com sucesso.")
        st.rerun()

def escalonar_por_zcore(df_train):
    num_continuas = ['nodule_size_mm', 'HU_mean', 'HU_std', 'PET_SUVmax', 'PET_SUVmean', 'patient_age', 'PD-L1_expression_level', 'tumor_mutational_burden']
    num_existentes = [col for col in num_continuas if col in df_train.columns]

    if num_existentes:
        scaler = StandardScaler()
        
        df_fill = df_train[num_existentes].fillna(df_train[num_existentes].median(numeric_only=True))
        dados_escalados_train = scaler.fit_transform(df_fill)
        df_depois_num = pd.DataFrame(dados_escalados_train, columns=num_existentes)

        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            fig_antes = go.Figure()
            fig_antes.add_trace(go.Violin(y=df_train['nodule_size_mm'].dropna(), name="Tamanho do Nódulo (mm)", line_color="#2B6CB0", box_visible=True))
            fig_antes.add_trace(go.Violin(y=df_train['HU_mean'].dropna(), name="Densidade (HU_mean)", line_color="#E75355", box_visible=True))
            
            if 'PD-L1_expression_level' in df_train.columns:
                fig_antes.add_trace(go.Violin(y=df_train['PD-L1_expression_level'].dropna(), name="Expressão PD-L1 (%)", line_color="#7BB8F2", box_visible=True))
            
            fig_antes.update_layout(title="Distribuição Original", height=350, margin=dict(t=40, b=40, l=20, r=20))
            st.plotly_chart(fig_antes, width='stretch')

        with col_graph2:
            fig_depois = go.Figure()
            fig_depois.add_trace(go.Violin(y=df_depois_num['nodule_size_mm'], name="Tamanho Nódulo (Z)", line_color="#2B6CB0", box_visible=True))
            fig_depois.add_trace(go.Violin(y=df_depois_num['HU_mean'], name="Densidade (Z)", line_color="#E75355", box_visible=True))
            
            if 'PD-L1_expression_level' in df_depois_num.columns:
                fig_depois.add_trace(go.Violin(y=df_depois_num['PD-L1_expression_level'], name="Expressão PD-L1 (Z)", line_color="#7BB8F2", box_visible=True))
            
            fig_depois.update_layout(title="Distribuição Após Z-SCORE", height=350, margin=dict(t=40, b=40, l=20, r=20))
            st.plotly_chart(fig_depois, width='stretch')

def mapear_categorias_encodings():
    col_enc1, col_enc2 = st.columns(2)
    with col_enc1:
        st.markdown("**One-Hot Encoding (Variáveis Nominais):**")
        st.caption("Aplicado na coluna qualitativa `smoking_history` (Never, Former, Current).")
        
        df_demo_enc = pd.DataFrame({"Antes: smoking_history": ["Never", "Former", "Current"]})
        df_demo_res = pd.DataFrame({
            "Depois: smoking_history_Current": [0, 0, 1],
            "Depois: smoking_history_Former": [0, 1, 0],
            "Depois: smoking_history_Never": [1, 0, 0]
        })
        st.dataframe(pd.concat([df_demo_enc, df_demo_res], axis=1), width='stretch', hide_index=True)

    with col_enc2:
        st.markdown("**Mapeamento Binário:**")
        st.caption("Variáveis booleanas: `patient_gender` (Male ➔ 0, Female ➔ 1) e `family_history` (0 ➔ Não, 1 ➔ Sim).")
        
        df_demo_bin = pd.DataFrame({
            "Antes: patient_gender": ["Male", "Female", "Male"],
            "Depois: patient_gender (Mapeado)": [0, 1, 0]
        })
        df_demo_bin["Antes: patient_gender"] = df_demo_bin["Antes: patient_gender"].astype(str)
        st.dataframe(df_demo_bin, width='stretch', hide_index=True)


def executar_transformacao_kdd(df_train, df_test, escopo='presenca'):
    features_radiomicas = ['nodule_size_mm', 'nodule_texture', 'HU_mean', 'HU_std', 
                           'GLCM_contrast', 'GLCM_correlation', 'PET_SUVmax', 'PET_SUVmean']
    features_clinicas = ['patient_age', 'patient_gender', 'smoking_history', 'family_history']
    biomarcadores = ['EGFR_mutation_status', 'KRAS_mutation_status', 'ALK_fusion_status', 
                     'PD-L1_expression_level', 'tumor_mutational_burden']
    
    if escopo == 'presenca':
        features_para_treino = features_radiomicas + features_clinicas
        target_coluna = 'cancer_presence'
        
        X_train = df_train[features_para_treino].copy()
        X_test = df_test[features_para_treino].copy()
        y_train = df_train[target_coluna].copy()
        y_test = df_test[target_coluna].copy()
        
    elif escopo == 'subtipo':
        df_train_filtrado = df_train[df_train['cancer_presence'] == 1]
        df_test_filtrado = df_test[df_test['cancer_presence'] == 1]
        
        features_para_treino = features_radiomicas + features_clinicas + biomarcadores
        target_coluna = 'cancer_subtype'
        
        X_train = df_train_filtrado[features_para_treino].copy()
        X_test = df_test_filtrado[features_para_treino].copy()
        y_train = df_train_filtrado[target_coluna].copy()
        y_test = df_test_filtrado[target_coluna].copy()

    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test = X_test.fillna(X_train.median(numeric_only=True))

    X_train = pd.get_dummies(X_train, columns=['smoking_history'], drop_first=False)
    X_test = pd.get_dummies(X_test, columns=['smoking_history'], drop_first=False)
    
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    num_continuas = ['nodule_size_mm', 'HU_mean', 'HU_std', 'PET_SUVmax', 'PET_SUVmean', 
                    'patient_age', 'PD-L1_expression_level', 'tumor_mutational_burden']
    
    cols_para_escalonar = [col for col in num_continuas if col in X_train.columns]
    
    scaler = StandardScaler()
    X_train[cols_para_escalonar] = scaler.fit_transform(X_train[cols_para_escalonar])
    X_test[cols_para_escalonar] = scaler.transform(X_test[cols_para_escalonar])

    return X_train, X_test, y_train, y_test