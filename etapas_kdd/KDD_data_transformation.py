import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from imblearn.over_sampling import SMOTE

def etapa_transformacao_dados(df_train, df_test):
    st.header("Transformação de Dados")
    st.subheader("Encoding e Padronização Estrutural")
    st.markdown("A transformação dos dados mapeia as variáveis qualitativas para formatos numéricos e reescalona os atributos contínuos.")

    st.subheader("Mapeamento de Categorias e Encodings")
    mapear_categorias_encodings()
    
    st.markdown("---")
    
    st.subheader("Escalonamento por Padronização Z-score")
    df_train_scaled = escalonar_por_zcore(df_train)
    
    st.markdown("---")
    
    st.subheader("Estruturação das Features - Modelo 1")
    estruturar_features(df_train_scaled, df_test, modelo=1)
    
    st.markdown("---")
    
    st.subheader("Balanceamento de Classes via SMOTE")
    df_train_m2 = balancear_smote_multiclasse(df_train_scaled)
    
    st.markdown("---")
    
    st.subheader("Estruturação das Features - Modelo 2")
    estruturar_features(df_train_m2, df_test, modelo=2)

def estruturar_features(df_final_treino, df_teste, modelo=1):
    features_continuas = ['nodule_size_mm', 'HU_mean', 'HU_std', 'PET_SUVmax', 'PET_SUVmean', 'patient_age', 'PD-L1_expression_level', 'tumor_mutational_burden']
    features = [col for col in features_continuas if col in df_final_treino.columns]
    
    if modelo == 1:
        st.markdown("#### Estruturação de Tensores - Modelo 1 (Detecção Binária)")
        
        X_train = df_final_treino[features]
        y_train = df_final_treino['cancer_presence']
        
        X_test = df_teste[features]
        y_test = df_teste['cancer_presence']
        
        st.session_state.X_train_m1 = X_train
        st.session_state.y_train_m1 = y_train
        st.session_state.X_test_m1 = X_test
        st.session_state.y_test_m1 = y_test
        
        st.dataframe(X_train.head(5), width='stretch')
        st.success(f"Tensores do Modelo 1 estruturados: {X_train.shape[0]} amostras de treino.")
    elif modelo == 2:
        st.markdown("#### Estruturação de Tensores - Modelo 2 (Subtipos Histológicos)")
        
        X_train = df_final_treino[features]
        y_train = df_final_treino['cancer_subtype']
        
        df_teste_cancer = df_teste[df_teste['cancer_presence'] == 1]
        X_test = df_teste_cancer[features]
        y_test = df_teste_cancer['cancer_subtype']
        
        st.session_state.X_train_m2 = X_train
        st.session_state.y_train_m2 = y_train
        st.session_state.X_test_m2 = X_test
        st.session_state.y_test_m2 = y_test
        
        st.dataframe(X_train.head(3), width='stretch')
        st.success(f"Tensores do Modelo 2 (SMOTE) estruturados: {X_train.shape[0]} amostras de treino balanceadas.")
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
            
        df_train_escalado = df_train.copy()
        df_train_escalado[num_existentes] = df_fill
        df_train_escalado[num_existentes] = df_depois_num.values
        
        return df_train_escalado

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

def balancear_smote_multiclasse(df_train_scaled):
    num_continuas = ['nodule_size_mm', 'HU_mean', 'HU_std', 'PET_SUVmax', 'PET_SUVmean', 'patient_age', 'PD-L1_expression_level', 'tumor_mutational_burden']
    
    features_existentes = [col for col in num_continuas if col in df_train_scaled.columns]
    target_col = 'cancer_subtype'

    if target_col not in df_train_scaled.columns:
        st.warning(f"Coluna alvo '{target_col}' não encontrada para o SMOTE. Pulando balanceamento.")
        return df_train_scaled
    
    df_filtrado = df_train_scaled[df_train_scaled['cancer_presence'] == 1].copy() if 'cancer_presence' in df_train_scaled.columns else df_train_scaled.copy()
    df_filtrado = df_filtrado.dropna(subset=[target_col])
    
    X = df_filtrado[features_existentes]
    y = df_filtrado[target_col]
    
    if len(y.value_counts()) < 2:
        st.warning("Classes insuficientes no subtipo de câncer para aplicar o SMOTE.")
        return df_train_scaled

    smote = SMOTE(k_neighbors=3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=features_existentes)
    df_resampled[target_col] = y_resampled
    
    if 'cancer_presence' in df_train_scaled.columns:
        df_resampled['cancer_presence'] = 1
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribuição Original (Real)")
        contagem_antes = y.value_counts().reset_index()
        contagem_antes.columns = [target_col, 'Qtd Amostras']
        fig_antes = px.bar(contagem_antes, x=target_col, y='Qtd Amostras', color=target_col, template="plotly_white")
        st.plotly_chart(fig_antes, width='stretch')
        
    with col2:
        st.markdown("#### Distribuição Sintética (SMOTE)")
        contagem_depois = pd.Series(y_resampled).value_counts().reset_index()
        contagem_depois.columns = [target_col, 'Qtd Amostras']
        fig_depois = px.bar(contagem_depois, x=target_col, y='Qtd Amostras', color=target_col, template="plotly_white")
        st.plotly_chart(fig_depois, width='stretch')

    st.success(f"**SMOTE Concluído:** A base do Modelo 2 foi balanceada. Amostras expandidas de {len(y)} para {len(y_resampled)} registros sintéticos.")

    return df_resampled

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