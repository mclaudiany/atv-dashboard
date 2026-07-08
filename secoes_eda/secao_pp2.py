import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def secao_analise_distribuicao_idade_subtipo_pp2(df_treino, df_teste):
    st.markdown("### 3.1. Distribuição etária dos Paciente por Subtipo")
    st.markdown("""
    * Mapear se determinados subtipos histológicos atacam populações mais jovens ou mais idosas.
    """)

    df = pd.concat([df_treino, df_teste], axis=0).copy()
    df_doentes = df[df['cancer_presence'] == 1]
    
    if df_doentes.empty:
        st.info("Aguardando registros positivos para calcular os subtipos.")
    else:
        df_subtipos_real = df_doentes[df_doentes['cancer_subtype'] != 'No Cancer']
    
        colunas_necessarias = ['cancer_subtype', 'patient_age']
        if all(col in df_subtipos_real.columns for col in colunas_necessarias):
            fig_paciente = px.box(
                df_subtipos_real,
                x='cancer_subtype',
                y='patient_age',
                color='cancer_subtype',
                title='Distribuição de Idade por Subtipo Tumoral (Pacientes com Câncer)',
                labels={
                    'cancer_subtype': 'Subtipo Tumoral',
                    'patient_age': 'Idade do Paciente (Anos)'
                },
                points="all", 
                color_discrete_sequence=["#375ca7",  "#983838", "#0370BD",  "#e93f3f",]
            )
            fig_paciente.update_layout(
                showlegend=False,
                xaxis_title="Subtipo Tumoral",
                yaxis_title="Idade (Anos)",
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_paciente, width='stretch')

            st.info("""
            **Insight:**
                
            * O gráfico permite identificar a mediana de idade e os quartis de cada subtipo tumoral presente no dataset *lungcancer2024*. 
            * Subtipos como *Pequenas Células* frequentemente apresentam comportamento mais agressivo.
            """)

def secao_analise_distribuicao_genero_subtipo_pp2(df_treino, df_teste): 
    st.markdown("### 3.2. Distribuição dos Subtipos por Gênero do Paciente")
    st.markdown("""
    * Mapear se existe maior prevalência de determinados subtipos em homens ou mulheres.
    """)

    df = pd.concat([df_treino, df_teste], axis=0).copy()
    if 'cancer_subtype' in df.columns:
        df = df[df['cancer_subtype'] != 'No Cancer']
        
    df_gender_subtype = (df.groupby(['cancer_subtype', 'patient_gender']).size().reset_index(name='patient_count'))
    
    if all(col in df.columns for col in ['cancer_subtype', 'patient_gender']):
        fig_subtipo = px.bar(
            df_gender_subtype,
            x='cancer_subtype',
            y='patient_count',
            color='patient_gender',
            barmode='group',
            title='Proporção de Subtipo Histológico por Gênero',
            labels={
                'cancer_subtype': 'Subtipo Tumoral',
                'patient_count': 'Número de Pacientes',
                'patient_gender': 'Gênero'
            },
            color_discrete_sequence=["#3b6bca",  "#e93f3f"]
        )
        
        fig_subtipo.update_layout(
            xaxis_title="Subtipo Tumoral",
            yaxis_title="Quantidade de Pacientes",
            legend_title="Gênero",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig_subtipo, width='stretch')

def secao_analise_correlacao_atributos_quantitativos_pp2(df_treino, df_teste):
    st.markdown("### 3.3. Matriz de Correlação das Principais Features")

    df = pd.concat([df_treino, df_teste], axis=0).copy()
    df_diagnostico = df[df['cancer_presence'] == 1].copy()
    
    features_radiomicas = ['nodule_size_mm', 'HU_mean', 'PET_SUVmax']
    features_clinicas = ['patient_age']
    biomarcadores = ['PD-L1_expression_level', 'tumor_mutational_burden']
    
    features_disponiveis = (
        features_radiomicas + 
        features_clinicas + 
        biomarcadores
    )

    features_presentes = [
        col for col in features_disponiveis 
        if col in df_diagnostico.columns and pd.api.types.is_numeric_dtype(df_diagnostico[col])
    ]

    if len(features_presentes) >= 2:
        corr_matrix = df_diagnostico[features_presentes].corr()
        
        fig_correlacao = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}", 
            hoverinfo="z"
        ))
        
        fig_correlacao.update_layout(
            title='Matriz de Correlação das Variáveis Principais)',
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_correlacao, width='stretch')
        
        st.info("""
            **Insight:** 
            * A análise estatística revelou que o consumo metabólico (`PET_SUVmax`) e a carga mutacional (`tumor_mutational_burden`) 
            apresentam **correlação linear nula (0.00)**, basicamente são variáveis totalmente independentes.
            * O Modelo 2 precisa de ambas as fontes para fazer a triagem dos subtipos com precisão, já que uma variável não prevê a outra.
        """)

def secao_identificacao_outliers_dbscan_pp2(df_treino, df_teste):
    st.markdown("### Detecção de Outliers com DBSCAN")
    st.markdown("#### 4.1. Mapeamento de Densidade Oncológica")
    st.markdown("""
    * Mapear o comportamento conjunto das duas features de bioimagem mais fortes e isolar na distribuição dos pacientes.
    """)

    df = pd.concat([df_treino, df_teste], axis=0, ignore_index=True).copy()
    
    var_x = 'nodule_size_mm'
    var_y = 'nodule_texture' 

    if var_x in df.columns and var_y in df.columns:
        df_dbscan = df.dropna(subset=[var_x, var_y]).copy()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_dbscan[[var_x, var_y]])
        
        st.sidebar.markdown("---")
        st.sidebar.header("Parâmetros do DBSCAN")
        
        eps_val = st.sidebar.slider("Raio de Vizinhança (Eps):", min_value=0.1, max_value=2.0, value=0.5, step=0.05)
        min_samples_val = st.sidebar.slider("Mínimo de Amostras (MinPts):", min_value=2, max_value=15, value=5, step=1)
        
        dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
        df_dbscan['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)
        
        df_dbscan['Classificação Espacial'] = df_dbscan['Cluster_DBSCAN'].apply(
            lambda x: 'Outlier (Ruído Raro)' if x == -1 else f'Região de Densidade'
        )
        
        fig_9 = px.scatter(
            df_dbscan,
            x=var_x,
            y=var_y,
            color='Classificação Espacial',
            symbol='Classificação Espacial',
            color_discrete_map={
                'Outlier (Ruído Raro)': '#e93f3f',   
                'Região de Densidade': "#1226bd"     
            },
            title="Dispersão Espacial e Identificação de Anomalias via DBSCAN",
            labels={
                var_x: "Tamanho do Nódulo (mm)",
                var_y: "Textura do Nódulo (Padrão de Nuance)"
            },
            hover_data=['patient_age', 'cancer_presence']
        )
        
        fig_9.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
        fig_9.update_layout(
            legend_title="Auditoria Espacial",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig_9, width='stretch')
        
        n_outliers = (df_dbscan['Cluster_DBSCAN'] == -1).sum()
        pct_outliers = (n_outliers / len(df_dbscan)) * 100
        
        st.caption(f"**Resultado do Algoritmo:** Foram identificados **{n_outliers}** pacientes como outliers ({pct_outliers:.1f}% da amostra analisada).")
        
        st.info("""
        **Aplicações Práticas & Camada de Proteção:** 
        * Os pontos mapeados em vermelho representam pacientes com características clínicas ou radiómicas extremamente raras (ex: nódulos gigantes com textura incomum). 
        * Na arquitetura do projeto, o DBSCAN atua como um **filtro de segurança**: isolando eses casos atípicos, evitando que as redes neurais (Modelos 1 e 2) induzam ao erro. 
        """)
