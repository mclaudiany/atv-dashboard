import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Dashboard - KDD Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("LungCanC2024 - Análise Exploratória de Dados (EDA)")
st.markdown("---")

st.markdown("""
    Análise estatística das variáveis de **Idade** e **Gênero** para validação dos pesos preditivos da arquitetura em cascata (**PP1**).
""")

@st.cache_data
def carregar_dataframe_real():
    df = pd.read_csv('./data/train.csv')
    if df is not None:
        st.session_state['df'] = df
    return df

df_lung = carregar_dataframe_real()

def mostrar_graph_dist_idade_x_classe(df_filtrado):
    df_prop = df_filtrado.groupby(['cancer_subtype', 'patient_gender']).size().reset_index(name='Contagem')
    fig_bar = px.bar(
                df_prop, 
                x="cancer_subtype", 
                y="Contagem", 
                color="patient_gender",
                barmode="relative",    
                color_discrete_map={'Female': "#e25e63", 'Male': '#1f77b4'},
                labels={'patient_gender': 'Gênero'}
            )

    fig_bar.update_layout(
                xaxis_title="Subtipo de Câncer", 
                yaxis_title="Proporção Relativa (%)"
            )

    st.plotly_chart(fig_bar, width='stretch')

def montar_filtragem(df_lung):
    st.sidebar.header("Filtros do Dataset")
    filtro_genero = st.sidebar.multiselect(
        "Filtrar por Gênero do Paciente:",
        options=df_lung['patient_gender'].unique(),
        default=df_lung['patient_gender'].unique()
    )

    filtro_subtipo = st.sidebar.multiselect(
        "Filtrar por Subtipo Histológico:",
        options=df_lung['cancer_subtype'].unique(),
        default=df_lung['cancer_subtype'].unique()
    )
    
    df_filtrado = df_lung[
    (df_lung['patient_gender'].isin(filtro_genero)) & 
    (df_lung['cancer_subtype'].isin(filtro_subtipo))
    ]

    df_filtrado = df_lung[
        (df_lung['patient_gender'].isin(filtro_genero)) & 
        (df_lung['cancer_subtype'].isin(filtro_subtipo))
    ]
    
    df_filtrado['Diagnóstico'] = df_filtrado['cancer_presence'].map({0: 'Benigno', 1: 'Maligno'})

    st.header("Dashboard de Distribuíções e Correlações")

    total_pacientes = len(df_filtrado)
    mediana_idade = int(df_filtrado['patient_age'].median()) if total_pacientes > 0 else 0
    pct_malignos = (len(df_filtrado[df_filtrado['Diagnóstico'] == 'Maligno']) / total_pacientes * 100) if total_pacientes > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total de Pacientes Analisados", value=f"{total_pacientes}")
    col2.metric(label="Mediana de Idade", value=f"{mediana_idade} anos")
    col3.metric(label="Prevalência de Patologia Maligna", value=f"{pct_malignos:.1f}%")
    return df_filtrado,total_pacientes

def mostrar_graph_proporcao_genero_x_perfil(df_filtrado):
    fig_box = px.box(
                df_filtrado, 
                x="cancer_subtype", 
                y="patient_age", 
                color="Diagnóstico",
                color_discrete_map={'Benigno': "#3b6bca", 'Maligno': "#e34040"},
                points="outliers"
            )
    fig_box.update_layout(xaxis_title="", yaxis_title="Idade do Paciente (Anos)")
    st.plotly_chart(fig_box, width='stretch')

def mostrar_graph_matrix_corr(df_lung):
    features_radiomicas = ['nodule_size_mm', 'HU_mean', 'PET_SUVmax']
    features_clinicas = ['patient_age', 'patient_gender', 'smoking_history']
    biomarcadores = ['PD-L1_expression_level', 'tumor_mutational_burden']
    
    features_corr = (
        features_radiomicas + 
        features_clinicas + 
        biomarcadores
    )
    
    features_existentes = [
        col for col in features_corr 
        if col in df_filtrado.columns and pd.api.types.is_numeric_dtype(df_filtrado[col])
    ]
    
    if len(features_existentes) > 0:
        df_corr_real = df_filtrado[features_existentes].corr()

        fig_corr = px.imshow(
            df_corr_real,
            text_auto='.2f',                   
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Variáveis", y="Variáveis", color="Correlação")     
        )

        fig_corr.update_traces(
            xgap=2,                             
            ygap=2,                            
            textfont=dict(size=13),
        )
        
        fig_corr.update_layout(
            title={
                'text': "Matriz de Correlação das Features Principais",
                'y': 0.98,
                'x': 0.5,                       
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            height=650,                         
            margin=dict(l=150, r=50, t=60, b=150), 
            coloraxis_showscale=True,           
            xaxis=dict(
                tickangle=45,                   
                tickfont=dict(size=15)
            ),
            yaxis=dict(
                tickfont=dict(size=15)          
            )
        )
            
        st.plotly_chart(fig_corr, width='stretch')

if df_lung.empty:
    st.warning("Dataset não encontrado.")
else:
    df_filtrado, total_pacientes = montar_filtragem(df_lung)

    st.subheader("Cruzamentos Epidemiológicos")

    if total_pacientes > 0:
        graf1, graf2 = st.columns(2)

        with graf1:
            st.markdown("**Distribuição de Idade por Classe Histológica**")
            mostrar_graph_proporcao_genero_x_perfil(df_filtrado)

        with graf2:
            st.markdown("**Proporção de Gênero por Perfil Histológico**")
            mostrar_graph_dist_idade_x_classe(df_filtrado)
    
        mostrar_graph_matrix_corr(df_lung)
    
    
    else:
        st.error("Nenhum dado selecionado nos filtros da barra lateral.")
    
    
