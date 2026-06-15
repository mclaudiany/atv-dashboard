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
    col3.metric(label="Prevalência de Matologia Maligna", value=f"{pct_malignos:.1f}%")
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
    
    


    # st.subheader("DASHBOARD DE DISTRIBUIÇÕES E CORRELAÇÕES")

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.markdown("**Distribuição: Presença de Câncer**")
    #     fig_presenca = px.bar(
    #         df_presenca_real,
    #         x='Status',
    #         y='Nº de Pacientes',
    #         color_discrete_sequence=['#1f77b4'], 
    #         height=350
    #     )
    #     fig_presenca.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    #     st.plotly_chart(fig_presenca, width='stretch')

    # with col2:
    #     st.markdown("**Distribuição: Subtipos de Câncer**")
    #     fig_subtipos = px.bar(
    #         df_subtipos_real,
    #         x='Nº de Pacientes',
    #         y='Subtipo',
    #         orientation='h',
    #         color='Subtipo',
    #         color_discrete_sequence=px.colors.qualitative.Safe,
    #         height=350
    #     )
    #     fig_subtipos.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    #     st.plotly_chart(fig_subtipos, width='stretch')

    # st.markdown("<div style='text-align: center; font-weight: bold;'>Matriz de Correlação das Features Principais</div>", unsafe_allow_html=True)

    # fig_corr = px.imshow(
    #     df_corr_real,
    #     text_auto='.2f', 
    #     aspect="auto",
    #     color_continuous_scale="Blues", 
    #     range_color=[-1.0, 1.0] 
    # )
    # fig_corr.update_layout(
    #     height=500,
    #     margin=dict(l=50, r=50, t=20, b=50),
    #     coloraxis_showscale=True
    # )
    # st.plotly_chart(fig_corr, width='stretch')

    # st.caption(
    #     "**Figura 1:** Painel do Dashboard demonstrando o desbalanceamento das variáveis meta "
    #     "e a correlação linear entre features de bio-imagem e genômica[cite: 91]."
    # )

    # st.markdown("### Interpretação e Achados Críticos da EDA")

    # st.info("""
    # * **Poder Discriminatório das Imagens:** O parâmetro `PET_SUVmax` exibe uma forte correlação positiva 
    #  com a carga mutacional do tumor (`tumor_mutational_burden`), sugerindo diretamente que tumores 
    #  metabolicamente ativos são mais instáveis genomicamente[cite: 29].
    # * **Correlação Estrutural Radiômica:** Nota-se uma correlação linear moderada entre o tamanho do nódulo 
    #  (`nodule_size_mm`) e a atenuação média em Unidades Hounsfield (`HU_mean`).
    # * **Alerta de Desbalanceamento Crítico:** Como observado de forma acentuada na distribuição das variáveis alvo, 
    #  há um desbalanço populacional severo tanto na detecção de presença quanto nos subtipos[cite: 92]. 
    #  Isso exige, obrigatoriamente, tratamentos de amostragem específicos (como a técnica combinada *SMOTE-Tomek*) nas próximas fases do pipeline[cite: 92, 105].
    # """)