import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

@st.cache_data
def caregar_csv_para_dataframe(data_path):
    df = pd.read_csv(data_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    df.attrs["dataset_name"] = Path(data_path).name.upper()
    
    return df

def carregar_features_selecionadas(df):
    features_radiomicas = ['nodule_size_mm', 'nodule_texture', 'HU_mean', 'HU_std', 
        'GLCM_contrast', 'GLCM_correlation', 'PET_SUVmax', 'PET_SUVmean']
    features_clinicas = ['patient_age', 'patient_gender', 'smoking_history', 'family_history']
    features_tumor_tratamento = [
        'tumor_location', 'tumor_stage', 'radiation_therapy', 
        'chemotherapy_received', 'immunotherapy_received', 'targeted_therapy_received'
    ]
    biomarcadores = [
        'EGFR_mutation_status', 'KRAS_mutation_status', 'ALK_fusion_status', 
        'PD-L1_expression_level', 'tumor_mutational_burden'
    ]
    target = ['cancer_presence', 'cancer_subtype']
    
    colunas_selecionadas = (
        features_radiomicas + 
        features_clinicas + 
        features_tumor_tratamento + 
        biomarcadores + target
    )
    
    df_selecionado = df[colunas_selecionadas]
        
    return df_selecionado.copy()

def etapa_selecao_mapeamento():
    st.header("Seleção e Mapeamento de Features")
    st.markdown("Foi realizada a triagem de atributos disponíveis no dataset.")
    
    st.markdown("""
        **Modelo 1 (Presença de câncer):**
        Serão utilizados dados de exames iniciais (Idade, Gênero, Histórico de Fumo) e características físicas do nódulo (Radiômica e PET Scan).
    """)
    st.markdown("""
        **Modelo 2 (Subtipagem de câncer)**
        Uma vez confirmado que o câncer existe, o escopo de dados é expandido. Serão adicionados os Biomarcadores (*EGFR, KRAS, ALK, TMB, PD-L1*). 
    """)

    st.subheader("Mapeamento Visual de Atributos por Modelo")
    mapear_features()

    st.subheader("Definição das Variáveis Targets")
    definir_targets()

    df_test = caregar_csv_para_dataframe('./data/test.csv')
    if df_test is not None:
        X_test = carregar_features_selecionadas(df_test)
    
    df_train = caregar_csv_para_dataframe('./data/train.csv')
    if df_train is not None:
        X_train = carregar_features_selecionadas(df_train)
        
    return X_test, X_train

def mapear_features():
    dados_mapeamento = {
        "Feature": [
            "patient_age", "patient_gender", "smoking_history", "family_history", "tumor_location",
            "nodule_size_mm", "nodule_texture", "HU_mean", "HU_std", "GLCM_contrast", "GLCM_correlation",
            "PET_SUVmax", "PET_SUVmean",
            "EGFR_mutation_status", "KRAS_mutation_status", "ALK_fusion_status", "PD-L1_expression_level", "tumor_mutational_burden",
            "tumor_stage", "radiation_therapy", "chemotherapy_received", "immunotherapy_received", "targeted_therapy_received", "survival_time_months"
        ],
        "Descrição": [
            "Idade do paciente (30 a 90 anos).", "Gênero biológico (M/F).", "Histórico de tabagismo (Nunca, Ex ou Atual).", "Histórico familiar de câncer pulmonar.", "Localização do tumor (Esquerdo/Direito).",
            "Tamanho do nódulo detectado (em mm).", "Textura radiológica derivada da tomografia.", "Média das Unidades Hounsfield (Densidade).", "Desvio padrão das Unidades Hounsfield.", "Contraste GLCM (Heterogeneidade tumoral).", "Correlação GLCM (Consistência do padrão).",
            "Valor máximo de captação padronizada (Atividade metabólica).", "Média de captação de glicose na região tumoral.",
            "Status mutacional do gene EGFR.", "Status mutacional do gene KRAS.", "Presença de fusão do gene ALK.", "Nível de expressão do biomarcador PD-L1 (0-100%).", "Carga Mutacional Tumoral (Instabilidade do DNA).",
            "Estágio do tumor (I a IV).", "Se o paciente realizou radioterapia.", "Se o paciente recebeu quimioterapia.", "Se o paciente recebeu imunoterapia.", "Se o paciente recebeu terapia-alvo.", "Tempo estimado de sobrevida em meses."
        ],
        "Modelo 1": [
            "Utilizada", "Utilizada", "Utilizada", "Utilizada", "Utilizada",
            "Utilizada", "Utilizada", "Utilizada", "Utilizada", "Utilizada", "Utilizada",
            "Utilizada", "Utilizada",
            "Não Utilizada", "Não Utilizada", "Não Utilizada", "Não Utilizada", "Não Utilizada",
            "Não Utilizada", "Não Utilizada", "Não Utilizada", "Não Utilizada", "Não Utilizada", "Não Utilizada"
        ],
        "Modelo 2": [
            "Utilizada", "Utilizada", "Utilizada", "Utilizada", "Utilizada",
            "Utilizada", "Utilizada", "Utilizada", "Utilizada", "Utilizada", "Utilizada",
            "Utilizada", "Utilizada",
            "Utilizada", "Utilizada", "Utilizada", "Utilizada", "Utilizada",
            "Não Utilizada", "Não Utilizada", "Não Utilizada", "Não Utilizada", "Não Utilizada", "Não Utilizada"
        ]
    }

    df_mapeamento = pd.DataFrame(dados_mapeamento)

    st.dataframe(
        df_mapeamento,
        width='stretch',
        hide_index=True,
        column_config={
            "Feature": st.column_config.TextColumn("Feature"),
            "Descrição": st.column_config.TextColumn("Descrição Clínica"),
            "Modelo 1": st.column_config.TextColumn("Modelo 1 (Presença de câncer)"),
            "Modelo 2": st.column_config.TextColumn("Modelo 2 (Subtipagem de câncer)")
        }
    )

def definir_targets():
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric(
            label="Target - Modelo 1", 
            value="cancer_presence", 
            delta_color="normal"
        )
    with col_t2:
        st.metric(
            label="Target - Modelo 2", 
            value="cancer_subtype", 
            delta_color="normal"
        )