import streamlit as st

st.set_page_config(
    page_title="Dashboard - KDD Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Detecção e Subtipagem de Câncer de Pulmão")
st.markdown("---")

st.markdown("""
    <div style="background-color:#F0F4F8; padding:20px; border-radius:10px; border-left: 6px solid #1A365D; margin-bottom: 25px;">
        <p style="color:#2D3748; font-size:16px; line-height:1.6;">
            Este projeto é um pipeline baseado na metodologia <b>KDD (Knowledge Discovery in Databases)</b> 
            para extrair conhecimento a partir de um dataset oncológico. O objetivo central é o desenvolvimento de 
            dois modelos preditivos supervisionados e uma análise de agrupamento não supervisionado (Clusterização).
        </p>
    </div>
""", unsafe_allow_html=True)

st.header("Governança de Dados")
st.markdown("As variáveis do conjunto de dados foram mapeadas para serem utilizadas nos modelos preditivos:")

with st.expander("Modelo 1: Detecção de Presença de Câncer", expanded=True):

    st.metric(label="Variável Alvo (Target)", value="cancer_presence", delta="0 = Ausente | 1 = Maligno")
    
    st.subheader("Features Utilizáveis (Modelo 1)")
    st.markdown("""
        * `nodule_size_mm`, `nodule_texture`, `HU_mean`, `HU_std`, `GLCM_contrast`, `GLCM_correlation`, `PET_SUVmax`, `PET_SUVmean`.
        * `patient_age`, `patient_gender`, `smoking_history`, `family_history`, `tumor_location`.
    """)
    
with st.expander("Modelo 2: Identificação do Subtipo de Câncer", expanded=True):
    
    st.metric(label="Variável Alvo (Target)", value="cancer_subtype", delta="Adenocarcinoma, Squamous, SCLC, Other")
    
    st.subheader("Features Utilizáveis (Modelo 2)")
    st.markdown("""
        * Todas as features utilizadas no Modelo 1 + Variáveis de alta complexidade molecular:
        * `EGFR_mutation_status`, `KRAS_mutation_status`, `ALK_fusion_status`, `PD-L1_expression_level`, `tumor_mutational_burden`.
    """)
    st.warning("Pacientes com valor \"No Cancer\" são automaticamente removidos desta modelagem.")

