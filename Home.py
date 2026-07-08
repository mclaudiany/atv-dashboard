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


with st.expander("Atributos por Modelo", expanded=True):
    col_md1, col_md2, col_md3 = st.columns(3)
    with col_md1:
        st.markdown("### Redes Neurais (MLP)")
        st.caption("Aceita dados numéricos e categóricos (tratados).")
        st.markdown("""
        **Atributos de Entrada (Predritores):**
        *   `patient_gender` *(Mapeado para 0/1)*
        *   Filtros de Mutação (`*mutation*`)
        *   Densidade Tomográfica (`*HU*`)
        *   Captação PET-Scan (`*SUV*`)
        *   Histórico de Tabagismo (`*smoke*`)
        *   Localização Anatômica (`*location*`)
        """)
        
    with col_md2:
        st.markdown("### Clusterização (K-Means)")
        st.caption("Filtro estrito para dados puramente numéricos.")
        st.markdown("""
        **Atributos Utilizados:**
        *   Métricas de Densidade (`*HU*`)
        *   Métricas de Captação (`*SUV*`)
        *   Variáveis numéricas de Tabagismo (`*smoke*`)
        """)
        
    with col_md3:
        st.markdown("### Clusterização (DBSCAN)")
        st.caption("Dados numéricos com ajuste de escala espacial.")
        st.markdown("""
        **Atributos Utilizados (Padronizados):**
        *   `StandardScaler( *HU* )`
        *   `StandardScaler( *SUV* )`
        *   `StandardScaler( *smoke* )`
        """)