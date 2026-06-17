import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Dashboard - Análise Exploratória de Dados (EDA)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Análise Exploratória de Dados (EDA)")

@st.cache_data
def carregar_dataframe(path):
    df = pd.read_csv(path)
    return df

if "df" not in st.session_state:
    st.session_state.df = None

df_treino = carregar_dataframe('./data/train.csv')
df_teste = carregar_dataframe('./data/test.csv')

if df_treino is not None:
    st.session_state.df = df_treino
  
tab_geral, tab_pp1, tab_pp2 = st.tabs([
    "1. Visão Geral e Balanceamento", 
    "2. Pergunta 1: Triagem Inicial", 
    "3. Pergunta 2: Subtipagem Tumoral"
])

def secao_introducao_visao_geral():
    st.markdown(f"""
    Ao integrar dados de diferentes naturezas podemos utilizar os dados para medicina preditiva. Esse pipeline tem como base os **três pilares** abaixo:
    * **Dados Clínicos:** `Idade`, `Gênero` e histórico do paciente.
    * **Radiômicos (Bioimagem):** `Tamanho`, `Densidade` e `Consumo Metabólico` do nódulo (estrutura física e atividade do tumor).
    * **Genômicos (Biomarcadores):** `Carga Mutacional` e `Expressão de Proteínas` (assinatura genética celular).
    O cruzamento destas variáveis permite que as Redes Neurais identifiquem padrões ocultos para a triagem inicial e a classificação dos subtipos tumorais.
    """)

    st.info(f"""
    **Objetivo da EDA:** Demonstrar visualmente a correlação e a desacoplamento destes três pilares antes do treino dos modelos de Inteligência Artificial.
    """)

def secao_volumetria(df_train, df_test):
    st.markdown("### 1.1. Volumetria do Dataset e Estrutura dos Dados")
    
    df = pd.concat([df_train, df_test], ignore_index=True).copy()
    
    total_linhas, total_colunas = df.shape
    
    metricas1, metricas2 = st.columns(2)
    with metricas1:
        st.metric(label="Total de Pacientes Registados (Linhas)", value=f"{total_linhas:,}")
    with metricas2:
        st.metric(label="Total de Características (Colunas/Features)", value=total_colunas)
        
    st.markdown("""
        A volumetria a seguir representa a base amostral completa do conjunto de dados antes da divisão em conjuntos de treino e teste. 
        Para garantir a integridade analítica e evitar o vazamento de dados (*data leakage*), as variáveis estão divididas entre 
        metadados clínicos, radiômicos e biomarcadores genômicos.
    """)
    
    with st.expander("Visualizar Características do Dataset"):
        info_colunas = pd.DataFrame({
            "Coluna / Feature": df.columns,
            "Tipo de Dado": [str(dtype) for dtype in df.dtypes],
            "Exemplo de Registro": [df[col].iloc[0] if len(df) > 0 else "N/A" for col in df.columns]
        })
        info_colunas["Exemplo de Registro"] = info_colunas["Exemplo de Registro"].astype(str)
        st.dataframe(info_colunas, hide_index=True, width='stretch')

def secao_qualidade_dados(df_train, df_test):
    st.markdown("### 1.2. Diagnóstico de Qualidade dos Dados")
    
    df = pd.concat([df_train, df_test], ignore_index=True).copy()
    dados_nulos = df.isnull().sum()
    total_nulos = dados_nulos.sum()
    df_nulos = pd.DataFrame({
        'Característica (Feature)': dados_nulos.index,
        'Valores Ausentes': dados_nulos.values
    })
    df_nulos = df_nulos[df_nulos['Valores Ausentes'] > 0] 
    
    total_duplicados = df.duplicated().sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Rastreamento de Dados Ausentes")
        if total_nulos == 0:
            st.success("**Nenhum valor ausente identificado!**")
        else:
            st.error(f"Detectados **{total_nulos}** valores nulos nas colunas.")
            st.dataframe(df_nulos.astype(str), hide_index=True, width='stretch')
            st.caption("O pipeline aplicará imputação (média/mediana) antes do treinamento.")

    with col2:
        st.markdown("#### Rastreamento de Registros Duplicados")
        if total_duplicados == 0:
            st.success("**Nenhum registro duplicado identificado!** Cada linha representa um perfil de paciente único e exclusivo.")
        else:
            st.warning(f"Foram detetadas **{total_duplicados}** linhas idênticas duplicadas na base de dados.")
            st.caption("É importante realziar a remoção desses registros redundantes para evitar overfitting.")
            
    st.markdown("#### Status do Pré-Processamento Inicial")
    
    if total_nulos == 0 and total_duplicados == 0:
        st.info("""
        **Confirmação de Integridade:** O diagnóstico de qualidade dos dados confirma que o **Pré-Processamento Inicial** foi executado com sucesso. 
        Os registros foram limpos e validados, garantindo que a **Padronização Estatística Z-score** utilize dados confiáveis, 
        integrando de forma equilibrada as variáveis.
        """)
    else:
        st.info("Será necessário realizar a limpeza de dados na Fase 2 (KDD) para aplicar as correções.")

def secao_balanceamento(df_train, df_test):
    st.markdown("### 1.3 Distribuição e Balanceamento das Variáveis Alvo")
    st.markdown("Antes de cruzar as variáveis, precisamos entender a proporção de classes que o modelo enfrentará.")
    
    df = pd.concat([df_train, df_test], ignore_index=True).copy()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. Target: Presença de câncer (`cancer_presence`)")
        
        contagem_presenca = df['cancer_presence'].value_counts().reset_index()
        contagem_presenca.columns = ['Classe', 'Nº de Pacientes']
        contagem_presenca['Classe'] = contagem_presenca['Classe'].map({0: '0 - Ausente (Saudável)', 1: '1 - Presente (Maligno)'})
        
        total_presenca = contagem_presenca['Nº de Pacientes'].sum()
        contagem_presenca['Proporção (%)'] = ((contagem_presenca['Nº de Pacientes'] / total_presenca) * 100).round(1)
        
        st.dataframe(contagem_presenca, hide_index=True, width='stretch')
        
        fig_presenca = px.bar(
            contagem_presenca, 
            x='Classe', 
            y='Nº de Pacientes',
            text='Proporção (%)',
            color='Classe',
            color_discrete_sequence=["#3b6bca",  "#e93f3f"]
        )
        fig_presenca.update_traces(texttemplate='%{text}%', textposition='inside')
        fig_presenca.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_presenca, width='stretch')

    with col2:
        st.markdown("#### 2. Target: Subtipos Tumorais (`cancer_subtype`)")
        
        df_doentes = df[df['cancer_presence'] == 1]
        if df_doentes.empty:
            st.info("Aguardando registros positivos para calcular os subtipos.")
        else:
            df_subtipos_real = df_doentes[df_doentes['cancer_subtype'] != 'No Cancer']
            contagem_subtipos = df_subtipos_real['cancer_subtype'].value_counts().reset_index()
            contagem_subtipos.columns = ['Subtipo', 'Nº de Pacientes']
            
            total_subtipos = contagem_subtipos['Nº de Pacientes'].sum()
            contagem_subtipos['Proporção (%)'] = ((contagem_subtipos['Nº de Pacientes'] / total_subtipos) * 100).round(1)
            
            st.dataframe(contagem_subtipos, hide_index=True, width='stretch')
            
            fig_subtipos = px.bar(
                contagem_subtipos, 
                y='Subtipo', 
                x='Nº de Pacientes',
                text='Proporção (%)',
                orientation='h',
                color='Subtipo',
                color_discrete_sequence=["#375ca7",  "#983838", "#0370BD",  "#e93f3f"]
            )
            fig_subtipos.update_traces(texttemplate='%{text}%', textposition='inside')
            fig_subtipos.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_subtipos, width='stretch')
   
def secao_justificativa_metodologica(df_train, df_test):
    st.markdown("### 1.4. Justificativa Metodológica")
    
    df = pd.concat([df_train, df_test], ignore_index=True).copy()
    
    st.markdown("### Equilíbrio na Triagem (Modelo 1)")
    total_pacientes = len(df)
    casos_malignos = int(df['cancer_presence'].sum())
    casos_saudaveis = total_pacientes - casos_malignos
    
    perc_maligno = (casos_malignos / total_pacientes) * 100
    perc_saudavel = (casos_saudaveis / total_pacientes) * 100
    
    st.markdown(f"""
    O **Modelo 1 (Triagem Inicial)** é baseado no total de dados do dataset com (**{total_pacientes:,} pacientes**) 
    com o objetivo de realizar uma classificação binária: determinar se o paciente possui ou não um tumor maligno.
    
    A análise estatística do dataset:
    * **Pacientes Saudáveis (Ausente):** {casos_saudaveis:,} registros ({perc_saudavel:.1f}%)
    * **Pacientes com câncer (Presente):** {casos_malignos:,} registros ({perc_maligno:.1f}%)
    """)
    if perc_saudavel >= 40 and perc_maligno >= 40:
        st.success(f""" A proporção exata de **{perc_saudavel:.0f}% vs. {perc_maligno:.0f}%** confirma que o dataset está **balanceado**. 
        **No Treinamento da Rede Neural**, a função de perda da MLP (*Binary Cross-Entropy*) dará a mesma importância para os erros de ambas as classes.
        """)
    else:
        st.warning(f"""A proporção exata de **{perc_saudavel:.0f}% vs. {perc_maligno:.0f}%** confirma que o dataset está **desbalanceado**. 
        Será necessário realizar ajuste com reamostragem, será aplicado a técnica de SMOTE (que cria dados sintéticos)**.""")
    
    
    st.markdown("### Filtro em Cascata e Balanceamento (Modelo 2)")
    df_doentes = df[(df['cancer_presence'] == 1) & (df['cancer_subtype'] != 'No Cancer')]
    
    total_doentes = len(df_doentes)
    contagem = df_doentes['cancer_subtype'].value_counts()
    
    classe_majoritaria = contagem.index[0]
    perc_majoritaria = (contagem.iloc[0] / total_doentes) * 100
    
    classe_minoritaria = contagem.index[-1]
    perc_minoritaria = (contagem.iloc[-1] / total_doentes) * 100
    
    st.markdown(f"""
    Na arquitetura em cascata do projeto, o **Modelo 2 (Multiclasse)** ignora os pacientes saudáveis e atua 
    unicamente sobre a população diagnosticada com câncer (**{total_doentes:,} pacientes**). 
    """)
    st.markdown(f"""
    Ao recalcular as proporções reais para esta etapa, o cenário de desbalanceamento torna-se evidente:
    * A classe maioritária (**{classe_majoritaria}**) domina o cenário com **{perc_majoritaria:.1f}%** dos casos.
    * A classe minoritária (**{classe_minoritaria}**) representa apenas **{perc_minoritaria:.1f}%** da amostra filtrada.""", unsafe_allow_html=True)
    
    df_barra = contagem.reset_index()
    df_barra.columns = ['Subtipo Tumoral', 'Nº de Pacientes']
    
    total_doentes = df_barra['Nº de Pacientes'].sum()
    df_barra['Porcentagem (%)'] = ((df_barra['Nº de Pacientes'] / total_doentes) * 100).round(1)
    
    fig_barra = px.bar(
        df_barra, 
        x='Subtipo Tumoral', 
        y='Nº de Pacientes',
        text=df_barra['Porcentagem (%)'].astype(str) + '%', 
        title="Distribuição Real dos Subtipos Histológicos (Pós-Filtro de Triagem)",
        color='Subtipo Tumoral', 
        color_discrete_sequence=["#375ca7",  "#983838", "#0370BD",  "#e93f3f"]
    )
    
    fig_barra.update_traces(
        textposition='outside', 
        cliponaxis=False        
    )
    
    fig_barra.update_layout(
        height=380, 
        showlegend=False,      
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Nº de Pacientes",
        xaxis_title="Subtipo Tumoral"
    )
    
    st.plotly_chart(fig_barra, width='stretch')
    
    st.markdown(f"""
    **Reamostragem Estatística no Modelo 2:**
    * O **Modelo 1** (está balanceado em {perc_saudavel:.1f}/{perc_maligno:.1f}). 
    * O **Modelo 2** enfrenta um desbalanceamento multiclasse: ({perc_majoritaria:.1f}% vs. {perc_minoritaria:.1f}%). 
    Se Rede Neura MLP Multiclasse for treinada com esses dados, ela sofrerá de **sub-aprendizagem (underfitting)**.
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    **Ações de Correção:**
    * Para mitigar esse problema, o conjunto de treino do Modelo 2 passará por um processo de **SMOTE Multiclasse (Synthetic Minority Over-sampling Technique)**. 
    Onde o algoritmo criará perfis sintéticos baseados em vizinhos próximos ($k$-NN) para as classes minoritárias até que todos os 4 subtipos histológicos alcancem o mesmo peso numérico antes do ajuste dos pesos da MLP.
    """, unsafe_allow_html=True)

def secao_analise_densidade_idade_pp1(df_train, df_test):
    st.markdown("### 2.1. Análise de Densidade por Idade")
  
    df = pd.concat([df_train, df_test], ignore_index=True)
    df_idade = df.copy()
    df_idade['Diagnóstico'] = df_idade['cancer_presence'].map({
        0: 'Ausente (Saudável)', 
        1: 'Presente (Maligno)'
    })
    
    fig_idade = px.histogram(
        df_idade,
        x="patient_age",
        color="Diagnóstico",         
        nbins=30,                     
        histnorm="probability density", 
        color_discrete_sequence=["#2655b4",  "#e82f2f"],
        labels={"patient_age": "Idade do Paciente (Anos)", "probability density": "Densidade"},
        title="Distribuição da Idade por Status do Diagnóstico"
    )
    
    fig_idade.update_traces(opacity=0.6) 
    fig_idade.update_layout(
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig_idade, width='stretch')
    
    media_saudavel = df[df['cancer_presence'] == 0]['patient_age'].mean()
    media_maligno = df[df['cancer_presence'] == 1]['patient_age'].mean()
    
    st.info(f"""
    **Insight:**
    * **Idade média (Saudáveis):** {media_saudavel:.1f} anos.
    * **Idade média (Com Câncer):** {media_maligno:.1f} anos.
    """)
    
    if media_maligno > media_saudavel:
        st.info("**Conclusão:** As barras azuis (Maligno) e vermelhas (Saudável) estão praticamente empilhadas de forma reta e uniforme ao longo de todo o eixo X (dos 30 aos 90 anos). Não há \"pico\" em nenhuma idade específica para nenhum dos dois grupos. Significa que nesse dataset, a idade isolada possui baixo poder discriminatório para a triagem inicial.")
    else:
        st.info("**Conclusão:** As curvas apresentam forte sobreposição, indicando que a idade isolada possui baixo poder de diferenciação na triagem inicial desta amostra.")

def secao_analise_distibuicao_genero_pp1(df_train, df_test):
    st.markdown("### 2.2. Proporção do Diagnóstico entre Gêneros")
    
    df_genero = pd.concat([df_train, df_test], ignore_index=True).copy()
    df_genero['Diagnóstico'] = df_genero['cancer_presence'].map({0: 'Saudável', 1: 'Maligno'})
    
    df_contagem = df_genero.groupby(['patient_gender', 'Diagnóstico']).size().reset_index(name='Pacientes')
    
    if not df_contagem.empty:
        fig_genero = px.bar(
            df_contagem,
            x="patient_gender",
            y="Pacientes",
            color="Diagnóstico",
            barmode="stack",
            text="Pacientes",
            color_discrete_sequence=["#3b6bca",  "#e93f3f"],
            labels={"patient_gender": "Gênero do Paciente", "Pacientes": "Nº Absoluto de Pacientes"},
            title="Relação de Diagnósticos por Gênero"
        )
        
        fig_genero.update_traces(textposition='inside')
        fig_genero.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_genero, width='stretch')
        
        st.info("""
        **Insight:**
        * Se a divisão estiver próxima de 50/50 em ambos os Gêneros, significa que a prevalência do câncer está uniformemente distribuída entre homens e mulheres na base dados.
        """)

def secao_analise_atributos_nodulo_pp1(df_train, df_test):
    st.markdown("### 2.3. Análise dos Atributos do Nódulo")
    st.markdown("""
    Os **atributos radiômicos** extraídos dos exames de imagem carregam a assinatura física do tumor.
    """)

    df_presenca = pd.concat([df_treino, df_teste], axis=0, ignore_index=True).copy()
    df_presenca['Diagnóstico'] = df_presenca['cancer_presence'].map({0: 'Saudável', 1: 'Maligno'})
    
    col_box1, col_box2 = st.columns(2)
    with col_box1:
        st.markdown("#### Dimensão Estrutural")
        fig_box1 = px.box(
            df_presenca, 
            x="Diagnóstico",
            y="nodule_size_mm",  
            color="Diagnóstico",
            color_discrete_sequence=["#3b6bca",  "#e93f3f"],
            labels={"nodule_size_mm": "Tamanho do Nódulo (mm)"},
            title="Tamanho do Nódulo vs. Diagnóstico"
        )
        fig_box1.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_box1, width='stretch')
        
    with col_box2:
        st.markdown("#### Atividade Metabólica")
        fig_box2 = px.box(
            df_presenca, 
            x="Diagnóstico",
            y="PET_SUVmax",  
            color="Diagnóstico",
            color_discrete_sequence=["#3b6bca",  "#e93f3f"],
            labels={"PET_SUVmax": "Captação Metabólica (PET SUVmax)"},
            title="Consumo de Glicose (SUVmax) vs. Diagnóstico"
        )
        fig_box2.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_box2, width='stretch')
        
    st.success("""
    **Evidência Visual de Separação:**
    
    O objetivo desta análise é **provar visualmente que nódulos malignos tendem a ser maiores e a consumir mais glicose**. 
    
    * **Nódulos Malignos (Vermelho):** As caixas deslocadas para o topo do eixo Y, indicando diâmetros superiores e altos índices de ${SUV_max}$ 
    (devido ao efeito Warburg de hipermetabolismo celular).
    * **Nódulos Benignos/Saudáveis (Azul):** Concentram-se na base inferior do gráfico.
    
    **Conclusão:** Para o modelo 1, a distância entre as caixas confirma que as features radiômicas possuem **alto poder discriminatório**, 
    compensando a neutralidade das variáveis demográficas e servindo como a principal fundação para os neurônios da MLP realizarem a 
    triagem binária.
    """)
 
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
        
        st.plotly_chart(fig_9, use_container_width=True)
        
        n_outliers = (df_dbscan['Cluster_DBSCAN'] == -1).sum()
        pct_outliers = (n_outliers / len(df_dbscan)) * 100
        
        st.caption(f"**Resultado do Algoritmo:** Foram identificados **{n_outliers}** pacientes como outliers ({pct_outliers:.1f}% da amostra analisada).")
        
        st.info("""
        **Aplicações Práticas & Camada de Proteção:** 
        * Os pontos mapeados em vermelho representam pacientes com características clínicas ou radiómicas extremamente raras (ex: nódulos gigantes com textura incomum). 
        * Na arquitetura do projeto, o DBSCAN atua como um **filtro de segurança**: isolando eses casos atípicos, evitando que as redes neurais (Modelos 1 e 2) induzam ao erro. 
        """)
    
with tab_geral:   
    secao_introducao_visao_geral()
    
    secao_volumetria(df_treino, df_teste)  
    
    secao_qualidade_dados(df_treino, df_teste)
    
    secao_balanceamento(df_treino, df_teste)
    
    secao_justificativa_metodologica(df_treino, df_teste)

with tab_pp1:
    st.markdown("## Pergunta 1: Como identificar na triagem inicial a existência de câncer?")
    
    secao_analise_densidade_idade_pp1(df_treino, df_teste)
    
    secao_analise_distibuicao_genero_pp1(df_treino, df_teste)
    
    secao_analise_atributos_nodulo_pp1(df_treino, df_teste)

with tab_pp2:
    st.markdown("## Pergunta 2: Uma vez identificada a patologia, como mapear o seu subtipo?")
    
    secao_analise_distribuicao_idade_subtipo_pp2(df_treino, df_teste)
    
    secao_analise_distribuicao_genero_subtipo_pp2(df_treino, df_teste)
    
    secao_analise_correlacao_atributos_quantitativos_pp2(df_treino, df_teste)
    
    secao_identificacao_outliers_dbscan_pp2(df_treino, df_teste)
        
        
