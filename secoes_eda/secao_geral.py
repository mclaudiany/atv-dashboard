import streamlit as st
import plotly.express as px
import pandas as pd

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
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    **Ações de Correção:**
    * Para mitigar esse problema, o conjunto de treino do Modelo 2 passará por um processo de **SMOTE Multiclasse (Synthetic Minority Over-sampling Technique)**. 
    Onde o algoritmo criará perfis sintéticos baseados em vizinhos próximos ($k$-NN) para as classes minoritárias até que todos os 4 subtipos histológicos alcancem o mesmo peso numérico antes do ajuste dos pesos da MLP.
    """, unsafe_allow_html=True)
