import streamlit as st
import plotly.express as px
import pandas as pd


def secao_analise_densidade_idade_pp1(df_train, df_test):
    st.markdown("### 2.1. Distribuição de Densidade por Faixa Etária")
  
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
        title="Análise de Frequência Relativa da Idade vs. Status do Tumor"
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
            title="Relação de Diagnósticos por Gênero Biológico"
        )
        
        fig_genero.update_traces(textposition='inside')
        fig_genero.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_genero, width='stretch')
        
        st.info("""
        **Insight:**
        * Se a divisão estiver próxima de 50/50 em ambos os Gêneros, significa que a prevalência do câncer está uniformemente distribuída entre homens e mulheres na base dados.
        """)

def secao_analise_atributos_nodulo_pp1(df_train, df_test):
    st.markdown("### 2.3. Assinatura Radiômica: Capacidade Discriminatória de Bioimagem")
    st.markdown("""
    Os **atributos radiômicos** extraídos dos exames de imagem carregam a assinatura física do tumor.
    """)

    df_presenca = pd.concat([df_train, df_test], axis=0, ignore_index=True).copy()
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
        
 
