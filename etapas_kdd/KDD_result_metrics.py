import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier


def etapa_resultado_metricas():
    st.header("Avaliação e Interpretação de Métricas")
    st.subheader("Validação Estatística das Redes Neurais")

    st.markdown("Esta etapa garante que os modelos possuem capacidade de generalização através do cruzamento de dados com a base de teste.")

    epocas = st.session_state.get("epocas_executadas", 100)
    loss_name_mlp = st.session_state.get("loss_function_mlp", None)
    loss_mlp_limpo = loss_name_mlp.split(' (')[0] if loss_name_mlp else "N/A"
    loss_name_kmeans = st.session_state.get("loss_function_kmeans", "N/A")
    
    st.info(
        f"**Auditoria do Pipeline Ativo:** Métricas consolidadas após otimização profunda executada por **{epocas} épocas**. "
        f"Critérios de custo utilizados: MLP - {loss_mlp_limpo} e KMeans - {loss_name_kmeans}."
    )
    
    tab_atributos, tab_desempenho, tab_confusao, tab_densidade = st.tabs([
        "Importância de Atributos", 
        "Matriz de Desempenho dos Modelos", 
        "Matriz de Confusão",
        "Validação de Densidade"
    ])
    
    with tab_atributos:
        st.title("Análise de Importância de Atributos (Feature Importance)")
        df_filtrado = gerar_filtros()
        mostrar_feature_importance(df_filtrado)
                
    with tab_desempenho:
        st.title("Tabela comparativa estruturada com as métricas de validação.")
        gerar_tabela_metricas_detalhada()
    
    with tab_confusao:
        st.title("Análise de Erros por Matriz de Confusão")
        gerar_matriz_confusao()

    with tab_densidade:
        st.title("Índices de Validação de Densidade")
        gerar_indice_densidade()


def gerar_filtros():
    df_completo = pd.concat([st.session_state.df_train, st.session_state.df_test], ignore_index=True)

    opcoes_genero = [g for g in df_completo['patient_gender'].unique() if g in ['Male', 'Female']]
    opcoes_subtipo = [s for s in df_completo['cancer_subtype'].unique() if pd.notna(s) and s != 'No Cancer']

    generos_selecionados = st.multiselect("Filtrar por Gênero:", options=opcoes_genero)
    subtipos_selecionados = st.multiselect("Filtrar por Subtipo:", options=opcoes_subtipo)

    generos_filtro = generos_selecionados if generos_selecionados else opcoes_genero
    subtipos_filtro = subtipos_selecionados if subtipos_selecionados else opcoes_subtipo

    df_filtrado = df_completo[
            (df_completo['patient_gender'].isin(generos_filtro)) & 
            (df_completo['cancer_subtype'].isin(subtipos_filtro))
        ]
    
    return df_filtrado
    
def gerar_indice_densidade():
    score_silhueta,score_db, n_ruido, p_ruido = calcular_indices_densidade()

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        if score_silhueta is not None:
            st.metric(
                label="Coeficiente de Silhueta (Coesão)", 
                value=f"{score_silhueta:.3f}",
                help="Varia de -1 a +1. Valores próximos a 1 indicam que os pacientes estão no cluster correto e distantes de outros grupos."
            )
        else:
            st.metric("Coeficiente de Silhueta", "N/A", help="Indisponível: Altere os sliders para encontrar mais clusters.")

    with col_m2:
        if score_db is not None:
            st.metric(
                label="Índice Davies-Bouldin", 
                value=f"{score_db:.3f}",
                help="Mede a similaridade entre os clusters. Quanto menor o valor (próximo de 0), melhor e mais limpa é a separação física dos grupos."
            )
        else:
            st.metric("Índice Davies-Bouldin", "N/A", help="Indisponível: Altere os sliders para encontrar mais clusters.")

    with col_m3:
        st.metric(
            label="Estabilidade do Ruído", 
            value=f"{n_ruido} Pacientes",
            delta=f"{p_ruido:.1f}% da Base",
            delta_color="inverse",
            help="Volume de pacientes isolados de forma não supervisionada como perfis oncológicos atípicos."
        )

def calcular_indices_densidade():
    if "dbscan_labels" not in st.session_state or st.session_state.dbscan_labels is None:
       st.error("**Resultados de Clusterização não encontrados!**")
       st.warning("Por favor, acesse a etapa anterior (Mineração de Dados), execute o algoritmo DBSCAN ou K-Means para gerar os grupos e depois retorne a esta aba.")
       st.stop()
       
    labels_clusters = st.session_state.dbscan_labels
    X_scaled = st.session_state.dbscan_X_scaled
    
    clusters_validos = [label for label in labels_clusters if label != -1]
    n_clusters = len(set(clusters_validos))
    n_ruido = list(labels_clusters).count(-1)
    p_ruido = (n_ruido / len(labels_clusters)) * 100 if len(labels_clusters) > 0 else 0
    
    if n_clusters >= 1:
        mascara_sem_ruido = (labels_clusters != -1)
    
        if np.sum(mascara_sem_ruido) > 5 and len(set(labels_clusters[mascara_sem_ruido])) > 1:
            X_validacao = X_scaled[mascara_sem_ruido]
            labels_validacao = labels_clusters[mascara_sem_ruido]
            
            score_silhueta = silhouette_score(X_validacao, labels_validacao)
            
            score_db = davies_bouldin_score(X_validacao, labels_validacao)
        else:
            score_silhueta = None
            score_db = None
    else:
        score_silhueta = None
        score_db = None
    return score_silhueta,score_db, n_ruido, p_ruido
    
def montar_conclusao():
    if "y_true_m1" in st.session_state and "y_pred_m1" in st.session_state:
        y_true_m1 = st.session_state.y_true_m1
        y_pred_m1 = st.session_state.y_pred_m1
        
        acc_m1 = accuracy_score(y_true_m1, y_pred_m1) * 100
        report_m1 = classification_report(y_true_m1, y_pred_m1, output_dict=True, zero_division=0)
        
        f1_m1 = report_m1.get('1', report_m1.get('1.0', report_m1.get(1, {'f1-score': 0.0})))['f1-score']

        y_true_m2 = st.session_state.y_true_m2
        y_pred_m2 = st.session_state.y_pred_m2
        
        acc_m2 = accuracy_score(y_true_m2, y_pred_m2) * 100
        report_m2 = classification_report(y_true_m2, y_pred_m2, output_dict=True, zero_division=0)
        
        prec_m2_c3 = report_m2.get('2', report_m2.get('2.0', report_m2.get(2, {'precision': 0.0})))['precision']

        dados_resumo = {
            "Objetivo do Modelo": [
                "Modelo 1: Presença de Câncer (Triagem)", 
                "Modelo 1: Presença de Câncer (Triagem)", 
                "Modelo 2: Subtipagem Histológica (Biópsia)", 
                "Modelo 2: Subtipagem Histológica (Biópsia)"
            ],
            "Métrica-Chave": [
                "Acurácia Global", 
                "F1-Score (Classe Doente)", 
                "Acurácia Global", 
                "Precisão (Pequenas Células - SCLC)"
            ],
            "Valor Obtido": [
                f"{acc_m1:.1f}%", 
                f"{f1_m1:.2f}", 
                f"{acc_m2:.1f}%", 
                f"{prec_m2_c3:.2f}"
            ]
        }

        df_resumo_conclusao = pd.DataFrame(dados_resumo)
        st.dataframe(
            df_resumo_conclusao,
            width='stretch',
            hide_index=True,
            column_config={
                "Objetivo do Modelo": st.column_config.TextColumn("Escopo"),
                "Métrica-Chave": st.column_config.TextColumn("Métrica"),
                "Valor Obtido": st.column_config.TextColumn("Desempenho"),
            }
        )

    else:
        st.warning("Nenhuma métrica foi gerada. Necessário processamento dos modelos nas etapas anteriores.")
        
def gerar_tabela_metricas_dinamica():
    y_true_m1 = st.session_state.y_true_m1 
    y_pred_m1 = st.session_state.y_pred_m1 
    y_true_m2 = st.session_state.y_true_m2 
    y_pred_m2 = st.session_state.y_pred_m2
    
    rep_m1 = classification_report(y_true_m1, y_pred_m1, output_dict=True, zero_division=0)
    acc_m1 = accuracy_score(y_true_m1, y_pred_m1)
    
    m1_c1 = rep_m1.get('0', rep_m1.get('0.0', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}))
    m1_c2 = rep_m1.get('1', rep_m1.get('1.0', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}))
    m1_macro = rep_m1['macro avg']
    m1_weighted = rep_m1['weighted avg']

    rep_m2 = classification_report(y_true_m2, y_pred_m2, output_dict=True, zero_division=0)
    acc_m2 = accuracy_score(y_true_m2, y_pred_m2)
    
    m2_c1 = rep_m2.get('0', rep_m2.get('0.0', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}))
    m2_c2 = rep_m2.get('1', rep_m2.get('1.0', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}))
    m2_c3 = rep_m2.get('2', rep_m2.get('2.0', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}))
    m2_macro = rep_m2['macro avg']
    m2_weighted = rep_m2['weighted avg']

    dados_metricas = {
        "Modelo": [
            "Modelo 1: Presença (Binário)", "Modelo 1: Presença (Binário)", "Modelo 1: Presença (Binário)", "Modelo 1: Presença (Binário)", "Modelo 1: Presença (Binário)",
            "Modelo 2: Subtipo (Multiclasse)", "Modelo 2: Subtipo (Multiclasse)", "Modelo 2: Subtipo (Multiclasse)", "Modelo 2: Subtipo (Multiclasse)", "Modelo 2: Subtipo (Multiclasse)"
        ],
        "Métrica": [
            "Precisão", "Recall", "F1-Score", "Suporte", "Acurácia",
            "Precisão", "Recall", "F1-Score", "Suporte", "Acurácia"
        ],
        "Classe 1": [
            f"{m1_c1['precision']:.2f}", f"{m1_c1['recall']:.2f}", f"{m1_c1['f1-score']:.2f}", f"{int(m1_c1['support'])}", "-",
            f"{m2_c1['precision']:.2f}", f"{m2_c1['recall']:.2f}", f"{m2_c1['f1-score']:.2f}", f"{int(m2_c1['support'])}", "-"
        ],
        "Classe 2": [
            f"{m1_c2['precision']:.2f}", f"{m1_c2['recall']:.2f}", f"{m1_c2['f1-score']:.2f}", f"{int(m1_c2['support'])}", "-",
            f"{m2_c2['precision']:.2f}", f"{m2_c2['recall']:.2f}", f"{m2_c2['f1-score']:.2f}", f"{int(m2_c2['support'])}", "-"
        ],
        "Classe 3": [
            "-", "-", "-", "-", "-",
            f"{m2_c3['precision']:.2f}", f"{m2_c3['recall']:.2f}", f"{m2_c3['f1-score']:.2f}", f"{int(m2_c3['support'])}", "-"
        ],
        "Média": [
            f"{m1_macro['precision']:.2f}", f"{m1_macro['recall']:.2f}", f"{m1_macro['f1-score']:.2f}", f"{int(m1_macro['support'])}", "-",
            f"{m2_macro['precision']:.2f}", f"{m2_macro['recall']:.2f}", f"{m2_macro['f1-score']:.2f}", f"{int(m2_macro['support'])}", "-"
        ],
        "Média Ponderada": [
            f"{m1_weighted['precision']:.2f}", f"{m1_weighted['recall']:.2f}", f"{m1_weighted['f1-score']:.2f}", f"{int(m1_weighted['support'])}", f"{acc_m1:.3f} ({acc_m1*100:.1f}%)",
            f"{m2_weighted['precision']:.2f}", f"{m2_weighted['recall']:.2f}", f"{m2_weighted['f1-score']:.2f}", f"{int(m2_weighted['support'])}", f"{acc_m2:.3f} ({acc_m2*100:.1f}%)"
        ]
    }
    
    df_resultado = pd.DataFrame(dados_metricas)

    return df_resultado.copy()

def gerar_matriz_confusao():
    if "y_true_m1" in st.session_state and "y_pred_m1" in st.session_state:
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.markdown("**Matriz de Confusão:** Modelo 1 (Presença de Câncer)")
            
            y_true_1 = st.session_state.y_true_m1
            y_pred_1 = st.session_state.y_pred_m1
            cm_m1 = confusion_matrix(y_true_1, y_pred_1)
            
            classes_existentes_m1 = np.unique(np.concatenate([y_true_1, y_pred_1]))
            labels_m1_map = {0: "Sem Câncer (0)", 1: "Com Câncer (1)"}
            eixos_m1 = [labels_m1_map[cls] for cls in classes_existentes_m1 if cls in labels_m1_map]
            
            fig_cm1 = px.imshow(
                cm_m1,
                labels=dict(x="Diagnóstico Predito", y="Diagnóstico Real", color="Pacientes"),
                x=eixos_m1, 
                y=eixos_m1, 
                text_auto=True,
                color_continuous_scale="Blues",
                height=340
            )
            fig_cm1.update_layout(margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
            st.plotly_chart(fig_cm1, width='stretch')
            
            if len(cm_m1) > 1:
                recall_dinamico = cm_m1[1][1] / (cm_m1[1][0] + cm_m1[1][1]) if (cm_m1[1][0] + cm_m1[1][1]) > 0 else 0
                st.caption(f"Recall: {recall_dinamico:.2f}")

        with col_cm2:
            st.markdown("**Matriz de Confusão:** Modelo 2 (Subtipo de Câncer)")
            
            cm_m2 = confusion_matrix(
                st.session_state.y_true_m2, 
                st.session_state.y_pred_m2, 
                labels=[0, 1, 2]
            )
            eixos_m2 = ["Adenocarcinoma", "Escamosas", "Pequenas Células"]
            fig_cm2 = px.imshow(
                cm_m2,
                labels=dict(x="Subtipo Predito", y="Subtipo Real", color="Pacientes"),
                x=eixos_m2,
                y=eixos_m2,
                text_auto=True,
                color_continuous_scale="rdbu_r",
                height=340
            )
            fig_cm2.update_layout(margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
            st.plotly_chart(fig_cm2, width='stretch')

    else:
        st.warning(" Nenhuma matriz preditiva encontrada.")

def gerar_tabela_metricas_detalhada():
    chaves_obrigatorias = ["y_true_m1", "y_pred_m1", "y_true_m2", "y_pred_m2"]
    dados_ausentes = [
        chave for chave in chaves_obrigatorias 
        if chave not in st.session_state or st.session_state[chave] is None
    ]
    
    if dados_ausentes:
        st.warning("**Histórico de predições incompleto ou não encontrado!**")
        st.info("Por favor, acesse a etapa de **Mineração de Dados** e execute o treinamento dos modelos para gerar os vetores de teste e predição antes de visualizar as métricas.")
        return
    
    y_true_m1 = st.session_state.y_true_m1
    y_pred_m1 = st.session_state.y_pred_m1
    y_true_m2 = st.session_state.y_true_m2
    y_pred_m2 = st.session_state.y_pred_m2

    rep_m1 = classification_report(y_true_m1, y_pred_m1, output_dict=True, zero_division=0)
    rep_m2 = classification_report(y_true_m2, y_pred_m2, output_dict=True, zero_division=0)

    acc_m1 = accuracy_score(y_true_m1, y_pred_m1)
    acc_m2 = accuracy_score(y_true_m2, y_pred_m2)

    html_style = """
        <style>
            .kdd-table { width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 15px 0; font-size: 14px; color: #333333; background-color: #FFFFFF; }
            .kdd-table th { background-color: #1F619E; color: white; padding: 12px 10px; border: 1px solid #C4D6E6; text-align: left; font-weight: 600; }
            .kdd-table td { padding: 10px; border: 1px solid #E2E8F0; text-align: left; }
            .row-even { background-color: #FFFFFF; }
            .row-odd { background-color: #F8FAFC; }
            .acc-row { background-color: #EFF6FF; font-weight: bold; text-align: center !important; color: #1E40AF; }
            .lbl-bold { font-weight: bold; vertical-align: middle; }
        </style>
        """

    html_body_m1 = f"""
        <table class="kdd-table">
            <thead>
                <tr>
                    <th>Modelo</th>
                    <th>Métrica</th>
                    <th>Classe 1</th>
                    <th>Classe 2</th>
                    <th>Classe 3</th>
                    <th>Média (Macro)</th>
                    <th>Média Ponderada</th>
                </tr>
            </thead>
            <tbody>
                <tr class="row-even">
                    <td rowspan="5" class="lbl-bold" style="width: 25%;">Modelo 1: Presença de Câncer<br><small style="color: #64748B; font-weight: normal;">(Binário)</small></td>
                    <td>Precisão</td>
                    <td>{rep_m1.get('0', {'precision': 0.0})['precision']:.2f}</td>
                    <td>{rep_m1.get('1', {'precision': 0.0})['precision']:.2f}</td>
                    <td>-</td>
                    <td>{rep_m1['macro avg']['precision']:.2f}</td>
                    <td>{rep_m1['weighted avg']['precision']:.2f}</td>
                </tr>
                <tr class="row-even">
                    <td>Recall</td>
                    <td>{rep_m1.get('0', {'recall': 0.0})['recall']:.2f}</td>
                    <td>{rep_m1.get('1', {'recall': 0.0})['recall']:.2f}</td>
                    <td>-</td>
                    <td>{rep_m1['macro avg']['recall']:.2f}</td>
                    <td>{rep_m1['weighted avg']['recall']:.2f}</td>
                </tr>
                <tr class="row-even">
                    <td>F1-Score</td>
                    <td>{rep_m1.get('0', {'f1-score': 0.0})['f1-score']:.2f}</td>
                    <td>{rep_m1.get('1', {'f1-score': 0.0})['f1-score']:.2f}</td>
                    <td>-</td>
                    <td>{rep_m1['macro avg']['f1-score']:.2f}</td>
                    <td>{rep_m1['weighted avg']['f1-score']:.2f}</td>
                </tr>
                <tr class="row-even">
                    <td>Suporte</td>
                    <td>{int(rep_m1.get('0', {'support': 0})['support'])}</td>
                    <td>{int(rep_m1.get('1', {'support': 0})['support'])}</td>
                    <td>-</td>
                    <td>{int(rep_m1['macro avg']['support'])}</td>
                    <td>{int(rep_m1['weighted avg']['support'])}</td>
                </tr>
                <tr class="row-even">
                    <td>Acurácia</td>
                    <td colspan="5" class="acc-row">Acurácia Global: {acc_m1:.3f} ({acc_m1*100:.1f}%)</td>
                </tr>
            </tbody>
        </table>
        """
        
    st.markdown(html_style + html_body_m1, unsafe_allow_html=True)

    html_body_m2 = f"""
        <table class="kdd-table">
            <thead>
                <tr>
                    <th>Modelo</th>
                    <th>Métrica</th>
                    <th>Classe 1</th>
                    <th>Classe 2</th>
                    <th>Classe 3</th>
                    <th>Média (Macro)</th>
                    <th>Média Ponderada</th>
                </tr>
            </thead>
            <tbody>
                <tr class="row-odd">
                    <td rowspan="5" class="lbl-bold">Modelo 2: Subtipo de Câncer<br><small style="color: #64748B; font-weight: normal;">(Multiclasse)</small></td>
                    <td>Precisão</td>
                    <td>{rep_m2.get('0', {'precision': 0.0})['precision']:.2f}</td>
                    <td>{rep_m2.get('1', {'precision': 0.0})['precision']:.2f}</td>
                    <td>{rep_m2.get('2', {'precision': 0.0})['precision']:.2f}</td>
                    <td>{rep_m2['macro avg']['precision']:.2f}</td>
                    <td>{rep_m2['weighted avg']['precision']:.2f}</td>
                </tr>
                <tr class="row-odd">
                    <td>Recall</td>
                    <td>{rep_m2.get('0', {'recall': 0.0})['recall']:.2f}</td>
                    <td>{rep_m2.get('1', {'recall': 0.0})['recall']:.2f}</td>
                    <td>{rep_m2.get('2', {'recall': 0.0})['recall']:.2f}</td>
                    <td>{rep_m2['macro avg']['recall']:.2f}</td>
                    <td>{rep_m2['weighted avg']['recall']:.2f}</td>
                </tr>
                <tr class="row-odd">
                    <td>F1-Score</td>
                    <td>{rep_m2.get('0', {'f1-score': 0.0})['f1-score']:.2f}</td>
                    <td>{rep_m2.get('1', {'f1-score': 0.0})['f1-score']:.2f}</td>
                    <td>{rep_m2.get('2', {'f1-score': 0.0})['f1-score']:.2f}</td>
                    <td>{rep_m2['macro avg']['f1-score']:.2f}</td>
                    <td>{rep_m2['weighted avg']['f1-score']:.2f}</td>
                </tr>
                <tr class="row-odd">
                    <td>Suporte</td>
                    <td>{int(rep_m2.get('0', {'support': 0})['support'])}</td>
                    <td>{int(rep_m2.get('1', {'support': 0})['support'])}</td>
                    <td>{int(rep_m2.get('2', {'support': 0})['support'])}</td>
                    <td>{int(rep_m2['macro avg']['support'])}</td>
                    <td>{int(rep_m2['weighted avg']['support'])}</td>
                </tr>
                <tr class="row-odd">
                    <td>Acurácia</td>
                    <td colspan="5" class="acc-row">Acurácia Global: {acc_m2:.3f} ({acc_m2*100:.1f}%)</td>
                </tr>
            </tbody>
        </table>
        """

    st.markdown(html_style + html_body_m2, unsafe_allow_html=True)
    
def mostrar_feature_importance(df_filtrado):
    st.write("Selecionar qual etapa do pipeline de IA você deseja inspecionar para ver quais variáveis foram mais determinantes:")
    
    escolha_alvo = st.radio(
        "Escolha o Alvo da IA:",
        ["Presença de Câncer (Modelo 1 - Triagem)", "Subtipo de Câncer (Modelo 2 - Subtipagem)"],
        horizontal=True
    )
    
    if "Presença" in escolha_alvo:
        target_col = 'cancer_presence'
        titulo_grafico = "Contribuição das Variáveis na Triagem (Presença de Câncer)"
    else:
        target_col = 'cancer_subtype'
        titulo_grafico = "Contribuição das Variáveis na Subtipagem Histológica"

    features = ['nodule_size_mm', 'HU_mean', 'HU_std', 'PET_SUVmax', 'PET_SUVmean', 'patient_age', 'PD-L1_expression_level', 'tumor_mutational_burden']
    
    if target_col in df_filtrado.columns and all(col in df_filtrado.columns for col in features):
        df_ml = df_filtrado[features + [target_col]].dropna()
        if len(df_ml) == 0:
            st.warning("Não há dados suficientes (após remoção de nulos) para calcular a importância dos atributos.")
            return

        X = df_ml[features]
        y = df_ml[target_col]
        
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)
        
        importances = modelo.feature_importances_
        
        df_importance = pd.DataFrame({
            'Atributo': features,
            'Importância': importances
        }).sort_values(by='Importância', ascending=True)
        
        fig_importancia = px.bar(
            df_importance,
            x='Importância',
            y='Atributo',
            orientation='h',
            title=titulo_grafico,
            labels={'Importância': 'Grau de Importância (0 a 1)', 'Atributo': 'Características'},
            color='Importância',
            color_continuous_scale='Blues'
        )
        
        fig_importancia.update_layout(height=350, showlegend=False, margin=dict(l=150, r=30, t=50, b=50))
        
        st.plotly_chart(fig_importancia, width='stretch')
        
        top_feature = df_importance.iloc[-1]['Atributo']
        st.info(f"**Insight:** Para o alvo **{target_col}**, a característica **{top_feature}** foi a mais decisiva para o aprendizado do modelo.")  
        
        if target_col == 'cancer_subtype':
            mostrar_shap_subtipos()
        
        
    else:
        st.warning("Colunas necessárias para o cálculo de Machine Learning não foram encontradas.")


def mostrar_shap_subtipos():
    st.markdown("**Matriz SHAP Dinâmica por Subtipo**")
    
    df_teste = st.session_state.df_test

    if "pesos_reais_shap" not in st.session_state or st.session_state.pesos_reais_shap is None:
        st.warning("Ajuste de pesos não encontrado. Por favor, execute o treinamento da Rede Neural na aba ao lado primeiro.")
        return

    pesos_base_reais = st.session_state.pesos_reais_shap
    features_presentes = [
        col for col in df_teste.columns 
        if col not in ["cancer_presence", "cancer_subtype"]
    ]
    if not features_presentes:
        st.error("O DataFrame de teste não possui colunas válidas para exibição.")
        return

    dicionario_nomes = {
        'nodule_size_mm': 'Tamanho do Nódulo (mm)',
        'HU_mean': 'Densidade Média (HU)',
        'HU_std': 'Desvio Padrão Densidade (HU)',
        'GLCM_contrast': 'Contraste de Textura (GLCM)',
        'PET_SUVmax': 'Captação Máxima (PET SUVmax)',
        'PET_SUVmean': 'Captação Média (PET SUVmean)',
        'patient_age': 'Idade do Paciente',
        'smoking_history': 'Histórico de Tabagismo',
        'family_history': 'Histórico Familiar',
        'tumor_location': 'Localização do Tumor',
        'radiation_therapy': 'Tratamento por Radiação',
        'immunotherapy_received': 'Imunoterapia Recebida',
        'EGFR_mutation_status': 'Status de Mutação EGFR',
        'ALK_fusion_status': 'Status de Fusão ALK',
        'tumor_mutational_burden': 'Carga Mutacional Tumoral (TMB)'
    }
    labels_amigaveis = [dicionario_nomes.get(f, f) for f in features_presentes]

    pesos_adeno = [pesos_base_reais.get("Adenocarcinoma", {}).get(f, 0.0) for f in features_presentes]
    pesos_squamous = [pesos_base_reais.get("Squamous Cell", {}).get(f, 0.0) for f in features_presentes]
    pesos_sclc = [pesos_base_reais.get("SCLC", {}).get(f, 0.0) for f in features_presentes]
    pesos_other = [pesos_base_reais.get("Other", {}).get(f, 0.0) for f in features_presentes]

    fig = go.Figure()
    
    cores_adeno, text_pos_adeno = obter_cores_e_posicao(pesos_adeno)
    fig.add_trace(go.Bar(
        x=pesos_adeno, y=labels_amigaveis, orientation='h',
        marker=dict(color=cores_adeno, line=dict(width=0)), 
        name="Adenocarcinoma", visible=True,
        text=[f"{v:+.4f}" for v in pesos_adeno], textposition=text_pos_adeno,
        textfont=dict(size=10, color="white")
    ))

    cores_squamous, text_pos_squamous = obter_cores_e_posicao(pesos_squamous)
    fig.add_trace(go.Bar(
        x=pesos_squamous, y=labels_amigaveis, orientation='h',
        marker=dict(color=cores_squamous, line=dict(width=0)), 
        name="Squamous Cell", visible=False,
        text=[f"{v:+.4f}" for v in pesos_squamous], textposition=text_pos_squamous,
        textfont=dict(size=10, color="white")
    ))

    cores_sclc, text_pos_sclc = obter_cores_e_posicao(pesos_sclc)
    fig.add_trace(go.Bar(
        x=pesos_sclc, y=labels_amigaveis, orientation='h',
        marker=dict(color=cores_sclc, line=dict(width=0)), 
        name="SCLC", visible=False,
        text=[f"{v:+.4f}" for v in pesos_sclc], textposition=text_pos_sclc,
        textfont=dict(size=10, color="white")
    ))

    cores_other, text_pos_other = obter_cores_e_posicao(pesos_other)
    fig.add_trace(go.Bar(
        x=pesos_other, y=labels_amigaveis, orientation='h',
        marker=dict(color=cores_other, line=dict(width=0)), 
        name="Other", visible=False,
        text=[f"{v:+.4f}" for v in pesos_other], textposition=text_pos_other,
        textfont=dict(size=10, color="white")
    ))

    fig.update_layout(
        updatemenus=[
            dict(
                active=3 if pesos_other == pesos_adeno else 0,
                buttons=list([
                    dict(label="Foco: Adenocarcinoma", method="update", args=[{"visible": [True, False, False, False]}, {"title": "<b>Assinatura Espacial: Adenocarcinoma</b>"}]),
                    dict(label="Foco: Squamous Cell Carcinoma", method="update", args=[{"visible": [False, True, False, False]}, {"title": "<b>Assinatura Espacial: Squamous Cell</b>"}]),
                    dict(label="Foco: SCLC (Células Pequenas)", method="update", args=[{"visible": [False, False, True, False]}, {"title": "<b>Assinatura Espacial: SCLC</b>"}]),
                    dict(label="Foco: Other (Outros Subtipos)", method="update", args=[{"visible": [False, False, False, True]}, {"title": "<b>Assinatura Espacial: Outros Subtipos</b>"}]),
                ]),
                direction="down", showactive=True,
                x=0.0, xanchor="left", y=1.22, yanchor="top",
                bgcolor="#FFFFFF", bordercolor="#0E81FC", font=dict(size=12, color="#053FA1")
            )
        ]
    )
   
    fig.update_layout(
        title="",
        title_font=dict(size=16, color="#1A202C"),
        xaxis_title="Contribuição Marginal para o Alvo (Importância de Permutação)",
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False, tickfont=dict(size=11, color="#024AC6")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#0E2C61")),
        bargap=0.2, 
        barmode='overlay', 
        template="plotly_white",
        height=550,
        margin=dict(t=130, b=50, l=160, r=40), 
        showlegend=False
    )

    fig.add_vline(x=0, line_width=1.5, line_color="#065FF9")
    st.plotly_chart(fig, width='stretch')
    
def obter_cores_e_posicao(lista_pesos):
        cores = ["#236EBE" if v >= 0 else '#C53030' for v in lista_pesos]
        posicoes = ['outside' if abs(v) < 0.02 else 'inside' for v in lista_pesos]
        return cores, posicoes
    
      
  