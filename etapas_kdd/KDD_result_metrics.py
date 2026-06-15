import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,silhouette_score, davies_bouldin_score

def etapa_resultado_metricas():
    st.header("Avaliação e Interpretação de Métricas")
    st.subheader("Validação Estatística das Redes Neurais")

    st.markdown("Esta etapa garante que os modelos possuem capacidade de generalização através do cruzamento de dados com a base de teste.")

    epocas = st.session_state.get("epocas_executadas", 100)
    loss_name_mlp = st.session_state.loss_function_mlp
    loss_mlp_limpo = loss_name_mlp.split(' (')[0] if loss_name_mlp else "N/A"
    loss_name_kmeans = st.session_state.loss_function_kmeans
    
    st.info(
        f"**Auditoria do Pipeline Ativo:** Métricas consolidadas após otimização profunda executada por **{epocas} épocas**. "
        f"Critérios de custo utilizados: MLP - {loss_mlp_limpo} e KMeans - {loss_name_kmeans}."
    )
    
    st.subheader("Matriz de Desempenho dos Modelos")
    st.markdown("Tabela comparativa estruturada com as métricas de validação.")
    gerar_tabela_metricas_detalhada()
    
    st.subheader("Análise de Erros por Matriz de Confusão")
    gerar_matriz_confusao()

    st.subheader("Índices de Validação de Densidade Estritamente Numérico")
    gerar_indice_densidade()
    
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
            use_container_width=True,
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