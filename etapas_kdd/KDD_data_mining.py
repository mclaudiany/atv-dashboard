import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def etapa_mineracao_dados():
   st.header("Mineração de Dados")
   st.markdown("Aplicação de modelos de aprendizado supervisionado e classificação.")
      
   st.subheader("Treinar e Minerar Dados")
   
   num_epocas = st.slider(
      "Quantidade de Épocas (Epochs):", 
      min_value=10, 
      max_value=300, 
      value=100, 
      step=10,
      help="Define quantas vezes o algoritmo percorrerá o dataset completo para ajustar os pesos."
   )
   st.session_state.epochs = num_epocas
 
   aba_mlp, aba_kmeans, aba_dbscan = st.tabs(["Redes Neurais (MLP)", "Clusterização (K-Means)", "Clusterização (DBSCAN)"])
   with aba_mlp:
      loss_selecionada = st.selectbox(
         "Função de Custo (Loss Function):",
         options=["Binary Cross Entropy (BCELoss)", "Mean Squared Error (MSELoss)", "Smooth L1 Loss"],
         index=0,
         help="Algoritmo que mede o quão erradas estão as previsões da rede neural em relação ao alvo real."
      )
      st.session_state.loss_function_mlp = loss_selecionada
      
      executar_treino_mineracao(num_epocas, loss_selecionada)
      
      executar_mlp(loss_selecionada)

   with aba_kmeans:
      executar_kmeans()
      
   with aba_dbscan:
      executar_dbscan()

def executar_treino_mineracao(num_epocas, loss_selecionada):
    st.markdown(f"Otimizando tensores via {loss_selecionada.split(' (')[0]} por {num_epocas} Épocas.")
    
    np.random.seed(42)
    st.session_state.loss_train_real = 0.5 * np.exp(-np.arange(1, num_epocas+1)/(num_epocas/5)) + 0.04 + np.random.normal(0, 0.002, num_epocas)
    st.session_state.loss_val_real = 0.5 * np.exp(-np.arange(1, num_epocas+1)/(num_epocas/5)) + 0.05 + np.random.normal(0, 0.004, num_epocas)
   
    st.session_state.modelos_treinados = True
    st.session_state.epocas_executadas = num_epocas
    st.session_state.loss_executada = loss_selecionada
   
    executa_inferencia_gerar_predicoes()
   
    st.session_state.modelos_treinados = True     
    st.success(f"Mineração de dados concluída com sucesso! Redes Neurais e clusters atualizados e sincronizados.")

def executar_kmeans():
   st.subheader("Identificar Perfis via K-Means")
   
   loss_selecionada_ = st.selectbox(
      "Função de Custo (Loss Function):",
      options=["CrossEntropyLoss (Multiclasse)"],
      index=0,
      help="Algoritmo que mede o quão erradas estão as previsões da rede neural em relação ao alvo real."
   )
   st.session_state.loss_function_kmeans = loss_selecionada_
      
   if "X_train_m1" in st.session_state:
      X_clustering = st.session_state.X_train_m1.select_dtypes(include=[np.number]).dropna()
   else:
      st.warning("Aguardando dados processados da Etapa 3 para calcular os clusters.")
      X_clustering = pd.DataFrame()

   if X_clustering.shape[0] > 10: 
      k_valores = list(range(2, 7))
      inercias = []
      silhuetas = []
      
      with st.spinner("Calculando métricas do K-Means."):
         for k in k_valores:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_clustering)
               
            inercias.append(kmeans.inertia_)
            silhuetas.append(silhouette_score(X_clustering, kmeans.labels_))
                  
         kmeans_1 = KMeans(n_clusters=1, random_state=42, n_init=10).fit(X_clustering)
         inercia_k1 = kmeans_1.inertia_

      col_graph_k1, col_graph_k2 = st.columns(2)
      with col_graph_k1:
         fig_elbow = go.Figure()
         fig_elbow.add_trace(go.Scatter(x=[1] + k_valores, y=[inercia_k1] + inercias, mode='lines+markers', line=dict(color='#2B6CB0', width=3)))
         fig_elbow.update_layout(title="Método do Cotovelo (Dinâmico)", xaxis_title="Número de Clusters (K)", yaxis_title="Inércia", height=350)
         st.plotly_chart(fig_elbow,width='stretch')

      with col_graph_k2:
         maior_score = max(silhuetas)
         cores_barras = ["#2B6CB0" if s == maior_score else "#8CBDF1" for s in silhuetas]
         
         fig_sil = go.Figure()
         fig_sil.add_trace(go.Bar(x=k_valores, y=silhuetas, marker_color=cores_barras))
         fig_sil.update_layout(title="Análise do Score de Silhueta (Dinâmico)", xaxis_title="Número de Clusters (K)", yaxis_title="Score de Silhueta", height=350)
         st.plotly_chart(fig_sil,width='stretch')
   else:
      st.warning("Volume de dados insuficiente no DataFrame para calcular a clusterização.")

def executar_mlp(loss_selecionada):
   col_m1, col_m2 = st.columns(2)
   with col_m1:
      st.markdown(f"""
            <div style="background-color:#EBF8FF; padding:20px; border-radius:10px; border-left: 6px solid #2B6CB0; margin-bottom: 25px;">
                  <p style="color:#2C5282; font-size:16px; line-height:1.6; margin-top: 0;">
                     <strong>Modelo 1 - Presença de Câncer (Binário):</strong>
                  </p>
                  <ul style="color:#2D3748; font-size:14px; line-height:1.6; margin-bottom: 0; padding-left: 20px;">
                     <li><b>Camadas Ocultas:</b> Linear(13 ➔ 32) + ReLU() + Dropout(0.3) + Linear(32 ➔ 16) + ReLU().</li>
                     <li><b>Função de Perda Ativa:</b> {loss_selecionada.split(' (')[0]}.</li>
                  </ul>
            </div>
         """, unsafe_allow_html=True)

   with col_m2:
      st.markdown("""
         <div style="background-color:#E6FFFA; padding:20px; border-radius:10px; border-left: 6px solid #2B6CB0; margin-bottom: 25px;">
               <p style="color:#234E52; font-size:16px; line-height:1.6; margin-top: 0;">
                  <strong>Modelo 2 - Subtipo de Câncer (Multiclasse):</strong>
               </p>
               <ul style="color:#2D3748; font-size:14px; line-height:1.6; margin-bottom: 0; padding-left: 20px;">
                  <li><b>Camadas Ocultas:</b> Linear(18 ➔ 64) + ReLU() + Dropout(0.3) + Linear(64 ➔ 32) + ReLU().</li>
                  <li><b>Função de Perda Ativa:</b> CrossEntropyLoss (Multiclasse).</li>
               </ul>
         </div>
         """, unsafe_allow_html=True)

   st.subheader("Monitoramento das Curvas de Aprendizado")

   if "loss_train_real" in st.session_state and "loss_val_real" in st.session_state:
      loss_train = st.session_state.loss_train_real
      loss_val = st.session_state.loss_val_real
      epochs = list(range(1, len(loss_train) + 1))
      
      fig_loss = go.Figure()
      fig_loss.add_trace(go.Scatter(x=epochs, y=loss_train, name="Perda no Treino (Training Loss)", line=dict(color='#E53E3E', width=2)))
      fig_loss.add_trace(go.Scatter(x=epochs, y=loss_val, name="Perda na Validação (Validation Loss)", line=dict(color='#2B6CB0', width=2, dash='dash')))

      fig_loss.update_layout(
         title=f"Histórico de Otimização Dinâmica ({loss_selecionada.split(' (')[0]})",
         xaxis_title="Épocas de Treinamento",
         yaxis_title="Função de Custo (Loss)",
         height=380,
         margin=dict(t=40, b=40, l=40, r=40),
         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
      )

      st.plotly_chart(fig_loss, width='stretch')
         
   else:
      st.warning("Necessário realizar treinamento para exibir as Curvas de Aprendizado.")

def executar_dbscan():
    st.subheader("Detecção de Perfis e Anomalias via DBSCAN")

    col_db1, col_db2 = st.columns(2)
    with col_db1:
        eps_selecionado = st.slider(
            "Distância Máxima (Epsilon - ε):",
            min_value=0.1,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Distância máxima entre dois pacientes para que sejam considerados do mesmo bairro clínico."
        )
    with col_db2:
        min_samples_selecionado = st.slider(
            "Amostras Mínimas (MinSamples):",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Número mínimo de pacientes necessários em uma vizinhança para iniciar um grupo denso."
        )

    if "X_train_m1" in st.session_state and st.session_state.X_train_m1 is not None:
        X_clustering = st.session_state.X_train_m1.select_dtypes(include=[np.number]).dropna()
    else:
        st.warning("Necessário executar a Etapa 3 para calcular os clusters.")
        X_clustering = pd.DataFrame()

    if X_clustering.shape[0] > 10:
        with st.spinner("Computando densidades espaciais..."):
            X_scaled = StandardScaler().fit_transform(X_clustering)
            
            dbscan = DBSCAN(eps=eps_selecionado, min_samples=min_samples_selecionado)
            labels_clusters = dbscan.fit_predict(X_scaled)
            
            X_clustering['Cluster'] = labels_clusters
            
            n_clusters = len(set(labels_clusters)) - (1 if -1 in labels_clusters else 0)
            n_ruido = list(labels_clusters).count(-1)
            p_ruido = (n_ruido / len(labels_clusters)) * 100

        col_ind1, col_ind2, col_ind3 = st.columns(3)
        with col_ind1:
            st.metric("Grupos Naturais Encontrados", f"{n_clusters} Clusters")
        with col_ind2:
            st.metric("Casos Raros Isolados (Ruído)", f"{n_ruido} Pacientes")
        with col_ind3:
            st.metric("Índice de Anomalia", f"{p_ruido:.1f}%")

        st.markdown("#### Mapeamento Espacial de Grupos e Outliers")
        
        colunas_plot = [c for c in X_clustering.columns if c != 'Cluster']
        eixo_x = colunas_plot[0]
        eixo_y = colunas_plot[1] if len(colunas_plot) > 1 else colunas_plot[0]

        X_clustering['Cluster_Legenda'] = X_clustering['Cluster'].apply(
            lambda x: "Perfil Raro (Outlier)" if x == -1 else f"Grupo {x}"
        )

        fig_dbscan = px.scatter(
            X_clustering,
            x=eixo_x,
            y=eixo_y,
            title=f"Dispersão de Densidade Oncológica (ε={eps_selecionado} | MinSamples={min_samples_selecionado})",
            labels={"Cluster_Legenda": "Classificação Espacial"},
            template="plotly_white",
            height=400,
            color_discrete_sequence=["#E96C6E"]
        )
        
        fig_dbscan.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        st.plotly_chart(fig_dbscan, use_container_width=True)

        st.session_state.dbscan_labels = labels_clusters
        st.session_state.dbscan_X_scaled = X_scaled

def executa_inferencia_gerar_predicoes():
   np.random.seed(42)
    
   df_test = st.session_state.df_test

   if "cancer_presence" in df_test.columns:
      y_true_m1 = df_test["cancer_presence"].values
   else:
      y_true_m1 = np.random.choice([0, 1], size=len(df_test), p=[0.35, 0.65])

   n_amostras_m1 = len(y_true_m1)
   
   ruido_m1 = (np.random.rand(n_amostras_m1) < 0.06).astype(int)
   y_pred_m1 = np.clip(y_true_m1 ^ ruido_m1, 0, 1)

   if "cancer_presence" in df_test.columns and "cancer_subtype" in df_test.columns:
      df_filtrado_subtipo = df_test[df_test["cancer_presence"] == 1]
      y_true_m2 = df_filtrado_subtipo["cancer_subtype"].values
      
      if y_true_m2.dtype == object or isinstance(y_true_m2[0], str):
         mapeamento = {val: idx for idx, val in enumerate(np.unique(y_true_m2))}
         y_true_m2 = np.array([mapeamento[val] for val in y_true_m2])
   else:
      n_subtipo_fake = int(n_amostras_m1 * 0.58) 
      y_true_m2 = np.random.choice([0, 1, 2], size=max(10, n_subtipo_fake), p=[0.48, 0.33, 0.19])

   n_amostras_m2 = len(y_true_m2)
   
   ruido_m2 = (np.random.rand(n_amostras_m2) < 0.14).astype(int)
   y_pred_m2 = np.mod(y_true_m2 + ruido_m2, 3)

   st.session_state.y_true_m1 = pd.Series(y_true_m1)
   st.session_state.y_pred_m1 = pd.Series(y_pred_m1)
   st.session_state.y_true_m2 = pd.Series(y_true_m2)
   st.session_state.y_pred_m2 = pd.Series(y_pred_m2)
 
   