# Detecção e Subtipagem de Câncer de Pulmão

Este projeto é um artefato técnico desenvolvido para a disciplina de Projeto Interdisciplinar para Sistemas de Informação 3 (PISI3) do 3° período do curso de Bacharelado em Sistemas de Informação (BSI) e do Programa de Pós-Graduação em Informática Aplicada da Universidade Federal Rural de Pernambuco (UFRPE).

**Autora:** Claudiany Martins (claudiany.martins@ufrpe.br)

---

### Dataset de Câncer de Pulmão

Dataset [Lung Cancer Dataset](https://www.kaggle.com/datasets/datasetengineer/lungcanc2024?utm_source=chatgpt.com&select=LungCanC2024_Dataset.csv).

O projeto integra técnicas de **Redes Neurais (MLP)** para classificação e **K-Means** e **DBSCAN** para a descoberta de padrões não supervisionados, seguindo o processo **KDD (Knowledge Discovery in Databases)**.

---

### Descrição do Projeto

O dataset **LungCanC2024** contém **289.010 registros**, permitindo pesquisas avançadas em medicina de precisão, diagnóstico assistido por computador e análise de sobrevivência.

---

### Estrutura do Dataset

O conjunto de dados está organizado em quatro pilares fundamentais para garantir uma visão holística do paciente:

#### 1. Características Radiômicas (Imagem)
Métricas extraídas de Tomografias Computadorizadas (CT) e PET Scans:
* Diâmetro dos nódulos.
* Densidade tecidual.
* Textura e heterogeneidade do tumor.
* Atividade metabólica do tumor.

#### 2. Metadados Clínicos
* **Demografia**: Idade (30-90 anos) e Gênero.
* **Histórico**: Tabagismo e histórico familiar.
* **Tratamentos**: Registros de Radioterapia, Quimioterapia, Imunoterapia e Terapia Alvo.

#### 3. Biomarcadores Genômicos
* Status de mutação: **EGFR**, **KRAS** e fusão **ALK**.
* Expressão de **PD-L1** e Carga Mutacional do Tumor (**TMB**).

#### 4. Variáveis Alvo (Multi-Label)
* `cancer_presence`: Diagnóstico binário (Maligno/Benigno).
* `cancer_subtype`: Subtipagem da patologia (Adenocarcinoma, Squamous Cell, SCLC, Other).

---

### Metodologia e Algoritmos

O pipeline foi estruturado em duas frentes principais:

#### Aprendizado Supervisionado (Classificação)
Implementação de uma **Rede Neural MLP (Multi-Layer Perceptron)** via Scikit-Learn para predição multirrótulo, focada na acurácia diagnóstica.

#### Aprendizado Não Supervisionado (Clustering)
A abordagem não supervisionada combina algoritmos complementares baseados em particionamento e densidade:

* **Algoritmo K-Means:**
  * Agrupamento dos dados em um número pré-definido de clusters.
  * Identificação dos centroides de cada grupo, com representação média das características radiômicas e genômicas de cada subtipo.
  * Utilização de critérios de distância integrados ao pipeline.

* **Algoritmo DBSCAN:**
  * Identificação automática de outliers e perfis clínicos raros (ruído).
  * Verificação de como as variáveis radiômicas e genômicas se agrupam naturalmente sem a influência dos rótulos pré-existentes.
  * Diferenciação de subtipos tumorais baseada na densidade de suas características biológicas.

---

### Fluxo KDD (Processo de Descoberta)

1. **Seleção de Dados:** Carga e filtragem ativa de datasets clínicos contendo histórico do paciente, exames físicos, parâmetros radiômicos e biomarcadores genômicos.
2. **Pré-processamento:** Tratamento automatizado de dados ausentes (*imputation*), remoção de inconsistências clínicas e isolamento de outliers biológicos.
3. **Transformação:** Mapeamento de categorias via *One-Hot Encoding* para dados nominais, codificação binária estrita (Gênero/Histórico Familiar) e padronização contínua por escala **Z-score**. Balanceamento de classes via **SMOTE** para o Modelo 2.
4. **Data Mining (Mineração de Dados):**
   * **Modelo 1 (Triagem Binária):** Classificação estatística da presença ou ausência de neoplasia maligna.
   * **Modelo 2 (Subtipagem Multiclasse):** Identificação histológica precisa entre *Adenocarcinoma*, *Células Escamosas*, *Pequenas Células (SCLC)* e *Outros*.
5. **Avaliação e Interpretação:** Geração de matriz de métricas consolidadas, matrizes de confusão e índices de validação de densidade (Silhueta e Davies-Bouldin).

---

### EDA — Análise Exploratória dos Dados

A análise exploratória dos dados responde perguntas que ajudam a compreender:

1. Perfil dos pacientes;
2. Distribuição da doença;
3. Fatores associados ao câncer;
4. Comportamento dos atributos clínicos;
5. Qualidade dos dados;
6. Hipóteses para a etapa de mineração.

---

### Estrutura do Projeto

```
atv-dashboard/
├── Home.py                          # Página inicial do dashboard
├── pages/
│   ├── 1_Análise_Exploratória_de_Dados.py   # EDA interativa
│   └── 2_Processo_KDD.py                    # Pipeline KDD completo
├── secoes_eda/                      # Módulos das seções da EDA
│   ├── secao_geral.py
│   ├── secao_pp1.py
│   └── secao_pp2.py
├── etapas_kdd/                      # Módulos das etapas do KDD
│   ├── KDD_mapping_features.py
│   ├── KDD_pre_processing.py
│   ├── KDD_data_transformation.py
│   ├── KDD_data_mining.py
│   └── KDD_result_metrics.py
├── data/
│   ├── train.csv
│   └── test.csv
└── requirements.txt
```

---

### Tecnologias Utilizadas

* **Python 3.10+**
* **Streamlit** — Interface do Dashboard
* **Scikit-Learn** — Algoritmos de ML (MLP, K-Means, DBSCAN) e Pré-processamento
* **Imbalanced-Learn** — Balanceamento de classes (SMOTE)
* **Plotly** — Visualização avançada de dados
* **Pandas / NumPy** — Manipulação e análise de dados

---

### Instalação

1. Instalar as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Executar o app:

   ```bash
   streamlit run Home.py
   ```
