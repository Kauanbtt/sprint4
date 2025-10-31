# Detecção de Pirataria de Produtos HP com Machine Learning

![Status](https://img.shields.io/badge/status-concluído-green)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Libraries](https://img.shields.io/badge/libraries-pandas%20%7C%20scikit--learn-orange)

## 1. Resumo do Projeto

Este projeto apresenta uma solução de Machine Learning desenvolvida para identificar anúncios de produtos HP (como cartuchos e toners) com alta probabilidade de serem falsificados ou piratas em plataformas de e-commerce. Utilizando um dataset enriquecido com informações de anúncios reais, foram treinados e avaliados múltiplos modelos de classificação, com o modelo **Gradient Boosting** alcançando uma **acurácia de 94%** e um excelente equilíbrio entre precisão e recall.

O principal desafio e aprendizado do projeto foi a identificação e correção de um caso de **Data Leakage**, que inicialmente gerou modelos com performance irrealista de 100%, transformando o projeto em um estudo de caso prático sobre a construção de modelos robustos e confiáveis.

## 2. O Problema de Negócio

A venda de produtos falsificados representa uma ameaça significativa para a HP, impactando negativamente em:
* **Receita:** Perda de vendas para concorrentes desleais.
* **Reputação da Marca:** Clientes insatisfeitos com produtos de baixa qualidade que podem danificar seus equipamentos.
* **Segurança do Cliente:** Produtos que não passam por controle de qualidade podem apresentar riscos.

Uma solução automatizada para sinalizar anúncios suspeitos permite que a equipe de proteção da marca atue de forma mais rápida e eficiente, otimizando recursos e protegendo o ecossistema da empresa.

## 3. Metodologia

O projeto seguiu um pipeline clássico de ciência de dados, desde a preparação dos dados até a seleção do modelo final.

### 3.1. Fonte de Dados

O trabalho foi realizado com base no arquivo `dados_enriquecidos_com_alertas.csv`, contendo 1268 registros e 22 features, incluindo:
* **Informações do Anúncio:** `titulo`, `preco`, `quantidade_vendida`.
* **Informações do Vendedor:** `nome_vendedor`, `status_vendedor`, `reputacao_cor`.
* **Métricas de Avaliação:** `avaliacao_nota`, `avaliacao_numero` e percentuais de reviews por estrelas.
* **Dados Técnicos do Produto:** `categoria_produto`, `modelo_cartucho`, `rendimento_paginas`, `custo_por_pagina`.
* **Variável Alvo:** `alerta_suspeita` (True/False), indicando se o produto já havia sido sinalizado como suspeito.

### 3.2. Pré-processamento e Feature Engineering

* **Limpeza de Dados:** Conversão de colunas como `capacidade` (ex: "200g") para formato numérico.
* **Seleção de Features:** Foram selecionadas as features mais relevantes para o problema, descartando identificadores e URLs.
* **Pipeline de Transformação:** Utilizou-se o `ColumnTransformer` do Scikit-learn para criar um pipeline robusto que aplica:
    * `SimpleImputer` para tratar valores ausentes.
    * `StandardScaler` para normalizar as features numéricas.
    * `OneHotEncoder` para transformar as features categóricas em formato numérico.

### 3.3. A Descoberta de Data Leakage: Uma Lição Crucial

Na primeira versão do modelo, todas as features foram utilizadas, resultando em uma **acurácia perfeita de 100%** em todos os algoritmos. Este resultado, embora parecesse ideal, foi um forte indicativo de **vazamento de dados (Data Leakage)**.

> A investigação revelou que a feature `motivo_suspeita` era a causa. Esta coluna só continha um valor quando a variável alvo `alerta_suspeita` era `True`. O modelo aprendeu uma regra simples e "trapaceira", em vez de identificar os padrões nos dados.

**Solução:** A feature `motivo_suspeita` foi removida do conjunto de treino, forçando os modelos a aprenderem a partir de indicadores legítimos como preço, avaliações e dados do vendedor, resultando em métricas realistas e um modelo verdadeiramente preditivo.

### 3.4. Modelagem e Avaliação

Foram treinados e avaliados três algoritmos de classificação distintos:
1.  **Regressão Logística:** Como um modelo de baseline.
2.  **Random Forest:** Um modelo robusto baseado em árvores de decisão.
3.  **Gradient Boosting:** Um modelo avançado conhecido por sua alta performance.

A avaliação foi focada no trade-off entre **Precisão** (evitar falsos positivos) e **Recall** (encontrar o máximo de suspeitos possível), métricas cruciais para o problema de negócio.

## 4. Resultados

Após a correção do data leakage, os modelos apresentaram performances realistas e de alto nível. O **Gradient Boosting** foi o grande destaque.

| Modelo                  | Acurácia | Precisão (Suspeito) | Recall (Suspeito) | F1-Score (Suspeito) |
| ----------------------- | -------- | ------------------- | ----------------- | ------------------- |
| Regressão Logística     | 83.6%    | 0.61                | 0.88              | 0.72                |
| Random Forest (Padrão)  | 93.4%    | 0.88                | 0.84              | 0.86                |
| **Gradient Boosting** | **94.0%**| **0.88** | **0.87** | **0.87** |

O modelo **Gradient Boosting** foi selecionado como a solução final por apresentar o melhor equilíbrio entre todas as métricas, maximizando tanto a capacidade de encontrar produtos suspeitos (`Recall`) quanto a confiabilidade de suas classificações (`Precisão`).

## 5. Como Executar o Projeto

### 5.1. Pré-requisitos
* Python 3.9 ou superior
* Git

### 5.2. Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [[https://github.com/seu-usuario/nome-do-repositorio.git](https://github.com/lancelot-rodrigues/FrontEnd_Sprint3)]
    cd nome-do-repositorio
    ```

2.  **Crie e ative um ambiente virtual (Recomendado):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install pandas numpy scikit-learn
    ```

### 5.3. Execução
Para executar o pipeline de treinamento e avaliação, basta rodar o script principal:
```bash
python app.py
```
O script irá carregar o dataset, treinar os modelos e imprimir os relatórios de classificação no terminal.

## 6. Conclusão e Próximos Passos

O projeto demonstrou com sucesso a viabilidade de utilizar Machine Learning para automatizar a detecção de produtos HP suspeitos com alta eficácia. O modelo final, baseado em Gradient Boosting, representa uma ferramenta valiosa para otimizar os esforços de proteção da marca.

Possíveis melhorias futuras incluem:
* **Análise de Erros:** Investigar os poucos casos em que o modelo falha para identificar padrões e criar novas features.
* **Ajuste de Threshold de Decisão:** Utilizar a probabilidade (`predict_proba`) para ajustar o limiar de classificação, otimizando ainda mais o balanço entre precisão e recall.
* **Expansão do Dataset:** Incorporar mais dados e, se possível, informações textuais dos anúncios para enriquecer o modelo.
