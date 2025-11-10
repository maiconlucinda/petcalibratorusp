# 🌡️ PET Thermal Comfort Calibrator

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

Sistema de calibração de conforto térmico baseado no índice PET (Physiological Equivalent Temperature) usando modelagem ordinal estatística.

**Autora**: Caroline Freire do Santos  
**Instituição**: Universidade de São Paulo (USP)  
**Programa**: Doutorado em Climatologia

---

## 📋 Índice

- [Introdução](#-introdução)
- [Metodologia](#-metodologia)
- [Fundamentos Matemáticos](#-fundamentos-matemáticos)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Estrutura de Dados](#-estrutura-de-dados)
- [Interpretação dos Resultados](#-interpretação-dos-resultados)
- [Exemplos de Visualizações](#-exemplos-de-visualizações)
- [Referências](#-referências)

---

## 🎯 Introdução

O **PET Thermal Comfort Calibrator** é uma ferramenta científica desenvolvida para calibrar faixas locais de conforto térmico usando o índice PET (Physiological Equivalent Temperature) e dados de sensação térmica subjetiva coletados através de questionários.

### Problema que Resolve

Índices de conforto térmico como o PET são calculados a partir de variáveis meteorológicas e fisiológicas, mas as faixas de conforto associadas a esses índices variam significativamente entre diferentes contextos climáticos e culturais. Este sistema permite:

1. **Calibração Local**: Determinar faixas de PET específicas para a população e clima estudados
2. **Análise Estatística Rigorosa**: Usar modelagem ordinal apropriada para dados de sensação térmica
3. **Métricas Objetivas**: Calcular PET neutro e faixas de conforto com intervalos de confiança
4. **Visualizações Publicáveis**: Gerar gráficos de alta qualidade para artigos científicos

### Características Principais

- ✅ Processamento de dados CSV/Excel com mapeamento flexível de colunas
- ✅ Modelagem de regressão logística ordinal proporcional
- ✅ Cálculo de PET neutro com intervalos de confiança (95%)
- ✅ Determinação de faixas de conforto (80% e 90%)
- ✅ **Faixas de PET por categoria de sensação** (novo!)
- ✅ Análise opcional de aceitabilidade térmica
- ✅ Geração automática de visualizações (300 DPI)
- ✅ Relatório completo em Markdown

---

## 🔬 Metodologia

### Regressão Logística Ordinal Proporcional

O sistema utiliza o **modelo de chances proporcionais** (proportional odds model) para relacionar o PET com a sensação térmica ordinal. Este é o método estatístico apropriado para variáveis resposta ordinais (como sensações térmicas em escala).

#### Por que Modelagem Ordinal?

Dados de sensação térmica têm uma estrutura ordinal natural:

```
muito frio < frio < frio moderado < confortável < calor moderado < quente < muito quente
```

Métodos inadequados (como regressão linear simples) ignoram essa estrutura ordinal e tratam as categorias como se fossem numéricas arbitrárias. A modelagem ordinal:

- **Preserva a ordem** das categorias
- **Não assume intervalos iguais** entre categorias
- **Modela probabilidades** de cada categoria de forma apropriada
- **É mais eficiente estatisticamente** que modelos multinomiais

### O Modelo

Para uma sensação térmica ordinal Y ∈ {-3, -2, -1, 0, +1, +2, +3} e PET como preditor:

```
logit(P(Y ≤ k | PET)) = τₖ - β × PET
```

Onde:
- **τₖ** são os limiares (cutpoints) entre categorias adjacentes
- **β** é o coeficiente que relaciona PET à sensação térmica
- O sinal negativo garante que PET maior → sensação mais quente

#### Interpretação dos Parâmetros

- **β > 0**: Aumento no PET aumenta a probabilidade de sensações mais quentes
- **τₖ**: Valores de PET onde ocorrem transições entre categorias (quando β = 1)

---

## 📐 Fundamentos Matemáticos

### 1. PET Neutro

O **PET neutro** é o valor de PET onde a probabilidade da categoria "confortável" (0) é maximizada. No modelo de chances proporcionais, isso ocorre no limiar entre "confortável" e "calor moderado":

```
PET_neutro = τ₀ / β
```

Onde:
- **τ₀** é o cutpoint entre categoria 0 e +1
- **β** é o coeficiente do modelo

#### Intervalo de Confiança

Usamos o **método delta** para propagar a incerteza:

```
Var(PET_neutro) ≈ (1/β)² × Var(τ₀) + (τ₀/β²)² × Var(β)
```

O IC 95% é então: `PET_neutro ± 1.96 × √Var(PET_neutro)`

### 2. Faixas de Conforto

As faixas de conforto são determinadas pela probabilidade combinada das três categorias centrais: **frio moderado (-1)**, **confortável (0)**, e **calor moderado (+1)**.

#### Probabilidade de Conforto

```
P_conf(PET) = P(Y ≤ +1 | PET) - P(Y ≤ -2 | PET)
            = P(-1 ≤ Y ≤ +1 | PET)
```

Usando o modelo:

```
P(Y ≤ k | PET) = expit(τₖ - β × PET)
```

Onde `expit(x) = 1 / (1 + exp(-x))` é a função logística.

#### Determinação das Faixas

Para um limiar de probabilidade p (ex: 0.80 ou 0.90):

1. Calcular P_conf(PET) para uma grade fina de valores de PET (-5°C a 55°C, passo 0.05°C)
2. Identificar o intervalo onde P_conf(PET) ≥ p
3. **L_p** = menor PET onde P_conf ≥ p (limite inferior)
4. **U_p** = maior PET onde P_conf ≥ p (limite superior)

**Faixa 80%**: [L₈₀, U₈₀] - 80% dos respondentes se sentem confortáveis  
**Faixa 90%**: [L₉₀, U₉₀] - 90% dos respondentes se sentem confortáveis

---

## 🚀 Instalação

### Requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo a Passo

1. **Clone ou baixe este repositório**:
```bash
git clone https://github.com/seu-usuario/pet-thermal-comfort-calibrator.git
cd pet-thermal-comfort-calibrator
```

2. **Crie um ambiente virtual (recomendado)**:
```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
```

3. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

### Dependências

O sistema requer as seguintes bibliotecas Python:

- **pandas** (≥1.5.0): Manipulação de dados
- **numpy** (≥1.23.0): Operações numéricas
- **matplotlib** (≥3.6.0): Visualizações
- **statsmodels** (≥0.13.0): Modelagem ordinal
- **scipy** (incluído com numpy): Funções estatísticas
- **openpyxl** (≥3.0.0): Suporte a arquivos Excel

---

## 💻 Uso

### Sintaxe Básica

```bash
python pet_calibrator.py --input <arquivo_entrada> --out <diretorio_saida> [--map <mapeamento.json>]
```

### Argumentos

| Argumento | Obrigatório | Descrição |
|-----------|-------------|-----------|
| `--input` | ✅ Sim | Caminho para arquivo CSV ou Excel com dados |
| `--out` | ✅ Sim | Diretório onde os resultados serão salvos |
| `--map` | ❌ Não | Arquivo JSON com mapeamento de colunas |
| `--pdf` | ❌ Não | Gera relatório em PDF (requer pandoc instalado) |
| `--verbose` | ❌ Não | Ativa logging detalhado |

### Exemplos de Uso

#### 1. Uso Básico com CSV

```bash
python pet_calibrator.py --input dados_questionario.csv --out resultados
```

#### 2. Com Arquivo Excel e Mapeamento de Colunas

```bash
python pet_calibrator.py --input dados.xlsx --out resultados --map mapeamento.json
```

#### 3. Com Relatório em PDF

```bash
python pet_calibrator.py --input dados.csv --out resultados --pdf
```

**Nota**: Requer `pandoc` instalado. Para instalar:
- **macOS**: `brew install pandoc`
- **Ubuntu/Debian**: `sudo apt-get install pandoc`
- **Windows**: [https://pandoc.org/installing.html](https://pandoc.org/installing.html)

#### 4. Com Logging Detalhado

```bash
python pet_calibrator.py --input dados.csv --out resultados --verbose
```

### Arquivos Gerados

Após a execução, o diretório de saída conterá:

```
resultados/
├── respostas_com_PET.csv          # Dados limpos com TSV ordinal
├── scatter_TSV_PET.png            # Gráfico de dispersão
├── probs_ordinais_PET.png         # Curvas de probabilidade
├── zona_conforto_logit.png        # Gráfico de zona de conforto
├── RELATORIO_PET.md               # Relatório completo em Markdown
├── RELATORIO_PET.pdf              # Relatório em PDF (se --pdf usado)
└── pet_calibrator.log             # Log de execução
```

---

## 🎓 Exemplo Prático

O repositório inclui dados de exemplo para você testar o sistema imediatamente.

### Executando o Exemplo

1. **Certifique-se de que as dependências estão instaladas**:
```bash
pip install -r requirements.txt
```

2. **Execute o calibrador com os dados de exemplo**:
```bash
python pet_calibrator.py --input examples/sample_data.csv --out examples/output --map examples/column_mapping.json
```

3. **Verifique os resultados**:
```bash
ls examples/output/
```

### Dados de Exemplo

O arquivo `examples/sample_data.csv` contém **150 respostas sintéticas** com:
- Valores de PET realistas (5.2°C a 41.9°C)
- Todas as 7 categorias de sensação térmica
- Coluna opcional de aceitabilidade
- Timestamps para contexto

**Distribuição das sensações**:
- Muito Frio: 11 respostas (7.3%)
- Frio: 16 respostas (10.7%)
- Frio Moderado: 27 respostas (18.0%)
- **Confortável: 42 respostas (28.0%)**
- Calor Moderado: 27 respostas (18.0%)
- Quente: 16 respostas (10.7%)
- Muito Quente: 11 respostas (7.3%)

### Resultados Esperados

Após executar o exemplo, você encontrará em `examples/output/`:

| Arquivo | Descrição |
|---------|-----------|
| `respostas_com_PET.csv` | Dados limpos com coluna TSV_ordinal adicionada |
| `RELATORIO_PET.md` | Relatório completo com todos os resultados |
| `scatter_TSV_PET.png` | Gráfico de dispersão PET vs Sensação |
| `probs_ordinais_PET.png` | Curvas de probabilidade por categoria |
| `zona_conforto_logit.png` | Gráfico da zona de conforto |
| `pet_calibrator.log` | Log detalhado da execução |

**Métricas do exemplo**:
- PET Neutro: ~1.1°C (dados sintéticos para demonstração)
- 150 observações válidas (100% dos dados)
- Modelo converge com sucesso
- Todas as visualizações geradas

**Exemplo de Faixas por Categoria** (do relatório gerado):

| Sensação | PET Observado Médio | Intervalo Observado | N |
|----------|---------------------|---------------------|---|
| Muito Frio (-3) | 9.9°C | [5.2, 13.9]°C | 11 |
| Frio (-2) | 14.9°C | [13.0, 17.2]°C | 16 |
| Frio Moderado (-1) | 19.7°C | [18.5, 21.5]°C | 27 |
| **Confortável (0)** | **23.9°C** | **[21.1, 27.7]°C** | **42** |
| Calor Moderado (+1) | 28.0°C | [25.7, 30.8]°C | 27 |
| Quente (+2) | 33.2°C | [29.0, 37.7]°C | 16 |
| Muito Quente (+3) | 38.9°C | [35.0, 41.9]°C | 11 |

Esta tabela mostra claramente as faixas de PET associadas a cada sensação térmica!

### Usando Seus Próprios Dados

Para analisar seus dados:

1. **Prepare seu arquivo CSV/Excel** com as colunas obrigatórias:
   - Valores de PET pré-calculados
   - Sensação térmica (uma das 7 categorias)

2. **Crie um arquivo de mapeamento** (se necessário):
```json
{
  "PET_C": "nome_da_sua_coluna_pet",
  "Sensation": "nome_da_sua_coluna_sensacao"
}
```

3. **Execute o calibrador**:
```bash
python pet_calibrator.py --input seus_dados.csv --out resultados --map seu_mapeamento.json
```

---

## 📊 Estrutura de Dados

### Colunas Obrigatórias

O arquivo de entrada deve conter **apenas 2 colunas obrigatórias**:

| Coluna | Tipo | Exemplo | Descrição |
|--------|------|---------|-----------|
| **PET_C** | Numérico | 23.5, 28.2, 32.1 | Valores de PET pré-calculados em °C |
| **Sensation** | Texto | "confortável", "quente" | Sensação térmica do respondente |

### Categorias de Sensação Térmica

A coluna de sensação deve conter **exatamente uma** destas 7 categorias:

| Categoria | Valor Ordinal | Descrição |
|-----------|---------------|-----------|
| muito frio | -3 | Muito desconfortável pelo frio |
| frio | -2 | Desconfortável pelo frio |
| frio moderado | -1 | Levemente frio |
| **confortável** | **0** | **Zona de conforto** |
| calor moderado | +1 | Levemente quente |
| quente | +2 | Desconfortável pelo calor |
| muito quente | +3 | Muito desconfortável pelo calor |

**Nota**: O sistema normaliza automaticamente (remove acentos, converte para minúsculas), então variações como "Confortável", "confortavel", "CONFORTÁVEL" são todas aceitas.

### Colunas Opcionais

| Coluna | Tipo | Uso |
|--------|------|-----|
| Acceptability | Texto | Análise comparativa de aceitabilidade |
| Timestamp | Datetime | Informação contextual |
| Outras | Variado | Ignoradas (podem ser usadas em extensões futuras) |

### Exemplo de Arquivo CSV

```csv
Timestamp,PET_C,Sensation,Acceptability
2024-01-15 14:30,23.5,confortável,aceitável
2024-01-15 14:35,28.2,calor moderado,aceitável
2024-01-15 14:40,32.1,quente,inaceitável
2024-01-15 14:45,19.8,frio moderado,aceitável
```

### Mapeamento de Colunas

Se suas colunas têm nomes diferentes, crie um arquivo JSON de mapeamento:

```json
{
  "PET_C": "Temperatura_Equivalente_PET",
  "Sensation": "Como você está se sentindo agora?",
  "Acceptability": "Este ambiente é aceitável?"
}
```

Use com: `--map mapeamento.json`

---

## 📖 Interpretação dos Resultados

### PET Neutro

O **PET neutro** representa a temperatura equivalente ideal para conforto térmico na população estudada.

**Exemplo**: PET neutro = 24.5°C (IC 95%: 23.8 - 25.2°C)

**Interpretação**: 
- A sensação "confortável" é mais provável em torno de 24.5°C
- Com 95% de confiança, o verdadeiro PET neutro está entre 23.8°C e 25.2°C
- Valores fora deste intervalo tendem a sensações de frio ou calor

### Faixas de Conforto

#### Faixa 80%

**Exemplo**: 22.3°C - 26.8°C (amplitude: 4.5°C)

**Interpretação**:
- Dentro desta faixa, **80% ou mais** dos respondentes se sentem confortáveis (categorias -1, 0, +1)
- Faixa mais ampla, adequada para design urbano e planejamento
- Aceita maior variabilidade térmica

#### Faixa 90%

**Exemplo**: 23.1°C - 25.9°C (amplitude: 2.8°C)

**Interpretação**:
- Dentro desta faixa, **90% ou mais** dos respondentes se sentem confortáveis
- Faixa mais restrita, ideal para ambientes controlados
- Maior garantia de conforto, mas menos flexível

### Faixas de PET por Categoria de Sensação

O sistema também calcula **faixas de PET características para cada categoria de sensação térmica**, permitindo entender em quais temperaturas cada sensação é mais provável.

#### O que são as Faixas por Categoria?

Para cada sensação (Muito Frio, Frio, Frio Moderado, Confortável, Calor Moderado, Quente, Muito Quente), o sistema determina:

1. **Faixa Modal**: Intervalo de PET onde aquela sensação é a **mais provável** (comparada às outras)
2. **Faixa de Probabilidade ≥30%**: Intervalo onde a probabilidade daquela sensação é ≥30%
3. **Dados Observados**: Estatísticas descritivas do PET quando as pessoas reportaram aquela sensação

#### Exemplo de Interpretação

**Categoria "Calor Moderado" (+1)**:
- **Faixa Modal**: [25.0, 30.0]°C
- **Pico**: 27.5°C (probabilidade máxima)
- **Dados Observados**: Média = 28.0°C, Mediana = 28.1°C, Intervalo = [25.7, 30.8]°C

**Interpretação**:
- Entre 25-30°C, "Calor Moderado" é a sensação mais provável
- O pico de probabilidade ocorre em 27.5°C
- Na prática, quando as pessoas reportaram "Calor Moderado", o PET médio foi 28.0°C
- Esta informação é útil para calibração local e design de espaços urbanos

#### Aplicações Práticas

1. **Design Urbano**: Saber que "Confortável" ocorre tipicamente entre 22-27°C ajuda no planejamento de áreas de sombra
2. **Alertas de Calor**: Identificar em qual PET as pessoas começam a sentir "Quente" ou "Muito Quente"
3. **Calibração Local**: Comparar suas faixas com literatura internacional para identificar adaptação climática
4. **Validação**: Verificar se as faixas observadas correspondem às faixas modais (coerência do modelo)

### Comparação com Literatura

Compare seus resultados com faixas de PET publicadas para diferentes climas:

- **Clima Temperado** (Europa Central): PET neutro ≈ 18-20°C
- **Clima Tropical** (Brasil): PET neutro ≈ 24-28°C
- **Clima Quente-Seco** (Oriente Médio): PET neutro ≈ 26-30°C

Diferenças indicam **adaptação climática e cultural** da população estudada.

**Dica**: Use a seção "Faixas de PET por Categoria" do relatório para comparar suas faixas locais com valores de referência da literatura.

---

## 📈 Exemplos de Visualizações

### 1. Gráfico de Dispersão (scatter_TSV_PET.png)

Mostra a relação entre PET e sensação térmica ordinal:
- Cada ponto representa uma resposta
- Cores indicam a categoria de sensação
- Linha vertical marca o PET neutro
- Jitter no eixo Y evita sobreposição de pontos

### 2. Curvas de Probabilidade (probs_ordinais_PET.png)

Mostra P(Y = k | PET) para cada categoria:
- 7 curvas coloridas (uma por categoria)
- Linha vertical no PET neutro
- Regiões sombreadas indicam faixas 80% e 90%
- Permite visualizar transições entre categorias

### 3. Zona de Conforto (zona_conforto_logit.png)

Mostra a probabilidade combinada de conforto P_conf(PET):
- Curva de probabilidade vs PET
- Linhas horizontais em 0.80 e 0.90
- Linhas verticais marcam limites das faixas
- Região sombreada destaca faixa 80%

---

## 📚 Referências

### Índice PET

1. **Höppe, P.** (1999). The physiological equivalent temperature – a universal index for the biometeorological assessment of the thermal environment. *International Journal of Biometeorology*, 43(2), 71-75.

2. **Matzarakis, A., Mayer, H., & Iziomon, M. G.** (1999). Applications of a universal thermal index: physiological equivalent temperature. *International Journal of Biometeorology*, 43(2), 76-84.

### Modelagem Ordinal

3. **McCullagh, P.** (1980). Regression models for ordinal data. *Journal of the Royal Statistical Society: Series B (Methodological)*, 42(2), 109-127.

4. **Agresti, A.** (2010). *Analysis of Ordinal Categorical Data* (2nd ed.). John Wiley & Sons.

5. **Christensen, R. H. B.** (2019). ordinal—Regression Models for Ordinal Data. R package version 2019.12-10.

### Conforto Térmico

6. **ASHRAE Standard 55** (2020). Thermal Environmental Conditions for Human Occupancy. American Society of Heating, Refrigerating and Air-Conditioning Engineers.

7. **ISO 7730** (2005). Ergonomics of the thermal environment — Analytical determination and interpretation of thermal comfort using calculation of the PMV and PPD indices and local thermal comfort criteria.

8. **Nikolopoulou, M., & Steemers, K.** (2003). Thermal comfort and psychological adaptation as a guide for designing urban spaces. *Energy and Buildings*, 35(1), 95-101.

### Calibração Local

9. **Lin, T. P.** (2009). Thermal perception, adaptation and attendance in a public square in hot and humid regions. *Building and Environment*, 44(10), 2017-2026.

10. **Lai, D., Guo, D., Hou, Y., Lin, C., & Chen, Q.** (2014). Studies of outdoor thermal comfort in northern China. *Building and Environment*, 77, 110-118.

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autora

**Caroline Freire do Santos**  
Doutoranda em Climatologia  
Universidade de São Paulo (USP)  
São Paulo, Brasil

---

## 🤝 Contribuições

Contribuições, issues e sugestões são bem-vindas! Sinta-se à vontade para abrir uma issue ou pull request.

---

## 📧 Contato

Para questões acadêmicas ou colaborações, entre em contato através da USP.

---

**Desenvolvido com 🌡️ para pesquisa em conforto térmico e climatologia urbana**
