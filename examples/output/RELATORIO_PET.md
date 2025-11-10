# Relatório de Calibração PET - Conforto Térmico

**Autora**: Carol Freire do Santos  
**Instituição**: Universidade de São Paulo (USP)  
**Programa**: Doutorado em Climatologia  
**Data de Geração**: 2025-11-10 11:47:48

---

## 1. Resumo dos Dados

- **Total de respostas**: 150
- **Respostas válidas**: 150 (100.0%)
- **PET médio**: 24.0°C (DP: 7.6°C)
- **PET mediana**: 24.1°C
- **Intervalo PET**: [5.2, 41.9]°C

### Distribuição de Sensação Térmica

| Categoria | Valor Ordinal | N Respostas | % |
|-----------|---------------|-------------|---|
| Muito Frio | -3 | 11 | 7.3% |
| Frio | -2 | 16 | 10.7% |
| Frio Moderado | -1 | 27 | 18.0% |
| Confortável | +0 | 42 | 28.0% |
| Calor Moderado | +1 | 27 | 18.0% |
| Quente | +2 | 16 | 10.7% |
| Muito Quente | +3 | 11 | 7.3% |


## 2. Modelo Ordinal

### Parâmetros Estimados

**Coeficiente β (PET_C)**:

- Estimativa: **2.1641**
- Erro padrão: 0.3190
- IC 95%: [1.5388, 2.7893]

**Limiares (Cutpoints) τ_k**:

| Limiar | Estimativa | Erro Padrão | IC 95% |
|--------|------------|-------------|---------|
| τ_-3 | 2.1482 | 0.1854 | [1.7849, 2.5115] |
| τ_-2 | 2.2938 | 0.2077 | [1.8867, 2.7009] |
| τ_-1 | 27.7411 | 4.3127 | [19.2884, 36.1938] |
| τ_+0 | 2.3433 | 0.1651 | [2.0196, 2.6670] |
| τ_+1 | 2.1940 | 0.1790 | [1.8433, 2.5448] |
| τ_+2 | 2.6208 | 0.1936 | [2.2413, 3.0002] |

### Qualidade do Ajuste

- **Convergência**: ✓ Sim
- **N observações**: 150
- **Log-verossimilhança**: -37.14
- **AIC**: 88.28
- **BIC**: 109.35

### Interpretação do Modelo

O modelo de regressão logística ordinal proporcional relaciona o PET 
com a sensação térmica usando a função de ligação logit:

```
logit(P(Y ≤ k | PET)) = τ_k - β × PET
```

O coeficiente β = 2.1641 indica que cada aumento de 1°C no PET 
está associado a um aumento na probabilidade de sensações mais quentes.


## 3. PET Neutro

**PET Neutro = 1.1°C**

- **Intervalo de Confiança 95%**: [0.7, 1.4]°C
- **Erro Padrão**: 0.18°C

O PET neutro representa a temperatura equivalente onde a sensação 
"confortável" (categoria 0) é mais provável. Este valor é calculado 
como o ponto médio entre as categorias de conforto no modelo ordinal 
(τ₀ / β).

## 4. Faixas de Conforto

As faixas de conforto representam os intervalos de PET onde a probabilidade 
combinada das categorias centrais (-1: Frio Moderado, 0: Confortável, 
+1: Calor Moderado) atinge os limiares especificados.

### Faixa de Conforto 80%

⚠️ Não foi possível determinar a faixa de 80% com os dados disponíveis.

### Faixa de Conforto 90%

⚠️ Não foi possível determinar a faixa de 90% com os dados disponíveis.

## 5. Faixas de PET por Categoria de Sensação

Esta seção apresenta as faixas de PET características para cada categoria 
de sensação térmica, baseadas no modelo probabilístico calibrado.


Para cada categoria, são apresentadas:

- **Faixa Modal**: Intervalo de PET onde esta sensação é a mais provável

- **Faixa de Probabilidade ≥30%**: Intervalo onde a probabilidade desta sensação é ≥30%

- **Dados Observados**: Estatísticas descritivas do PET quando esta sensação foi reportada


### Resumo das Faixas de PET


| Sensação | Faixa Modal (°C) | Pico PET (°C) | PET Observado Médio (°C) | N Obs. |

|----------|------------------|---------------|--------------------------|--------|

| Muito Frio (-3) | [-5.0, 1.0] | -5.0 | 9.9 | 11 |

| Frio (-2) | — | — | 14.9 | 16 |

| Frio Moderado (-1) | [1.0, 6.7] | 6.7 | 19.7 | 27 |

| Confortável (+0) | — | — | 23.9 | 42 |

| Calor Moderado (+1) | — | — | 28.0 | 27 |

| Quente (+2) | — | — | 33.2 | 16 |

| Muito Quente (+3) | [6.7, 55.0] | 18.5 | 38.9 | 11 |



### Detalhamento por Categoria


#### Muito Frio (-3)


**Faixa Modal**: [-5.0, 1.0]°C  

- Pico de probabilidade em -5.0°C 
(P = 100.0%)  

- Amplitude: 6.0°C


**Faixa com P ≥ 30%**: [-5.0, 1.3]°C  

- Amplitude: 6.3°C


**Dados Observados** (n = 11):  

- Média: 9.9°C (DP: 2.3°C)  

- Mediana: 9.7°C  

- Intervalo: [5.2, 13.9]°C  

- Percentis 25-75: [9.1, 11.3]°C


#### Frio (-2)


**Faixa Modal**: Não identificada (sensação nunca é a mais provável)


**Faixa com P ≥ 30%**: Não identificada


**Dados Observados** (n = 16):  

- Média: 14.9°C (DP: 1.1°C)  

- Mediana: 14.9°C  

- Intervalo: [13.0, 17.2]°C  

- Percentis 25-75: [14.2, 15.4]°C


#### Frio Moderado (-1)


**Faixa Modal**: [1.0, 6.7]°C  

- Pico de probabilidade em 6.7°C 
(P = 100.0%)  

- Amplitude: 5.6°C


**Faixa com P ≥ 30%**: [0.7, 13.2]°C  

- Amplitude: 12.5°C


**Dados Observados** (n = 27):  

- Média: 19.7°C (DP: 0.9°C)  

- Mediana: 19.5°C  

- Intervalo: [18.5, 21.5]°C  

- Percentis 25-75: [18.9, 20.4]°C


#### Confortável (+0)


**Faixa Modal**: Não identificada (sensação nunca é a mais provável)


**Faixa com P ≥ 30%**: Não identificada


**Dados Observados** (n = 42):  

- Média: 23.9°C (DP: 1.4°C)  

- Mediana: 24.1°C  

- Intervalo: [21.1, 27.7]°C  

- Percentis 25-75: [22.9, 24.5]°C


#### Calor Moderado (+1)


**Faixa Modal**: Não identificada (sensação nunca é a mais provável)


**Faixa com P ≥ 30%**: Não identificada


**Dados Observados** (n = 27):  

- Média: 28.0°C (DP: 1.2°C)  

- Mediana: 28.1°C  

- Intervalo: [25.7, 30.8]°C  

- Percentis 25-75: [26.8, 28.7]°C


#### Quente (+2)


**Faixa Modal**: Não identificada (sensação nunca é a mais provável)


**Faixa com P ≥ 30%**: Não identificada


**Dados Observados** (n = 16):  

- Média: 33.2°C (DP: 2.2°C)  

- Mediana: 33.4°C  

- Intervalo: [29.2, 38.0]°C  

- Percentis 25-75: [31.6, 34.3]°C


#### Muito Quente (+3)


**Faixa Modal**: [6.7, 55.0]°C  

- Pico de probabilidade em 18.5°C 
(P = 100.0%)  

- Amplitude: 48.2°C


**Faixa com P ≥ 30%**: [0.8, 55.0]°C  

- Amplitude: 54.1°C


**Dados Observados** (n = 11):  

- Média: 38.9°C (DP: 2.3°C)  

- Mediana: 39.2°C  

- Intervalo: [33.9, 41.9]°C  

- Percentis 25-75: [37.7, 40.5]°C


## 6. Faixas de PET Observadas (Análise Descritiva)


Esta análise apresenta as faixas de PET baseadas diretamente nos dados 
coletados, sem depender de modelagem probabilística.


### Resumo das Faixas Observadas


| Sensação | N | Média (°C) | Faixa 50% (°C) | Faixa 80% (°C) | Amplitude Total (°C) |

|----------|---|------------|----------------|----------------|----------------------|

| Muito Frio (-3) | 11 | 9.9 | [9.1, 11.3] | [7.5, 11.9] | [5.2, 13.9] |

| Frio (-2) | 16 | 14.9 | [14.2, 15.4] | [13.3, 16.1] | [13.0, 17.2] |

| Frio Moderado (-1) | 27 | 19.7 | [18.9, 20.4] | [18.7, 20.8] | [18.5, 21.5] |

| Confortável (+0) | 42 | 23.9 | [22.9, 24.5] | [21.9, 25.5] | [21.1, 27.7] |

| Calor Moderado (+1) | 27 | 28.0 | [26.8, 28.7] | [26.5, 29.3] | [25.7, 30.8] |

| Quente (+2) | 16 | 33.2 | [31.6, 34.3] | [30.8, 35.5] | [29.2, 38.0] |

| Muito Quente (+3) | 11 | 38.9 | [37.7, 40.5] | [37.3, 41.4] | [33.9, 41.9] |



### Zona de Conforto Observada


- Faixa Central (50%): [22.9, 24.5]°C

- Faixa Ampla (80%): [21.9, 25.5]°C

- PET médio: 23.9°C


### Interpretação Detalhada das Faixas


As três faixas apresentadas representam diferentes níveis de confiança e abrangência, 
cada uma adequada para aplicações específicas. Todas são baseadas em **estatísticas descritivas robustas** 
calculadas diretamente dos dados observados, sem depender de suposições de distribuição probabilística.


#### 1. Faixa 50% (Intervalo Interquartil: P25-P75)


**Definição**: Intervalo entre o percentil 25 (P25) e o percentil 75 (P75), também conhecido como 
Intervalo Interquartil (IQR). Contém os 50% centrais das observações para cada categoria de sensação.


**Fundamentação Estatística**:

- Remove automaticamente os 25% mais baixos e 25% mais altos dos dados

- Altamente resistente a valores extremos e outliers

- Medida robusta de dispersão, amplamente utilizada em análise exploratória de dados

- Base para identificação de outliers pela regra de Tukey (IQR × 1.5)


**Por que é confiável?**

- **Robustez**: Não é afetada por valores extremos que podem ser erros de medição ou condições atípicas

- **Representatividade**: Captura o comportamento típico da maioria das pessoas

- **Estabilidade**: Menos sensível a variações amostrais que a média ou desvio padrão

- **Validação**: Método padrão em climatologia e estudos de conforto térmico


**Quando usar**:

- ✅ **Design urbano e arquitetônico**: Para garantir conforto para a maioria das pessoas

- ✅ **Normas e diretrizes**: Quando é necessário estabelecer faixas conservadoras

- ✅ **Projetos com alta exigência de conforto**: Espaços públicos, áreas de permanência

- ✅ **Comparação entre locais**: Faixa mais estável para comparações científicas


#### 2. Faixa 80% (P10-P90)


**Definição**: Intervalo entre o percentil 10 (P10) e o percentil 90 (P90). 
Contém 80% das observações centrais, excluindo apenas os 10% mais extremos de cada lado.


**Fundamentação Estatística**:

- Equilibra abrangência e robustez, incluindo variabilidade natural sem extremos

- Percentis P10 e P90 são pontos de corte comuns em análises climáticas

- Mantém resistência razoável a outliers enquanto captura maior variabilidade

- Aproxima-se de ±1.28 desvios padrão em distribuições normais


**Por que é confiável?**

- **Realismo**: Reflete a variabilidade natural do conforto térmico em condições reais

- **Abrangência**: Cobre a grande maioria dos casos sem incluir extremos raros

- **Aplicabilidade**: Útil para entender a amplitude esperada do fenômeno

- **Contexto climático**: Alinha-se com análises de variabilidade climática (decis)


**Quando usar**:

- ✅ **Análise de variabilidade**: Para entender a amplitude real do conforto térmico

- ✅ **Planejamento adaptativo**: Quando é necessário considerar maior diversidade de condições

- ✅ **Estudos de adaptação**: Para avaliar a capacidade de adaptação da população

- ✅ **Contexto de pesquisa**: Apresentar a variabilidade completa sem extremos


#### 3. Amplitude Total (Min-Max)


**Definição**: Intervalo completo dos dados observados, do valor mínimo absoluto ao valor máximo absoluto. 
Representa 100% das observações coletadas na pesquisa.


**Fundamentação Estatística**:

- Medida de dispersão mais simples e direta: Range = Max - Min

- Não faz suposições sobre a distribuição dos dados

- Sensível a todos os valores, incluindo outliers e casos extremos

- Aumenta com o tamanho da amostra (mais dados = maior chance de extremos)


**Por que é confiável?**

- **Completude**: Mostra os limites absolutos observados na pesquisa

- **Transparência**: Não oculta nenhum dado, apresenta a realidade completa

- **Contexto**: Essencial para identificar condições extremas que realmente ocorreram

- **Validação**: Permite verificar se há valores implausíveis ou erros de medição


**Quando usar**:

- ✅ **Identificação de extremos**: Para conhecer os limites absolutos observados

- ✅ **Análise de casos especiais**: Quando extremos são relevantes (ondas de calor/frio)

- ✅ **Contexto completo**: Para apresentar toda a amplitude de condições encontradas

- ✅ **Validação de dados**: Verificar se há valores fora do esperado


**⚠️ Atenção**: A amplitude total é sensível a outliers e aumenta com o tamanho da amostra. 
Valores extremos podem representar condições raras ou erros de medição. Use com cautela para design.


### Comparação e Recomendações de Uso


| Faixa | Abrangência | Robustez | Melhor Aplicação |

|-------|-------------|----------|------------------|

| **50% (IQR)** | 50% central | ⭐⭐⭐⭐⭐ Muito alta | Design urbano, normas |

| **80% (P10-P90)** | 80% central | ⭐⭐⭐⭐ Alta | Análise de variabilidade |

| **Total (Min-Max)** | 100% completo | ⭐⭐ Moderada | Contexto, extremos |


**Recomendação Geral**: Para a maioria dos projetos de design urbano e arquitetônico, 
recomenda-se usar a **Faixa 50%** como referência principal, consultando a **Faixa 80%** 
para entender a variabilidade esperada e a **Amplitude Total** para contexto completo.


### Faixa Única Recomendada para Cada Sensação


Para facilitar a aplicação prática dos resultados, apresentamos abaixo uma **faixa única** 
para cada categoria de sensação térmica, baseada no **Intervalo Interquartil (IQR)**, 
que corresponde à Faixa 50% (P25-P75) apresentada anteriormente.


#### Metodologia: Por que usar o Intervalo Interquartil (IQR)?


**Contexto**: Em pesquisas de percepção térmica com entrevistas, os dados apresentam 
características específicas que exigem métodos estatísticos robustos:


1. **Alta Variabilidade Individual**: Pessoas têm metabolismos, vestimentas e níveis de 
aclimatação diferentes, resultando em percepções térmicas variadas para o mesmo PET.


2. **Presença de Outliers**: Sempre existem respostas atípicas em pesquisas (erros de 
resposta, condições de saúde específicas, aclimatação extrema).


3. **Distribuição Não-Normal**: A percepção térmica humana raramente segue uma distribuição 
normal, tornando inadequados métodos baseados em média e desvio padrão.


**Solução: Intervalo Interquartil (IQR)**


O IQR é definido como o intervalo entre o percentil 25 (P25) e o percentil 75 (P75), 
contendo os **50% centrais** das observações. Esta é a escolha ideal porque:


✅ **Robustez**: Remove automaticamente os 25% mais extremos de cada lado, eliminando 
outliers sem perder informação relevante


✅ **Não-paramétrico**: Não assume distribuição normal, adequado para dados de percepção humana


✅ **Representatividade**: Captura o comportamento típico da maioria das pessoas, 
não casos extremos


✅ **Validação Científica**: Método padrão em normas internacionais (ISO 7730, ASHRAE 55) 
e amplamente usado em estudos de conforto térmico


✅ **Estabilidade**: Menos sensível a variações amostrais que média ou amplitude total


✅ **Aplicabilidade**: Ideal para design urbano e arquitetônico, onde se busca garantir 
conforto para a maioria das pessoas


#### Tabela de Faixas Únicas Recomendadas


| Sensação | N | Faixa Recomendada (°C) | Amplitude (°C) | PET Médio (°C) |

|----------|---|------------------------|----------------|----------------|

| Muito Frio (-3) | 11 | [9.1, 11.3] | 2.2 | 9.9 |

| Frio (-2) | 16 | [14.2, 15.4] | 1.2 | 14.9 |

| Frio Moderado (-1) | 27 | [18.9, 20.4] | 1.4 | 19.7 |

| Confortável (+0) | 42 | [22.9, 24.5] | 1.6 | 23.9 |

| Calor Moderado (+1) | 27 | [26.8, 28.7] | 1.9 | 28.0 |

| Quente (+2) | 16 | [31.6, 34.3] | 2.7 | 33.2 |

| Muito Quente (+3) | 11 | [37.7, 40.5] | 2.8 | 38.9 |



#### Interpretação da Tabela


**Faixa Recomendada**: Intervalo de PET onde 50% das pessoas reportaram aquela sensação térmica. 
Esta é a faixa mais confiável para uso em projetos de design urbano e arquitetônico.


**Amplitude**: Largura da faixa em graus Celsius. Amplitudes menores indicam maior consenso 
entre as pessoas sobre aquela sensação térmica.


**PET Médio**: Valor central de PET para aquela sensação. Útil como referência rápida.


#### Como Usar Estas Faixas


**Para Design Urbano e Arquitetônico**:


1. **Zona de Conforto Térmico**: Mantenha o PET entre **22.9°C e 24.5°C** 
para garantir que a maioria das pessoas se sinta confortável.


2. **Valor de Referência**: Use **23.9°C** como PET ideal para conforto térmico.


3. **Evitar Desconforto**: Identifique as faixas de sensações indesejadas (muito frio/quente) 
e projete para evitar que o PET atinja esses valores.


4. **Estratégias de Mitigação**: Para cada faixa de desconforto identificada, desenvolva 
estratégias específicas (sombreamento, ventilação, aquecimento).


#### Análise de Sobreposição entre Categorias


É importante notar que as faixas de diferentes sensações podem se sobrepor. Isso é **esperado e natural** 
em dados de percepção humana, pois:


- Pessoas têm diferentes níveis de sensibilidade térmica

- A aclimatação local influencia a percepção

- Fatores individuais (idade, metabolismo, vestimenta) afetam o conforto


**Sobreposições observadas**:


- **Muito Frio** e **Frio**: Sem sobreposição (gap de 2.9°C)

- **Frio** e **Frio Moderado**: Sem sobreposição (gap de 3.6°C)

- **Frio Moderado** e **Confortável**: Sem sobreposição (gap de 2.5°C)

- **Confortável** e **Calor Moderado**: Sem sobreposição (gap de 2.3°C)

- **Calor Moderado** e **Quente**: Sem sobreposição (gap de 2.9°C)

- **Quente** e **Muito Quente**: Sem sobreposição (gap de 3.4°C)



**Implicação Prática**: Em zonas de sobreposição, diferentes pessoas podem ter percepções diferentes. 
Para design, priorize manter o PET dentro da faixa de conforto.


#### Validação Científica


O método do Intervalo Interquartil (IQR) é:


✅ **ISO 7730**: Norma internacional para ambientes térmicos


✅ **ASHRAE 55**: Padrão americano para conforto térmico


✅ **Literatura**: Nikolopoulou & Lykoudis (2006), Matzarakis et al. (1999)


**Seus dados**: Conforto em [22.9, 24.5]°C (média: 23.9°C)


💡 **Dica**: Diferenças em relação à literatura indicam adaptação climática local!


💡 **Nota**: Faixas baseadas exclusivamente nos dados observados.


## 7. Faixas de Aceitabilidade (Análise Complementar)

As faixas de aceitabilidade são baseadas em um modelo logístico binário 
separado e fornecem uma perspectiva complementar sobre o conforto térmico.

## 6. Visualizações

### Relação PET vs Sensação Térmica

![Scatter TSV vs PET](scatter_TSV_PET.png)

Gráfico de dispersão mostrando a relação entre PET e sensação 
térmica ordinal. A linha vertical vermelha indica o PET neutro.

### Curvas de Probabilidade por Categoria

![Curvas de Probabilidade](probs_ordinais_PET.png)

Probabilidades de cada categoria de sensação térmica em função 
do PET. As regiões sombreadas indicam as faixas de conforto 
(80% e 90%).

### Zona de Conforto Térmico

![Zona de Conforto](zona_conforto_logit.png)

Probabilidade de conforto (P(-1 ≤ Y ≤ +1)) em função do PET. 
As linhas horizontais indicam os limiares de 80% e 90%, e as 
linhas verticais marcam os limites das faixas de conforto.

## 7. Interpretação dos Resultados

### Como usar as faixas de conforto

1. **Faixa 80%**: Recomendada para aplicações gerais de planejamento 
   urbano e design de espaços externos. Garante que a maioria das 
   pessoas (80%) se sentirá confortável.

2. **Faixa 90%**: Recomendada para espaços que requerem maior rigor 
   de conforto, como áreas de permanência prolongada ou populações 
   sensíveis.

3. **PET Neutro**: Representa a temperatura ideal de conforto térmico 
   para a população estudada. Pode ser usado como referência para 
   estratégias de mitigação térmica.

### Limitações e considerações

- Os resultados são específicos para a população e contexto climático 
  estudados. Extrapolações para outras regiões devem ser feitas com cautela.

- O modelo assume proporcionalidade dos odds (proportional odds assumption). 
  Violações desta suposição podem afetar a precisão das estimativas.

- O tamanho amostral (N = 150) influencia a precisão 
  dos intervalos de confiança. Amostras maiores produzem estimativas 
  mais precisas.

## 8. Referências

### Metodologia Estatística

- **McCullagh, P.** (1980). Regression Models for Ordinal Data. 
  *Journal of the Royal Statistical Society: Series B*, 42(2), 109-127.

- **Agresti, A.** (2010). *Analysis of Ordinal Categorical Data* 
  (2nd ed.). Wiley.

### Índice PET

- **Höppe, P.** (1999). The physiological equivalent temperature - 
  a universal index for the biometeorological assessment of the thermal 
  environment. *International Journal of Biometeorology*, 43(2), 71-75.

- **Matzarakis, A., Mayer, H., & Iziomon, M. G.** (1999). Applications 
  of a universal thermal index: physiological equivalent temperature. 
  *International Journal of Biometeorology*, 43(2), 76-84.

### Conforto Térmico

- **ASHRAE** (2020). *ASHRAE Standard 55: Thermal Environmental 
  Conditions for Human Occupancy*. American Society of Heating, 
  Refrigerating and Air-Conditioning Engineers.

- **ISO 7730** (2005). *Ergonomics of the thermal environment - 
  Analytical determination and interpretation of thermal comfort using 
  calculation of the PMV and PPD indices and local thermal comfort criteria*. 
  International Organization for Standardization.

---

*Relatório gerado automaticamente pelo PET Thermal Comfort Calibrator*  
*Desenvolvido por Carol Freire do Santos - Doutorado em Climatologia, USP*
