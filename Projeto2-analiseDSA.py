#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Data Science Academy</font>
# 
# ## <font color='blue'>Fundamentos de Linguagem Python Para Análise de Dados e Data Science</font>
# 
# ## <font color='blue'>Projeto 2</font>
# 
# ## <font color='blue'>Análise Exploratória de Dados em Linguagem Python Para a Área de Varejo</font>

# ![DSA](imagens/projeto2.png)

# In[1]:


# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())


# In[2]:


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# ## Carregando os Dados

# In[3]:


# Carrega o dataset
df_dsa = pd.read_csv('dados/dataset.csv')


# In[4]:


# Shape
df_dsa.shape


# In[5]:


# Amostra dos dados
df_dsa.head()


# In[6]:


# Amostra dos dados
df_dsa.tail()


# ## Análise Exploratória

# In[7]:


# Colunas do conjunto de dados
df_dsa.columns


# In[8]:


# Verificando o tipo de dado de cada coluna
df_dsa.dtypes


# In[9]:


# Resumo estatístico da coluna com o valor de venda
df_dsa['Valor_Venda'].describe()


# In[10]:


# Verificando se há registros duplicados
df_dsa[df_dsa.duplicated()]


# In[11]:


# Verificando de há valores ausentes
df_dsa.isnull().sum()


# In[12]:


df_dsa.head()


# ## Pergunta de Negócio 1:
# 
# ### Qual Cidade com Maior Valor de Venda de Produtos da Categoria 'Office Supplies'?

# In[13]:


produtos_office_supplies = df_dsa[df_dsa['Categoria'] == 'Office Supplies']
vendas_por_cidade = produtos_office_supplies.groupby('Cidade')['Valor_Venda'].sum()
cidade_maior_venda = vendas_por_cidade.idxmax()
maior_valor_venda = vendas_por_cidade.max()
print(f"A cidade com o maior valor de vendas para 'Office Supplies' é {cidade_maior_venda} com um valor total de vendas de {maior_valor_venda}.")


# ## Pergunta de Negócio 2:
# 
# ### Qual o Total de Vendas Por Data do Pedido?
# 
# Demonstre o resultado através de um gráfico de barras.

# In[27]:


total_vendas_por_data = df_dsa.groupby('Data_Pedido')['Valor_Venda'].sum()
plt.figure(figsize = (20, 6))
df_dsa.plot(x = 'Data_Pedido', y = 'Valor_Venda', color = 'green')
plt.title('Total de Vendas Por Data do Pedido')
plt.show()


# ## Pergunta de Negócio 3:
# 
# ### Qual o Total de Vendas por Estado?
# 
# Demonstre o resultado através de um gráfico de barras.

# In[15]:


total_vendas_por_estado = df_dsa.groupby('Estado')['Valor_Venda'].sum()
plt.figure(figsize=(12, 6))
total_vendas_por_estado.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Total de Vendas por Estado')
plt.xlabel('Estado')
plt.ylabel('Total de Vendas')
plt.show()


# ## Pergunta de Negócio 4:
# 
# ### Quais São as 10 Cidades com Maior Total de Vendas?
# 
# Demonstre o resultado através de um gráfico de barras.

# In[18]:


total_vendas_por_cidade = df_dsa.groupby('Cidade')['Valor_Venda'].sum()
top_10_cidades = total_vendas_por_cidade.nlargest(10)
plt.figure(figsize=(12, 6))
top_10_cidades.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Top 10 Cidades com Maior Total de Vendas')
plt.xlabel('Cidade')
plt.ylabel('Total de Vendas')
plt.show()


# ### Pergunta de Negócio 5:
# 
# ### Qual Segmento Teve o Maior Total de Vendas?
# 
# Demonstre o resultado através de um gráfico de pizza.

# In[19]:


total_vendas_por_segmento = df_dsa.groupby('Segmento')['Valor_Venda'].sum()
segmento_maior_venda = total_vendas_por_segmento.idxmax()
plt.figure(figsize=(8, 8))
total_vendas_por_segmento.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue', 'lightgreen'])
plt.title('Distribuição de Vendas por Segmento')
plt.ylabel('')  # Remover o rótulo do eixo y
plt.show()

print(f"O segmento com o maior total de vendas é '{segmento_maior_venda}'.")


# ## Pergunta de Negócio 6 (Desafio Nível Baby):
# 
# ### Qual o Total de Vendas Por Segmento e Por Ano?

# In[28]:


df_dsa['Data_Pedido'] = pd.to_datetime(df_dsa['Data_Pedido'], dayfirst = True)
df_dsa['Ano'] = df_dsa['Data_Pedido'].dt.year
total_vendas_por_segmento_ano = df_dsa.groupby(['Ano', 'Segmento'])['Valor_Venda'].sum()
print(total_vendas_por_segmento_ano)


# ## Pergunta de Negócio 7 (Desafio Nível Júnior):
# 
# Os gestores da empresa estão considerando conceder diferentes faixas de descontos e gostariam de fazer uma simulação com base na regra abaixo:
# 
# - Se o Valor_Venda for maior que 1000 recebe 15% de desconto.
# - Se o Valor_Venda for menor que 1000 recebe 10% de desconto.
# 
# ### Quantas Vendas Receberiam 15% de Desconto?

# In[22]:


df_dsa['Desconto'] = df_dsa['Valor_Venda'].apply(lambda x: 0.15 if x > 1000 else 0.10)
vendas_com_desconto_15 = df_dsa[df_dsa['Desconto'] == 0.15]
numero_vendas_desconto_15 = len(vendas_com_desconto_15)
print(f"O número de vendas que receberiam 15% de desconto é: {numero_vendas_desconto_15}")


# ## Pergunta de Negócio 8 (Desafio Nível Master):
# 
# ### Considere Que a Empresa Decida Conceder o Desconto de 15% do Item Anterior. Qual Seria a Média do Valor de Venda Antes e Depois do Desconto?

# In[23]:


df_dsa['Valor_Venda_Com_Desconto'] = df_dsa['Valor_Venda'] * (1 - df_dsa['Desconto'])
media_valor_venda_antes_desconto = df_dsa['Valor_Venda'].mean()
media_valor_venda_com_desconto = df_dsa['Valor_Venda_Com_Desconto'].mean()
print(f"A média do valor de venda antes do desconto é: {media_valor_venda_antes_desconto:.2f}")
print(f"A média do valor de venda após o desconto de 15% é: {media_valor_venda_com_desconto:.2f}")


# ## Pergunta de Negócio 9 (Desafio Nível Master Ninja):
# 
# ### Qual o Média de Vendas Por Segmento, Por Ano e Por Mês?
# 
# Demonstre o resultado através de gráfico de linha.

# In[24]:


df_dsa['Data_Pedido'] = pd.to_datetime(df_dsa['Data_Pedido'])
df_dsa['Ano'] = df_dsa['Data_Pedido'].dt.year
df_dsa['Mês'] = df_dsa['Data_Pedido'].dt.month

media_vendas_por_segmento_ano_mes = df_dsa.groupby(['Segmento', 'Ano', 'Mês'])['Valor_Venda'].mean().reset_index()

plt.figure(figsize=(12, 6))
for segmento in media_vendas_por_segmento_ano_mes['Segmento'].unique():
    segmento_data = media_vendas_por_segmento_ano_mes[media_vendas_por_segmento_ano_mes['Segmento'] == segmento]
    plt.plot(segmento_data['Ano'].astype(str) + '-' + segmento_data['Mês'].astype(str), segmento_data['Valor_Venda'], label=segmento)

plt.title('Média de Vendas Por Segmento, Por Ano e Por Mês')
plt.xlabel('Ano e Mês')
plt.ylabel('Média de Vendas')
plt.legend(title='Segmento', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ## Pergunta de Negócio 10 (Desafio Nível Master Ninja das Galáxias):
# 
# ### Qual o Total de Vendas Por Categoria e SubCategoria, Considerando Somente as Top 12 SubCategorias? 
# 
# Demonstre tudo através de um único gráfico.

# In[25]:


total_vendas_por_categoria_subcategoria = df_dsa.groupby(['Categoria', 'SubCategoria'])['Valor_Venda'].sum()

top_12_subcategorias = total_vendas_por_categoria_subcategoria.nlargest(12)

plt.figure(figsize=(12, 6))
top_12_subcategorias.sort_values(ascending=True).plot(kind='barh', color='skyblue')
plt.title('Total de Vendas Por Categoria e SubCategoria (Top 12)')
plt.xlabel('Total de Vendas')
plt.ylabel('Categoria e SubCategoria')
plt.show()


# # Fim
