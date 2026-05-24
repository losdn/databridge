# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline DataBridge: 03_classificacao_hierarquica
# MAGIC **Descrição:** Notebook que utiliza Processamento de Linguagem Natural (Hugging Face Zero-Shot) para classificar os produtos limpos em 4 níveis hierárquicos.
# MAGIC **Autor:** Engenharia de Dados
# MAGIC **Stack:** PySpark, Hugging Face, Delta Lake, Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Instalação de dependências

# COMMAND ----------

import importlib
import subprocess
import sys

def instalar_se_necessario(pacote, import_nome=None):
    nome = import_nome or pacote
    if importlib.util.find_spec(nome) is None:
        print(f"📦 Instalando {pacote}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pacote, "--quiet"])
        print(f"✅ {pacote} instalado!")
    else:
        print(f"✅ {pacote} já disponível, pulando.")

instalar_se_necessario("transformers")
instalar_se_necessario("torch")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Importações e Configuração de Logs

# COMMAND ----------

import logging
from datetime import datetime
from pyspark.sql.functions import col, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType

# Configuração de log estruturado em português
logger = logging.getLogger("DataBridge_Classificacao")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.info("Bibliotecas do PySpark importadas e logger configurado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Definição de Variáveis e Schemas

# COMMAND ----------

# Variáveis (Padrão Unity Catalog: catalog.schema.table)
# Nota: No Free Tier puro (Community Edition), o catálogo raiz nativo costuma ser 'hive_metastore'.
TABELA_SILVER = "databridge.raw_data.silver_produtos"
TABELA_CLASSIFICADA = "databridge.raw_data.silver_classificado"
TABELA_AUDITORIA = "databridge.raw_data.audit_logs"

# Schema explícito para o Log de Auditoria
schema_auditoria = StructType([
    StructField("etapa", StringType(), True),
    StructField("status", StringType(), True),
    StructField("qtd_registros", IntegerType(), True),
    StructField("dt_execucao", TimestampType(), True),
    StructField("mensagem_erro", StringType(), True)
])

logger.info("Variáveis e schema de auditoria definidos com sucesso.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Função de Auditoria

# COMMAND ----------

def registrar_auditoria(etapa, status, qtd_registros=0, mensagem_erro=None):
    """
    Grava eventos na tabela de auditoria garantindo o uso do Schema Explícito (StructType).
    """
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TABELA_AUDITORIA} (
        etapa STRING,
        status STRING,
        qtd_registros INT,
        dt_execucao TIMESTAMP,
        mensagem_erro STRING
    ) USING DELTA
    """)
    
    # Cria o DataFrame usando explicitamente o schema criado
    df_audit = spark.createDataFrame(
        [(etapa, status, qtd_registros, datetime.now(), mensagem_erro)],
        schema=schema_auditoria
    )
    
    df_audit.write.format("delta").mode("append").saveAsTable(TABELA_AUDITORIA)
    logger.info(f"Auditoria salva: {etapa} | {status} | Registros: {qtd_registros}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Motor de Classificação NLP (Zero-Shot com mapInPandas)

# COMMAND ----------

# COMMAND ----------
# Baixa o modelo no driver primeiro
from transformers import pipeline as hf_pipeline
import pandas as pd

logger.info("Baixando modelo no driver...")
classifier = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
logger.info("Modelo carregado com sucesso!")

# COMMAND ----------
# Classificação no driver (sem mapInPandas)
cand_n1 = ["Material", "Equipamento", "Serviço"]
cand_n2 = ["Fixadores", "Elétrica", "Hidráulica", "Mecânica", "Pneumática", "EPI", "Lubrificantes"]
cand_n3 = ["Parafusos", "Porcas", "Válvulas", "Filtros", "Rolamentos", "Cabos", "Óleo", "Luvas"]
cand_n4 = ["Parafuso Sextavado", "Porca Inox", "Rolamento de Esfera", "Óleo Hidráulico", "Cabo Flexível"]

df_pandas = spark.table(TABELA_SILVER).toPandas()
qtd_registros = len(df_pandas)

n1_list, n2_list, n3_list, n4_list, scores = [], [], [], [], []

for i, row in df_pandas.iterrows():
    texto = f"{row['nome_produto']} - {row['categoria']}"
    try:
        r1 = classifier(texto, candidate_labels=cand_n1)
        r2 = classifier(texto, candidate_labels=cand_n2)
        r3 = classifier(texto, candidate_labels=cand_n3)
        r4 = classifier(texto, candidate_labels=cand_n4)
        n1_list.append(r1['labels'][0])
        n2_list.append(r2['labels'][0])
        n3_list.append(r3['labels'][0])
        n4_list.append(r4['labels'][0])
        scores.append(round((r1['scores'][0]+r2['scores'][0]+r3['scores'][0]+r4['scores'][0])/4, 4))
    except:
        n1_list.append("erro"); n2_list.append("erro")
        n3_list.append("erro"); n4_list.append("erro")
        scores.append(0.0)
    
    if i % 10 == 0:
        logger.info(f"Processados {i}/{qtd_registros} produtos...")

df_pandas['nivel_1'] = n1_list
df_pandas['nivel_2'] = n2_list
df_pandas['nivel_3'] = n3_list
df_pandas['nivel_4'] = n4_list
df_pandas['score_classificacao'] = scores

df_classificado = spark.createDataFrame(df_pandas).withColumn("dt_atualizacao_ml", current_timestamp())

df_classificado.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(TABELA_CLASSIFICADA)
registrar_auditoria("classificacao_nlp", "sucesso", qtd_registros)
logger.info("Classificação concluída!")
display(df_classificado.select("nome_produto","nivel_1","nivel_2","nivel_3","nivel_4","score_classificacao").limit(10))