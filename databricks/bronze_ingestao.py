# Databricks notebook source
# MAGIC %md
# MAGIC # DataBridge - Camada Bronze: Ingestão de Produtos
# MAGIC **Objetivo:** Ler os dados brutos de produtos do almoxarifado, aplicar o schema explícito, adicionar colunas de controle e salvar na tabela Delta `bronze_produtos`.
# MAGIC **Stack:** PySpark, Delta Lake, Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Importação de Bibliotecas e Configuração de Logs

# COMMAND ----------

import logging
import traceback
from datetime import datetime
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.sql.functions import current_timestamp, lit, expr
from pyspark.sql import Row

# Configuração de log estruturado
logger = logging.getLogger("DataBridge_Ingestao")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.info("Iniciando notebook de ingestão: 01_ingestao")

# Caminhos e tabelas (Unity Catalog)
ARQUIVO_ORIGEM = "/Volumes/databridge/raw_data/files/produtos.csv"
TABELA_DESTINO = "databridge.raw_data.bronze_produtos"
TABELA_AUDITORIA = "databridge.raw_data.audit_logs"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Definição de Schemas

# COMMAND ----------

# Schema dos dados brutos — lendo quantidade e preco como String na Bronze para não perder dados sujos
schema_produtos = StructType([
    StructField("sistema_origem", StringType(), True),
    StructField("nome_produto",   StringType(), True),
    StructField("marca",          StringType(), True),
    StructField("categoria",      StringType(), True),
    StructField("unidade",        StringType(), True),
    StructField("quantidade",     StringType(), True),
    StructField("preco",          StringType(), True)
])

# Schema da tabela de auditoria
schema_audit = StructType([
    StructField("etapa",          StringType(),   True),
    StructField("status",         StringType(),   True),
    StructField("qtd_registros",  IntegerType(),  True),
    StructField("dt_execucao",    TimestampType(),True),
    StructField("mensagem",       StringType(),   True)
])

logger.info("Schemas definidos com sucesso.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Função de Auditoria

# COMMAND ----------

def registrar_auditoria(etapa, status, qtd_registros, mensagem=""):
    try:
        linha_audit = Row(
            etapa=etapa,
            status=status,
            qtd_registros=qtd_registros,
            dt_execucao=datetime.now(),
            mensagem=mensagem
        )
        df_audit = spark.createDataFrame([linha_audit], schema=schema_audit)
        df_audit.write \
            .format("delta") \
            .mode("append") \
            .saveAsTable(TABELA_AUDITORIA)
        logger.info(f"Auditoria registrada: Etapa={etapa} | Status={status} | Qtd={qtd_registros}")
    except Exception as e:
        logger.error(f"Erro ao registrar auditoria: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Ingestão Principal

# COMMAND ----------

logger.info("Iniciando leitura do arquivo CSV...")

try:
    # Leitura do CSV com schema explícito
    df_raw = spark.read \
        .format("csv") \
        .option("header", "true") \
        .schema(schema_produtos) \
        .load(ARQUIVO_ORIGEM)

    qtd_linhas_raw = df_raw.count()
    logger.info(f"Leitura concluída. {qtd_linhas_raw} registros encontrados.")

    # Adição de colunas de controle
    df_bronze = df_raw \
        .withColumn("id_registro",    expr("uuid()")) \
        .withColumn("dt_ingestao",    current_timestamp()) \
        .withColumn("arquivo_origem", lit("produtos.csv"))

    # Gravação na tabela Delta Bronze
    logger.info(f"Gravando na tabela Delta: {TABELA_DESTINO}...")
    df_bronze.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .saveAsTable(TABELA_DESTINO)

    qtd_linhas_gravadas = spark.table(TABELA_DESTINO).count()
    logger.info(f"Gravação concluída! Total na tabela: {qtd_linhas_gravadas}")

    registrar_auditoria(
        etapa="ingestao",
        status="sucesso",
        qtd_registros=qtd_linhas_raw,
        mensagem=f"Arquivo processado. Total acumulado: {qtd_linhas_gravadas}"
    )

except Exception as e:
    erro_msg = f"Falha na ingestão: {str(e)}"
    logger.error(erro_msg)
    logger.error(traceback.format_exc())
    registrar_auditoria(
        etapa="ingestao",
        status="erro",
        qtd_registros=0,
        mensagem=erro_msg[:250]
    )
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Validação Visual

# COMMAND ----------

print(f"--- RESUMO DA INGESTÃO ---")
print(f"Tabela Alvo     : {TABELA_DESTINO}")
print(f"Registros Batch : {qtd_linhas_raw}")
print("-" * 30)
print("Amostra de 5 registros:")
display(spark.table(TABELA_DESTINO).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Log de Auditoria

# COMMAND ----------

display(
    spark.table(TABELA_AUDITORIA)
    .orderBy(expr("dt_execucao DESC"))
    .limit(5)
)