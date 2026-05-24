# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline DataBridge: 06_audit_trail
# MAGIC **Descrição:** Notebook responsável por ler os logs de auditoria, calcular métricas consolidadas do pipeline (tempo total, taxa de deduplicação) e exibir um dashboard textual resumido.
# MAGIC **Autor:** Engenharia de Dados
# MAGIC **Stack:** PySpark, Delta Lake, Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Importação de Bibliotecas e Configuração

# COMMAND ----------

import logging
from datetime import datetime
from pyspark.sql.functions import col, min, max, sum as spark_sum, current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

logger = logging.getLogger("DataBridge_Audit")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.info("Bibliotecas importadas com sucesso.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Definição de Variáveis (Unity Catalog)

# COMMAND ----------

TABELA_AUDITORIA = "databridge.raw_data.audit_logs"
TABELA_BRONZE    = "databridge.raw_data.bronze_produtos"
TABELA_GOLD      = "databridge.raw_data.gold_produtos_unificados"
TABELA_CONSOLIDADA = "databridge.raw_data.gold_audit_consolidado"

logger.info("Variáveis de ambiente do Unity Catalog definidas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Cálculo de Métricas de Negócio e Deduplicação

# COMMAND ----------

qtd_bronze = 0
qtd_gold = 0
qtd_deduplicados = 0

try:
    logger.info("Calculando taxa de deduplicação entre Bronze e Gold...")
    if spark.catalog.tableExists(TABELA_BRONZE):
        qtd_bronze = spark.table(TABELA_BRONZE).count()
    if spark.catalog.tableExists(TABELA_GOLD):
        qtd_gold = spark.table(TABELA_GOLD).count()
    qtd_deduplicados = qtd_bronze - qtd_gold
except Exception as e:
    logger.warning(f"Aviso ao calcular deduplicação: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Processamento dos Logs de Auditoria

# COMMAND ----------

try:
    logger.info(f"Lendo logs de auditoria da tabela {TABELA_AUDITORIA}...")
    df_audit = spark.table(TABELA_AUDITORIA)

    df_tempo = df_audit.select(
        min("inicio").alias("inicio"),
        max("fim").alias("fim")
    ).collect()[0]

    inicio_pipeline = df_tempo["inicio"]
    fim_pipeline    = df_tempo["fim"]

    if inicio_pipeline and fim_pipeline:
        tempo_total_segundos = int((fim_pipeline - inicio_pipeline).total_seconds())
        tempo_total_minutos  = round(tempo_total_segundos / 60, 2)
    else:
        tempo_total_segundos = 0
        tempo_total_minutos  = 0.0

    df_etapas = df_audit.groupBy("nome_job", "status") \
        .agg(spark_sum("registros_lidos").alias("total_registros_processados")) \
        .orderBy("nome_job")

    etapas_list  = df_etapas.collect()
    detalhes_str = " | ".join([
        f"{r['nome_job']}({r['status']}): {r['total_registros_processados']}"
        for r in etapas_list
    ])

except Exception as e:
    msg = str(e)
    logger.error(f"Erro ao processar auditoria: {msg}")
    raise Exception(f"Falha na leitura dos logs de auditoria: {msg}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Salvar View Consolidada (Gold Audit)

# COMMAND ----------

try:
    logger.info(f"Gerando tabela consolidada em {TABELA_CONSOLIDADA}...")

    schema_consolidado = StructType([
        StructField("dt_consolidacao",          TimestampType(), True),
        StructField("qtd_bruta_bronze",          IntegerType(),   True),
        StructField("qtd_unificada_gold",        IntegerType(),   True),
        StructField("qtd_deduplicada",           IntegerType(),   True),
        StructField("tempo_pipeline_segundos",   IntegerType(),   True),
        StructField("resumo_etapas",             StringType(),    True)
    ])

    dados_consolidados = [(
        datetime.now(),
        int(qtd_bronze),
        int(qtd_gold),
        int(qtd_deduplicados),
        int(tempo_total_segundos),
        detalhes_str
    )]

    df_consolidado = spark.createDataFrame(dados_consolidados, schema=schema_consolidado)

    df_consolidado.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(TABELA_CONSOLIDADA)

    logger.info("Tabela consolidada salva com sucesso.")

except Exception as e:
    logger.error(f"Erro ao salvar a tabela consolidada: {e}")
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Dashboard Textual de Execução

# COMMAND ----------

print("=" * 70)
print(" 🚀 DASHBOARD CONSOLIDADO DE EXECUÇÃO - DATABRIDGE ")
print("=" * 70)
print(f"🕒 Início do Pipeline  : {inicio_pipeline.strftime('%Y-%m-%d %H:%M:%S') if inicio_pipeline else 'N/A'}")
print(f"🏁 Fim do Pipeline     : {fim_pipeline.strftime('%Y-%m-%d %H:%M:%S') if fim_pipeline else 'N/A'}")
print(f"⏱️  Tempo Total         : {tempo_total_minutos} minutos ({tempo_total_segundos} seg)")
print("-" * 70)
print(" 📦 FUNIL DE DADOS (DEDUPLICAÇÃO) ")
print("-" * 70)
print(f"📥 Entradas (Bronze)   : {qtd_bronze} produtos")
print(f"💎 Saídas (Gold)       : {qtd_gold} produtos unificados")
print(f"🗑️  Deduplicados/Sujos  : {qtd_deduplicados} registros eliminados")
print("-" * 70)
print(" 📋 STATUS POR ETAPA ")
print("-" * 70)
for r in etapas_list:
    print(f"  ▸ {r['nome_job']:<40} | {r['status']:<10} | {r['total_registros_processados']} registros")
print("=" * 70)
print(" ✅ Pipeline DataBridge concluído com sucesso! ")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Amostra da tabela consolidada

# COMMAND ----------

display(spark.table(TABELA_CONSOLIDADA))
