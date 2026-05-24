# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline DataBridge: 02_limpeza_padronizacao
# MAGIC **Descrição:** Notebook responsável por ler os dados da camada Bronze, aplicar regras de qualidade de dados (limpeza, padronização e tratamentos de nulos) e salvar na camada Silver via Unity Catalog.
# MAGIC **Autor:** Engenharia de Dados
# MAGIC **Stack:** PySpark, Delta Lake, Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Importação de Bibliotecas e Configuração de Logs

# COMMAND ----------

import logging
from pyspark.sql.functions import (
    col, lower, trim, regexp_replace, translate, when, lit, current_timestamp
)
from datetime import datetime

# Configuração de log estruturado em português
logger = logging.getLogger("DataBridge_Limpeza")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.info("Bibliotecas importadas e logger configurado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Definição de Variáveis (Unity Catalog)

# COMMAND ----------

# Padrão Unity Catalog: catalog.schema.table
TABELA_BRONZE = "databridge.raw_data.bronze_produtos"
TABELA_SILVER = "databridge.raw_data.silver_produtos"
TABELA_AUDITORIA = "databridge.raw_data.audit_logs"

spark.sql("DROP TABLE IF EXISTS databridge.raw_data.audit_logs")
spark.sql("DROP TABLE IF EXISTS databridge.raw_data.silver_produtos")

logger.info(f"Tabelas de Origem/Destino configuradas. Origem: {TABELA_BRONZE} | Destino: {TABELA_SILVER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Função de Auditoria

# COMMAND ----------

def registrar_auditoria(etapa, status, qtd_registros=0, mensagem_erro=None):
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
    
    schema_audit = StructType([
        StructField("etapa",         StringType(),   True),
        StructField("status",        StringType(),   True),
        StructField("qtd_registros", IntegerType(),  True),
        StructField("dt_execucao",   TimestampType(),True),
        StructField("mensagem_erro", StringType(),   True)
    ])

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TABELA_AUDITORIA} (
        etapa STRING,
        status STRING,
        qtd_registros INT,
        dt_execucao TIMESTAMP,
        mensagem_erro STRING
    ) USING DELTA
    """)

    from pyspark.sql import Row
    linha = Row(
        etapa=etapa,
        status=status,
        qtd_registros=int(qtd_registros),
        dt_execucao=datetime.now(),
        mensagem_erro=str(mensagem_erro) if mensagem_erro else None
    )
    df_audit = spark.createDataFrame([linha], schema=schema_audit)
    df_audit.write.format("delta").mode("append").saveAsTable(TABELA_AUDITORIA)
    logger.info(f"Auditoria registrada: Etapa={etapa}, Status={status}, Registros={qtd_registros}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Processamento: Limpeza e Padronização

# COMMAND ----------

try:
    logger.info(f"Lendo dados da tabela Bronze: {TABELA_BRONZE}...")
    
    # Leitura da tabela Bronze
    df_bronze = spark.table(TABELA_BRONZE)
    qtd_lida = df_bronze.count()
    
    if qtd_lida == 0:
        logger.warning("A tabela Bronze está vazia. Nenhuma ação será tomada.")
        registrar_auditoria("limpeza_padronizacao", "alerta_vazio", 0, "Tabela Bronze sem registros")
    else:
        logger.info("Iniciando transformações e regras de qualidade...")

        # ---------------------------------------------------------
        # REGRA 1: Normalizar nome_produto
        # ---------------------------------------------------------
        # Converte para minúsculas
        df_silver = df_bronze.withColumn("nome_produto", lower(col("nome_produto")))
        # Remove acentuação (mapeamento de caracteres com acento para sem acento)
        caracteres_com_acento = 'áàâãäéèêëíìîïóòôõöúùûüç'
        caracteres_sem_acento = 'aaaaaeeeeiiiiooooouuuuc'
        df_silver = df_silver.withColumn("nome_produto", translate(col("nome_produto"), caracteres_com_acento, caracteres_sem_acento))
        # Remove caracteres especiais (mantém apenas letras, números e espaços)
        df_silver = df_silver.withColumn("nome_produto", regexp_replace(col("nome_produto"), '[^a-z0-9 ]', ''))
        # Colapsa espaços duplos e remove espaços em branco nas extremidades
        df_silver = df_silver.withColumn("nome_produto", trim(regexp_replace(col("nome_produto"), '\\s+', ' ')))

        # ---------------------------------------------------------
        # REGRA 2: Tratar nulos em marca e categoria
        # ---------------------------------------------------------
        df_silver = df_silver.withColumn("marca", 
            when((col("marca").isNull()) | (trim(col("marca")) == ""), "desconhecida")
            .otherwise(trim(col("marca")))
        )
        df_silver = df_silver.withColumn("categoria", 
            when((col("categoria").isNull()) | (trim(col("categoria")) == ""), "sem_categoria")
            .otherwise(trim(col("categoria")))
        )

        # ---------------------------------------------------------
        # REGRA 3: Tratar quantidade
        # ---------------------------------------------------------
        df_silver = df_silver.withColumn("quantidade", col("quantidade").cast("int"))
        df_silver = df_silver.withColumn("quantidade", 
            when(col("quantidade").isNull() | (col("quantidade") < 0), lit(0))
            .otherwise(col("quantidade"))
        )

        # ---------------------------------------------------------
        # REGRA 4: Tratar preco
        # ---------------------------------------------------------
        df_silver = df_silver.withColumn("preco", col("preco").cast("double"))
        df_silver = df_silver.withColumn("preco", 
            when(col("preco").isNull() | (col("preco") > 99999), lit(None).cast("double"))
            .otherwise(col("preco"))
        )

        # ---------------------------------------------------------
        # REGRA 5: Padronizar unidade
        # ---------------------------------------------------------
        # Coluna temporária para facilitar a avaliação
        df_silver = df_silver.withColumn("unidade_norm", lower(trim(col("unidade"))))
        
        df_silver = df_silver.withColumn("unidade", 
            when(col("unidade_norm").isin('kilo', '1000g', 'quilos'), 'kg')
            .when(col("unidade_norm").isin('metro', 'mts'), 'm')
            .when(col("unidade_norm").isin('litro', 'lts'), 'l')
            .otherwise(col("unidade_norm")) # Se não cair na regra, mantém o valor original em minúsculas
        ).drop("unidade_norm") # Remove a coluna temporária
        
        # Adiciona a data de atualização na camada Silver
        df_silver = df_silver.withColumn("dt_atualizacao_silver", current_timestamp())

        logger.info(f"Transformações aplicadas. Salvando na tabela Silver: {TABELA_SILVER}...")
        
        # ---------------------------------------------------------
        # GRAVAÇÃO
        # ---------------------------------------------------------
        # Utilizando o modo 'overwrite' para reescrever a Silver com os dados limpos
        df_silver.write \
            .format("delta") \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .saveAsTable(TABELA_SILVER)
            
        logger.info("Processamento da camada Silver concluído com sucesso.")
        
        # Registrar sucesso
        registrar_auditoria("limpeza_padronizacao", "sucesso", qtd_lida)
        
        # Output visual
        print("-" * 50)
        print("✅ Limpeza e Padronização finalizada (Bronze -> Silver)")
        print(f"📊 Registros processados: {qtd_lida}")
        print(f"📂 Destino: {TABELA_SILVER}")
        print("-" * 50)
        
        display(spark.sql(f"SELECT nome_produto, marca, categoria, unidade, quantidade, preco FROM {TABELA_SILVER} LIMIT 5"))

except Exception as e:
    erro_msg = str(e)
    logger.error(f"Falha na camada Silver. Erro: {erro_msg}")
    
    registrar_auditoria("limpeza_padronizacao", "erro", 0, erro_msg)
    
    raise Exception(f"Erro no pipeline de Limpeza (DataBridge): {erro_msg}")