# Databricks notebook source
# MAGIC %md
# MAGIC # Gold - Classificação NCM dos Produtos Unificados
# MAGIC
# MAGIC Notebook responsável por atribuir o(s) código(s) NCM mais provável(is) a cada produto unificado.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. Verificação de dependências via `importlib` (instala apenas o que faltar).
# MAGIC 2. Leitura de `databridge.raw_data.gold_produtos_unificados` (produtos unificados).
# MAGIC 3. Leitura de `databridge.raw_data.ncm_embeddings` (embeddings da NCM, gerados pelo notebook 00).
# MAGIC 4. Geração dos embeddings de `nome_padrao` com `sentence-transformers/all-MiniLM-L6-v2`.
# MAGIC 5. Cálculo da similaridade de cosseno entre cada produto e todos os NCMs.
# MAGIC 6. Seleção dos 3 NCMs com maior score (`top_ncm_1`, `top_ncm_2`, `top_ncm_3`) e seus scores.
# MAGIC 7. Coluna `ncm_confianca` = score do `top_ncm_1`.
# MAGIC 8. Persistência em `databridge.raw_data.gold_produtos_com_ncm`.
# MAGIC 9. Registro de auditoria em `databridge.raw_data.audit_logs` com `StructType` explícito.
# MAGIC
# MAGIC **Stack:** PySpark + Sentence-BERT + Delta Lake + Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verificação e instalação condicional de dependências
# MAGIC
# MAGIC Em vez de `%pip install` direto, usamos `importlib.util.find_spec` para detectar o que
# MAGIC realmente está faltando no cluster e só então invocamos o `pip` via `subprocess`.
# MAGIC Isso evita reinicializar o Python desnecessariamente quando as libs já existem no runtime.

# COMMAND ----------

import importlib.util
import subprocess
import sys

# Mapa: nome do módulo a importar -> spec de instalação no pip
PACOTES_NECESSARIOS = {
    "sentence_transformers": "sentence-transformers==2.2.2",
    "huggingface_hub": "huggingface-hub==0.16.4",
    "unidecode": "unidecode==1.3.8",
}

faltantes = [
    spec
    for modulo, spec in PACOTES_NECESSARIOS.items()
    if importlib.util.find_spec(modulo) is None
]

if faltantes:
    print(f"[DEPENDENCIAS] Instalando pacotes ausentes: {faltantes}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *faltantes]
    )
    print("[DEPENDENCIAS] Instalação concluída. Reiniciando Python.")
    dbutils.library.restartPython()
else:
    print("[DEPENDENCIAS] Todas as bibliotecas já estão disponíveis. Nada a instalar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports e configuração

# COMMAND ----------

import json
import time
import uuid
import traceback
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
CONFIG = {
    "catalogo": "databridge",
    "schema": "raw_data",
    "tabela_produtos": "gold_produtos_unificados",
    "tabela_embeddings_ncm": "ncm_embeddings",
    "tabela_ncm": "ncm_codigos",
    "tabela_destino": "gold_produtos_com_ncm",
    "tabela_auditoria": "audit_logs",
    "modelo_embeddings": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size_embeddings": 64,
    "top_k": 3,
    "nome_job": "gold_classificacao_ncm",
}

NOME_TABELA_PRODUTOS = (
    f"{CONFIG['catalogo']}.{CONFIG['schema']}.{CONFIG['tabela_produtos']}"
)
NOME_TABELA_EMB_NCM = (
    f"{CONFIG['catalogo']}.{CONFIG['schema']}.{CONFIG['tabela_embeddings_ncm']}"
)
NOME_TABELA_NCM = f"{CONFIG['catalogo']}.{CONFIG['schema']}.{CONFIG['tabela_ncm']}"
NOME_TABELA_DESTINO = (
    f"{CONFIG['catalogo']}.{CONFIG['schema']}.{CONFIG['tabela_destino']}"
)
NOME_TABELA_AUDITORIA = (
    f"{CONFIG['catalogo']}.{CONFIG['schema']}.{CONFIG['tabela_auditoria']}"
)


def log(etapa: str, mensagem: str, nivel: str = "INFO") -> None:
    """Log estruturado em JSON, uma linha por evento."""
    print(
        json.dumps(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "nivel": nivel,
                "etapa": etapa,
                "mensagem": mensagem,
            },
            ensure_ascii=False,
        )
    )


ESTADO_EXEC = {
    "execution_id": str(uuid.uuid4()),
    "inicio": datetime.utcnow(),
    "status": "EM_EXECUCAO",
    "mensagem_erro": None,
    "produtos_classificados": 0,
    "ncm_referencia": 0,
    "confianca_media": 0.0,
    "confianca_minima": 0.0,
    "confianca_maxima": 0.0,
}

log("INIT", f"Execution ID: {ESTADO_EXEC['execution_id']}.")
log("INIT", f"Origem produtos: {NOME_TABELA_PRODUTOS}.")
log("INIT", f"Origem NCM embeddings: {NOME_TABELA_EMB_NCM}.")
log("INIT", f"Destino: {NOME_TABELA_DESTINO}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Schema de auditoria
# MAGIC
# MAGIC Schema `StructType` explícito. A gravação usa `mergeSchema=true` para conviver com
# MAGIC registros gravados por outros notebooks no mesmo `audit_logs`.

# COMMAND ----------

SCHEMA_AUDITORIA = StructType(
    [
        StructField("execution_id", StringType(), nullable=False),
        StructField("nome_job", StringType(), nullable=False),
        StructField("inicio", TimestampType(), nullable=False),
        StructField("fim", TimestampType(), nullable=False),
        StructField("duracao_segundos", DoubleType(), nullable=False),
        StructField("status", StringType(), nullable=False),
        StructField("produtos_classificados", LongType(), nullable=False),
        StructField("ncm_referencia", LongType(), nullable=False),
        StructField("confianca_media", DoubleType(), nullable=False),
        StructField("confianca_minima", DoubleType(), nullable=False),
        StructField("confianca_maxima", DoubleType(), nullable=False),
        StructField("modelo_embeddings", StringType(), nullable=False),
        StructField("mensagem_erro", StringType(), nullable=True),
    ]
)


def gravar_auditoria(estado: Dict) -> None:
    fim = datetime.utcnow()
    duracao = (fim - estado["inicio"]).total_seconds()

    linha = (
        estado["execution_id"],
        CONFIG["nome_job"],
        estado["inicio"],
        fim,
        float(duracao),
        estado["status"],
        int(estado["produtos_classificados"]),
        int(estado["ncm_referencia"]),
        float(estado["confianca_media"]),
        float(estado["confianca_minima"]),
        float(estado["confianca_maxima"]),
        CONFIG["modelo_embeddings"],
        estado["mensagem_erro"],
    )

    sdf = spark.createDataFrame([linha], schema=SCHEMA_AUDITORIA)
    (
        sdf.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(NOME_TABELA_AUDITORIA)
    )
    log(
        "AUDITORIA",
        f"Auditoria gravada em {NOME_TABELA_AUDITORIA} "
        f"(status={estado['status']}, duração={duracao:.1f}s).",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Leitura dos produtos unificados (Gold)

# COMMAND ----------

try:
    log("LEITURA_PRODUTOS", f"Lendo tabela {NOME_TABELA_PRODUTOS}.")
    df_produtos = spark.table(NOME_TABELA_PRODUTOS)

    if "nome_padrao" not in df_produtos.columns:
        raise RuntimeError(
            f"Tabela {NOME_TABELA_PRODUTOS} não possui a coluna obrigatória 'nome_padrao'."
        )

    # Trabalhamos com nomes únicos para evitar reprocessar embeddings repetidos.
    nomes_unicos = (
        df_produtos.select(F.trim(F.col("nome_padrao")).alias("nome_padrao"))
        .filter(F.length("nome_padrao") > 0)
        .distinct()
        .toPandas()
    )

    if nomes_unicos.empty:
        raise RuntimeError(
            f"Nenhum 'nome_padrao' válido em {NOME_TABELA_PRODUTOS}."
        )

    log(
        "LEITURA_PRODUTOS",
        f"Nomes padrão únicos a classificar: {len(nomes_unicos)}.",
    )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha ao ler produtos: {exc}"
    log("LEITURA_PRODUTOS", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Leitura dos embeddings NCM

# COMMAND ----------

try:
    log("LEITURA_NCM", f"Lendo tabela {NOME_TABELA_EMB_NCM}.")
    df_ncm_emb = spark.table(NOME_TABELA_EMB_NCM)

    colunas_ncm_obrig = {"codigo_ncm", "embedding"}
    faltantes_ncm = colunas_ncm_obrig - set(df_ncm_emb.columns)
    if faltantes_ncm:
        raise RuntimeError(
            f"Tabela {NOME_TABELA_EMB_NCM} sem colunas obrigatórias: {faltantes_ncm}."
        )

    pdf_ncm = df_ncm_emb.select("codigo_ncm", "embedding").toPandas()
    if pdf_ncm.empty:
        raise RuntimeError(f"Tabela {NOME_TABELA_EMB_NCM} está vazia.")

    matriz_ncm = np.asarray(
        [np.asarray(v, dtype=np.float32) for v in pdf_ncm["embedding"].tolist()],
        dtype=np.float32,
    )
    codigos_ncm = pdf_ncm["codigo_ncm"].astype(str).tolist()

    # Garantia de normalização L2 (defensivo, mesmo já vindo normalizado do notebook 00).
    normas = np.linalg.norm(matriz_ncm, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    matriz_ncm = matriz_ncm / normas

    ESTADO_EXEC["ncm_referencia"] = int(matriz_ncm.shape[0])
    log(
        "LEITURA_NCM",
        f"Matriz NCM carregada. Shape: {matriz_ncm.shape}. "
        f"Códigos: {ESTADO_EXEC['ncm_referencia']}.",
    )

    # Descrições NCM são opcionais — apenas para enriquecer a saída.
    descricoes_ncm: Dict[str, str] = {}
    try:
        df_ncm_codigos = spark.table(NOME_TABELA_NCM).select(
            "codigo_ncm", "descricao_ncm"
        )
        descricoes_ncm = {
            r["codigo_ncm"]: r["descricao_ncm"]
            for r in df_ncm_codigos.collect()
        }
        log(
            "LEITURA_NCM",
            f"Descrições NCM carregadas: {len(descricoes_ncm)}.",
        )
    except Exception as exc_desc:  # noqa: BLE001
        log(
            "LEITURA_NCM",
            f"Tabela {NOME_TABELA_NCM} indisponível ({exc_desc}). "
            "Descrições não serão anexadas à saída.",
            nivel="WARN",
        )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha ao ler embeddings NCM: {exc}"
    log("LEITURA_NCM", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Geração dos embeddings dos nomes_padrao

# COMMAND ----------

try:
    from sentence_transformers import SentenceTransformer

    log("EMBEDDINGS", f"Carregando modelo {CONFIG['modelo_embeddings']}.")
    inicio = time.time()
    modelo = SentenceTransformer(CONFIG["modelo_embeddings"])
    log("EMBEDDINGS", f"Modelo carregado em {time.time() - inicio:.1f}s.")

    nomes = nomes_unicos["nome_padrao"].astype(str).tolist()
    log(
        "EMBEDDINGS",
        f"Codificando {len(nomes)} nomes (batch={CONFIG['batch_size_embeddings']}).",
    )
    inicio = time.time()
    matriz_prod = modelo.encode(
        nomes,
        batch_size=CONFIG["batch_size_embeddings"],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    log(
        "EMBEDDINGS",
        f"Embeddings gerados em {time.time() - inicio:.1f}s. Shape: {matriz_prod.shape}.",
    )

    if matriz_prod.shape[1] != matriz_ncm.shape[1]:
        raise RuntimeError(
            f"Dimensão de embedding incompatível: produtos={matriz_prod.shape[1]} "
            f"vs NCM={matriz_ncm.shape[1]}. "
            "Reprocesse o notebook 00 com o mesmo modelo."
        )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha em embeddings: {exc}"
    log("EMBEDDINGS", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cálculo de similaridade e Top-K NCMs
# MAGIC
# MAGIC Como ambas as matrizes estão L2-normalizadas, o cosseno é o produto interno.
# MAGIC O cálculo é vetorizado em NumPy. Para evitar pico de memória em bases grandes,
# MAGIC processamos em blocos de produtos.

# COMMAND ----------

try:
    top_k = CONFIG["top_k"]
    n_prod = matriz_prod.shape[0]
    bloco = 1024  # produtos por bloco

    top_idx = np.empty((n_prod, top_k), dtype=np.int32)
    top_score = np.empty((n_prod, top_k), dtype=np.float32)

    log(
        "SIMILARIDADE",
        f"Calculando Top-{top_k} NCMs para {n_prod} produtos x "
        f"{matriz_ncm.shape[0]} NCMs (blocos de {bloco}).",
    )
    inicio = time.time()

    for ini in range(0, n_prod, bloco):
        fim_b = min(ini + bloco, n_prod)
        sim = matriz_prod[ini:fim_b] @ matriz_ncm.T  # (bloco, n_ncm)

        # argpartition é O(n) e suficiente para extrair top-k
        idx_part = np.argpartition(-sim, kth=top_k - 1, axis=1)[:, :top_k]

        # Ordenamos os top-k por score decrescente
        for r in range(idx_part.shape[0]):
            cols = idx_part[r]
            scores = sim[r, cols]
            ordem = np.argsort(-scores)
            top_idx[ini + r] = cols[ordem]
            top_score[ini + r] = scores[ordem]

    log("SIMILARIDADE", f"Top-K calculado em {time.time() - inicio:.1f}s.")

    # Estatísticas de confiança (score do top 1)
    confianca = top_score[:, 0].astype(float)
    ESTADO_EXEC["produtos_classificados"] = int(n_prod)
    ESTADO_EXEC["confianca_media"] = float(np.mean(confianca))
    ESTADO_EXEC["confianca_minima"] = float(np.min(confianca))
    ESTADO_EXEC["confianca_maxima"] = float(np.max(confianca))
    log(
        "SIMILARIDADE",
        f"Confiança (top_ncm_1): média={ESTADO_EXEC['confianca_media']:.3f}, "
        f"min={ESTADO_EXEC['confianca_minima']:.3f}, "
        f"max={ESTADO_EXEC['confianca_maxima']:.3f}.",
    )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha em similaridade: {exc}"
    log("SIMILARIDADE", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Montagem do DataFrame de classificação

# COMMAND ----------

try:
    registros_classif: List[Dict] = []
    for i, nome in enumerate(nomes):
        codigos = [codigos_ncm[int(j)] for j in top_idx[i]]
        scores = [float(s) for s in top_score[i]]
        registros_classif.append(
            {
                "nome_padrao": nome,
                "top_ncm_1": codigos[0],
                "score_ncm_1": scores[0],
                "descricao_ncm_1": descricoes_ncm.get(codigos[0]),
                "top_ncm_2": codigos[1] if top_k > 1 else None,
                "score_ncm_2": scores[1] if top_k > 1 else None,
                "descricao_ncm_2": descricoes_ncm.get(codigos[1]) if top_k > 1 else None,
                "top_ncm_3": codigos[2] if top_k > 2 else None,
                "score_ncm_3": scores[2] if top_k > 2 else None,
                "descricao_ncm_3": descricoes_ncm.get(codigos[2]) if top_k > 2 else None,
                "ncm_confianca": scores[0],
            }
        )

    schema_classif = StructType(
        [
            StructField("nome_padrao", StringType(), nullable=False),
            StructField("top_ncm_1", StringType(), nullable=False),
            StructField("score_ncm_1", DoubleType(), nullable=False),
            StructField("descricao_ncm_1", StringType(), nullable=True),
            StructField("top_ncm_2", StringType(), nullable=True),
            StructField("score_ncm_2", DoubleType(), nullable=True),
            StructField("descricao_ncm_2", StringType(), nullable=True),
            StructField("top_ncm_3", StringType(), nullable=True),
            StructField("score_ncm_3", DoubleType(), nullable=True),
            StructField("descricao_ncm_3", StringType(), nullable=True),
            StructField("ncm_confianca", DoubleType(), nullable=False),
        ]
    )

    df_classif = spark.createDataFrame(
        pd.DataFrame(registros_classif), schema=schema_classif
    )
    log(
        "CLASSIFICACAO",
        f"DataFrame de classificação montado: {df_classif.count()} linhas.",
    )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha na montagem da classificação: {exc}"
    log("CLASSIFICACAO", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Persistência em `gold_produtos_com_ncm`
# MAGIC
# MAGIC Junta a classificação de volta a `gold_produtos_unificados` por `nome_padrao` para que
# MAGIC todas as linhas de origem (várias por grupo) recebam a mesma classificação NCM.

# COMMAND ----------

try:
    df_final = (
        df_produtos.join(df_classif, on="nome_padrao", how="left")
        .withColumn("data_classificacao", F.current_timestamp())
    )

    qtd_final = df_final.count()
    log("PERSISTENCIA", f"Gravando {qtd_final} linhas em {NOME_TABELA_DESTINO}.")
    (
        df_final.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(NOME_TABELA_DESTINO)
    )
    log("PERSISTENCIA", f"Tabela {NOME_TABELA_DESTINO} gravada com sucesso.")
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha na persistência: {exc}"
    log("PERSISTENCIA", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Auditoria e fechamento

# COMMAND ----------

try:
    ESTADO_EXEC["status"] = "SUCESSO"
    gravar_auditoria(ESTADO_EXEC)
    log(
        "FIM",
        f"Execução concluída. Produtos classificados="
        f"{ESTADO_EXEC['produtos_classificados']}, "
        f"NCM referência={ESTADO_EXEC['ncm_referencia']}, "
        f"confiança média={ESTADO_EXEC['confianca_media']:.3f}.",
    )
except Exception as exc:  # noqa: BLE001
    log("AUDITORIA", f"Falha ao registrar auditoria final: {exc}.", nivel="ERROR")
    log("AUDITORIA", traceback.format_exc(), nivel="ERROR")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Amostras

# COMMAND ----------

display(
    spark.table(NOME_TABELA_DESTINO)
    .select(
        "nome_padrao",
        "top_ncm_1",
        "score_ncm_1",
        "descricao_ncm_1",
        "top_ncm_2",
        "score_ncm_2",
        "top_ncm_3",
        "score_ncm_3",
        "ncm_confianca",
    )
    .orderBy(F.col("ncm_confianca").desc())
    .limit(20)
)

# COMMAND ----------

display(
    spark.table(NOME_TABELA_AUDITORIA)
    .filter(F.col("nome_job") == CONFIG["nome_job"])
    .orderBy(F.col("inicio").desc())
    .limit(5)
)
