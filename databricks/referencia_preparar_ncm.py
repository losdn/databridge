# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Preparação da Base NCM
# MAGIC
# MAGIC Notebook responsável por preparar a base oficial da **Nomenclatura Comum do Mercosul (NCM)** no Databricks.
# MAGIC
# MAGIC **Etapas executadas:**
# MAGIC 1. Download da base oficial NCM (com fallback para CSV local em `/dbfs/FileStore/databridge/ncm_raw.csv`).
# MAGIC 2. Limpeza dos dados: remoção de duplicatas, normalização de texto (minúsculas, sem acentos) e tratamento de nulos.
# MAGIC 3. Persistência na tabela Delta `ncm_codigos` (`codigo_ncm`, `descricao_ncm`, `capitulo_ncm`, `grupo_ncm`).
# MAGIC 4. Geração de embeddings semânticos com `sentence-transformers/all-MiniLM-L6-v2`.
# MAGIC 5. Persistência dos embeddings na tabela Delta `ncm_embeddings` (`codigo_ncm`, `embedding`).
# MAGIC
# MAGIC **Stack:** PySpark + sentence-transformers + Delta Lake. Compatível com Databricks Free Tier.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Instalação de dependências
# MAGIC
# MAGIC O `sentence-transformers` não vem pré-instalado no Databricks Runtime padrão. Esta célula garante a presença das bibliotecas necessárias.

# COMMAND ----------

# MAGIC %pip install --quiet sentence-transformers==2.2.2 huggingface-hub==0.16.4 unidecode==1.3.8

# COMMAND ----------

# MAGIC %md
# MAGIC Reinicia o Python para que o cluster reconheça os pacotes recém-instalados.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports e configuração global

# COMMAND ----------

import io
import os
import re
import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    FloatType,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
CONFIG = {
    # URLs oficiais (tentativas em ordem). A primeira que retornar 200 será usada.
    "urls_oficiais": [
        "https://portalunico.siscomex.gov.br/classif/api/publico/nomenclatura/download/json",
        "https://www.gov.br/siscomex/pt-br/classif/nomenclatura-comum-do-mercosul/Tabela_NCM.json",
    ],
    # Fallback local (DBFS)
    "csv_local_dbfs": "/dbfs/FileStore/databridge/ncm_raw.csv",
    "csv_local_spark": "dbfs:/FileStore/databridge/ncm_raw.csv",
    # Tabelas Delta de saída
    "tabela_ncm": "ncm_codigos",
    "tabela_embeddings": "ncm_embeddings",
    # Modelo de embeddings
    "modelo_embeddings": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size_embeddings": 64,
    # Timeout do download em segundos
    "timeout_download": 60,
}


def log(etapa: str, mensagem: str, nivel: str = "INFO") -> None:
    """Log estruturado em formato JSON para fácil rastreio."""
    registro = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "nivel": nivel,
        "etapa": etapa,
        "mensagem": mensagem,
    }
    print(json.dumps(registro, ensure_ascii=False))


log("INIT", "Notebook 00_preparar_ncm iniciado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download da base oficial (com fallback local)

# COMMAND ----------

def _baixar_url(url: str, timeout: int) -> Optional[bytes]:
    """Baixa o conteúdo de uma URL. Retorna `None` em caso de falha."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "databridge/1.0 (+https://github.com/databridge)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                log(
                    "DOWNLOAD",
                    f"Status HTTP inesperado ({resp.status}) para {url}.",
                    nivel="WARN",
                )
                return None
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        log("DOWNLOAD", f"Falha ao baixar {url}: {exc}.", nivel="WARN")
        return None
    except Exception as exc:  # noqa: BLE001
        log("DOWNLOAD", f"Erro inesperado em {url}: {exc}.", nivel="WARN")
        return None


def _parse_json_oficial(conteudo: bytes) -> Optional[pd.DataFrame]:
    """Converte o JSON oficial do Siscomex em DataFrame pandas."""
    try:
        dados = json.loads(conteudo.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        log("DOWNLOAD", f"JSON oficial inválido: {exc}.", nivel="WARN")
        return None

    # Estrutura esperada: {"Data_Atualizacao": "...", "Nomenclaturas": [{...}, ...]}
    registros = (
        dados.get("Nomenclaturas")
        or dados.get("nomenclaturas")
        or dados.get("data")
        or []
    )
    if not registros:
        log("DOWNLOAD", "JSON oficial sem registros utilizáveis.", nivel="WARN")
        return None

    df = pd.DataFrame(registros)
    # Colunas comuns na fonte oficial
    mapa_colunas = {
        "Codigo": "codigo_ncm",
        "codigo": "codigo_ncm",
        "Descricao": "descricao_ncm",
        "descricao": "descricao_ncm",
    }
    df = df.rename(columns={k: v for k, v in mapa_colunas.items() if k in df.columns})

    if "codigo_ncm" not in df.columns or "descricao_ncm" not in df.columns:
        log(
            "DOWNLOAD",
            f"Colunas ausentes no JSON oficial. Encontradas: {list(df.columns)}.",
            nivel="WARN",
        )
        return None

    return df[["codigo_ncm", "descricao_ncm"]]


def carregar_ncm_bruto() -> pd.DataFrame:
    """
    Tenta baixar a base oficial NCM. Em caso de falha, usa o CSV local.
    Lança RuntimeError com mensagem em português se nenhuma fonte estiver disponível.
    """
    # 1) Tentativa via URLs oficiais
    for url in CONFIG["urls_oficiais"]:
        log("DOWNLOAD", f"Tentando baixar base oficial em {url}.")
        conteudo = _baixar_url(url, timeout=CONFIG["timeout_download"])
        if not conteudo:
            continue

        df_oficial = _parse_json_oficial(conteudo)
        if df_oficial is not None and not df_oficial.empty:
            log(
                "DOWNLOAD",
                f"Base oficial carregada com sucesso. Registros: {len(df_oficial)}.",
            )
            return df_oficial

    # 2) Fallback para CSV local em DBFS
    caminho_local = CONFIG["csv_local_dbfs"]
    log(
        "DOWNLOAD",
        f"Falha nas fontes remotas. Tentando CSV local em {caminho_local}.",
        nivel="WARN",
    )

    if not os.path.exists(caminho_local):
        raise RuntimeError(
            "Não foi possível obter a base NCM. As URLs oficiais falharam e o "
            f"arquivo local '{caminho_local}' não existe. "
            "Faça upload do CSV em FileStore/databridge/ncm_raw.csv com as "
            "colunas 'codigo_ncm' e 'descricao_ncm' antes de reexecutar."
        )

    try:
        df_local = pd.read_csv(caminho_local, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        df_local = pd.read_csv(caminho_local, dtype=str, encoding="latin-1")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Falha ao ler o CSV local '{caminho_local}': {exc}."
        ) from exc

    df_local.columns = [c.strip().lower() for c in df_local.columns]

    # Tenta mapear colunas alternativas comumente usadas
    aliases = {
        "codigo_ncm": ["codigo_ncm", "codigo", "ncm", "cod_ncm"],
        "descricao_ncm": ["descricao_ncm", "descricao", "descrição", "desc_ncm"],
    }
    for destino, fontes in aliases.items():
        if destino not in df_local.columns:
            for fonte in fontes:
                if fonte in df_local.columns:
                    df_local = df_local.rename(columns={fonte: destino})
                    break

    if "codigo_ncm" not in df_local.columns or "descricao_ncm" not in df_local.columns:
        raise RuntimeError(
            "CSV local não contém as colunas obrigatórias 'codigo_ncm' e "
            f"'descricao_ncm'. Colunas encontradas: {list(df_local.columns)}."
        )

    df_local = df_local[["codigo_ncm", "descricao_ncm"]]
    log("DOWNLOAD", f"CSV local carregado. Registros: {len(df_local)}.")
    return df_local


try:
    df_bruto_pd = carregar_ncm_bruto()
    log("DOWNLOAD", f"Total de registros brutos: {len(df_bruto_pd)}.")
except RuntimeError as exc:
    log("DOWNLOAD", str(exc), nivel="ERROR")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Limpeza e normalização
# MAGIC
# MAGIC - Remove acentos e converte para minúsculas.
# MAGIC - Remove caracteres não imprimíveis e espaços excedentes.
# MAGIC - Trata nulos e descrições vazias.
# MAGIC - Mantém apenas códigos NCM com 8 dígitos (padrão oficial), tolerando entradas com pontos/traços.
# MAGIC - Deriva `capitulo_ncm` (2 primeiros dígitos) e `grupo_ncm` (4 primeiros dígitos).
# MAGIC - Remove duplicatas pelo `codigo_ncm`.

# COMMAND ----------

from unidecode import unidecode  # noqa: E402  (precisa do restartPython acima)


def _normalizar_codigo(valor: object) -> Optional[str]:
    """Mantém apenas dígitos. Considera válido apenas códigos com 8 dígitos."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    digitos = re.sub(r"\D", "", str(valor))
    if len(digitos) == 8:
        return digitos
    # Aceita também códigos hierárquicos (capítulo/posição) preenchendo à direita
    # apenas quando explicitamente desejado. Aqui descartamos os demais.
    return None


def _normalizar_texto(valor: object) -> Optional[str]:
    """Lowercase, sem acentos, espaços normalizados."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    texto = unidecode(str(valor)).lower()
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto or None


def limpar_ncm(df: pd.DataFrame) -> pd.DataFrame:
    total_inicial = len(df)
    log("LIMPEZA", f"Início da limpeza. Registros iniciais: {total_inicial}.")

    df = df.copy()
    df["codigo_ncm"] = df["codigo_ncm"].map(_normalizar_codigo)
    df["descricao_ncm"] = df["descricao_ncm"].map(_normalizar_texto)

    antes_nulos = len(df)
    df = df.dropna(subset=["codigo_ncm", "descricao_ncm"])
    log(
        "LIMPEZA",
        f"Removidos {antes_nulos - len(df)} registros com código/descrição inválidos.",
    )

    antes_duplicatas = len(df)
    df = df.drop_duplicates(subset=["codigo_ncm"], keep="first")
    log(
        "LIMPEZA",
        f"Removidas {antes_duplicatas - len(df)} duplicatas por codigo_ncm.",
    )

    df["capitulo_ncm"] = df["codigo_ncm"].str[:2]
    df["grupo_ncm"] = df["codigo_ncm"].str[:4]

    df = df[["codigo_ncm", "descricao_ncm", "capitulo_ncm", "grupo_ncm"]].reset_index(
        drop=True
    )
    log("LIMPEZA", f"Registros finais após limpeza: {len(df)}.")
    return df


try:
    df_ncm_pd = limpar_ncm(df_bruto_pd)
    if df_ncm_pd.empty:
        raise RuntimeError(
            "Nenhum registro válido restou após a limpeza. Verifique a base de origem."
        )
except Exception as exc:  # noqa: BLE001
    log("LIMPEZA", f"Falha na etapa de limpeza: {exc}.", nivel="ERROR")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persistência da tabela Delta `ncm_codigos`

# COMMAND ----------

def salvar_tabela_ncm(df_pd: pd.DataFrame, nome_tabela: str) -> int:
    """Converte para Spark DataFrame e grava como tabela Delta gerenciada."""
    schema = StructType(
        [
            StructField("codigo_ncm", StringType(), nullable=False),
            StructField("descricao_ncm", StringType(), nullable=False),
            StructField("capitulo_ncm", StringType(), nullable=False),
            StructField("grupo_ncm", StringType(), nullable=False),
        ]
    )

    sdf = spark.createDataFrame(df_pd, schema=schema)
    qtd = sdf.count()

    (
        sdf.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(nome_tabela)
    )
    log("PERSISTENCIA", f"Tabela Delta '{nome_tabela}' gravada com {qtd} registros.")
    return qtd


try:
    qtd_ncm = salvar_tabela_ncm(df_ncm_pd, CONFIG["tabela_ncm"])
except Exception as exc:  # noqa: BLE001
    log(
        "PERSISTENCIA",
        f"Falha ao salvar tabela '{CONFIG['tabela_ncm']}': {exc}.",
        nivel="ERROR",
    )
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Geração de embeddings com Sentence-BERT
# MAGIC
# MAGIC Modelo: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensões, leve, compatível com Free Tier).
# MAGIC
# MAGIC O encoding é executado em lote no driver para evitar overhead de serialização do modelo
# MAGIC em UDFs distribuídas. Para a base NCM (~10k linhas) isso roda em poucos minutos.

# COMMAND ----------

def gerar_embeddings(descricoes: List[str], modelo_nome: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # import tardio

    log("EMBEDDINGS", f"Carregando modelo '{modelo_nome}'.")
    inicio = time.time()
    modelo = SentenceTransformer(modelo_nome)
    log("EMBEDDINGS", f"Modelo carregado em {time.time() - inicio:.1f}s.")

    log(
        "EMBEDDINGS",
        f"Gerando embeddings para {len(descricoes)} descrições (batch={batch_size}).",
    )
    inicio = time.time()
    vetores = modelo.encode(
        descricoes,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(
        "EMBEDDINGS",
        f"Embeddings gerados em {time.time() - inicio:.1f}s. "
        f"Shape: {vetores.shape}.",
    )
    return vetores.astype(np.float32)


try:
    descricoes = df_ncm_pd["descricao_ncm"].tolist()
    matriz_embeddings = gerar_embeddings(
        descricoes,
        modelo_nome=CONFIG["modelo_embeddings"],
        batch_size=CONFIG["batch_size_embeddings"],
    )
except Exception as exc:  # noqa: BLE001
    log("EMBEDDINGS", f"Falha ao gerar embeddings: {exc}.", nivel="ERROR")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Persistência da tabela Delta `ncm_embeddings`

# COMMAND ----------

def salvar_tabela_embeddings(
    codigos: List[str], vetores: np.ndarray, nome_tabela: str
) -> int:
    schema = StructType(
        [
            StructField("codigo_ncm", StringType(), nullable=False),
            StructField("embedding", ArrayType(FloatType()), nullable=False),
        ]
    )

    linhas = [
        (codigo, [float(x) for x in vetor])
        for codigo, vetor in zip(codigos, vetores)
    ]
    sdf = spark.createDataFrame(linhas, schema=schema)
    qtd = sdf.count()

    (
        sdf.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(nome_tabela)
    )
    log(
        "PERSISTENCIA",
        f"Tabela Delta '{nome_tabela}' gravada com {qtd} registros "
        f"(dim={vetores.shape[1] if vetores.ndim == 2 else 'N/A'}).",
    )
    return qtd


try:
    qtd_emb = salvar_tabela_embeddings(
        df_ncm_pd["codigo_ncm"].tolist(),
        matriz_embeddings,
        CONFIG["tabela_embeddings"],
    )
except Exception as exc:  # noqa: BLE001
    log(
        "PERSISTENCIA",
        f"Falha ao salvar tabela '{CONFIG['tabela_embeddings']}': {exc}.",
        nivel="ERROR",
    )
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validação final

# COMMAND ----------

try:
    qtd_ncm_check = spark.table(CONFIG["tabela_ncm"]).count()
    qtd_emb_check = spark.table(CONFIG["tabela_embeddings"]).count()

    if qtd_ncm_check != qtd_emb_check:
        log(
            "VALIDACAO",
            f"Inconsistência: ncm_codigos={qtd_ncm_check} x "
            f"ncm_embeddings={qtd_emb_check}.",
            nivel="WARN",
        )
    else:
        log(
            "VALIDACAO",
            f"OK: {qtd_ncm_check} registros em ambas as tabelas Delta.",
        )

    log(
        "FIM",
        "Notebook 00_preparar_ncm concluído com sucesso. "
        f"Tabelas: {CONFIG['tabela_ncm']}, {CONFIG['tabela_embeddings']}.",
    )
except Exception as exc:  # noqa: BLE001
    log("VALIDACAO", f"Falha na validação final: {exc}.", nivel="ERROR")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Amostra das tabelas geradas

# COMMAND ----------

display(spark.table(CONFIG["tabela_ncm"]).limit(10))

# COMMAND ----------

display(spark.table(CONFIG["tabela_embeddings"]).limit(5))