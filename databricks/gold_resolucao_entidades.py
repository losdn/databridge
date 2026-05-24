# Databricks notebook source
# MAGIC %md
# MAGIC # Gold - Resolução de Entidades de Produtos
# MAGIC
# MAGIC Notebook responsável por unificar produtos equivalentes vindos de múltiplos sistemas-fonte.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. Leitura de `databridge.raw_data.silver_classificado`.
# MAGIC 2. Geração de embeddings semânticos com `sentence-transformers/all-MiniLM-L6-v2` sobre `nome_produto`.
# MAGIC 3. Similaridade de cosseno entre todos os pares.
# MAGIC 4. Similaridade textual Jaro-Winkler via `jellyfish`.
# MAGIC 5. Score combinado = `0.70 * cosseno + 0.30 * jaro_winkler`.
# MAGIC 6. Agrupamento por união de pares com `score >= 0.88` (Union-Find).
# MAGIC 7. Para cada grupo: `id_produto_unico` (UUID), `nome_padrao` (nome mais longo) e `justificativa` textual.
# MAGIC 8. Persistência em `databridge.raw_data.gold_produtos_unificados`.
# MAGIC 9. Registro de execução em `databridge.raw_data.audit_logs` com `StructType` explícito.
# MAGIC
# MAGIC **Stack:** PySpark + Sentence-BERT + Jellyfish + Delta Lake + Unity Catalog.

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

instalar_se_necessario("sentence-transformers", "sentence_transformers")
instalar_se_necessario("jellyfish")
instalar_se_necessario("unidecode")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports e configuração

# COMMAND ----------

import json
import time
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
    DoubleType,
    TimestampType,
    ArrayType,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
CONFIG = {
    "catalogo": "databridge",
    "schema": "raw_data",
    "tabela_origem": "silver_classificado",
    "tabela_destino": "gold_produtos_unificados",
    "tabela_auditoria": "audit_logs",
    # Modelo de embeddings
    "modelo_embeddings": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size_embeddings": 64,
    # Pesos do score combinado
    "peso_cosseno": 0.70,
    "peso_jaro_winkler": 0.30,
    # Limiar de agrupamento
    "limiar_match": 0.88,
    # Limite duro para impedir explosão combinatória (N*(N-1)/2)
    "max_pares_brutos": 5_000_000,
    # Nome do job para auditoria
    "nome_job": "gold_resolucao_entidades",
}

NOME_TABELA_ORIGEM = f"{CONFIG['catalogo']}.{CONFIG['schema']}.{CONFIG['tabela_origem']}"
NOME_TABELA_DESTINO = f"{CONFIG['catalogo']}.{CONFIG['schema']}.{CONFIG['tabela_destino']}"
NOME_TABELA_AUDITORIA = f"{CONFIG['catalogo']}.{CONFIG['schema']}.{CONFIG['tabela_auditoria']}"


def log(etapa: str, mensagem: str, nivel: str = "INFO") -> None:
    """Log estruturado em JSON (uma linha por evento)."""
    registro = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "nivel": nivel,
        "etapa": etapa,
        "mensagem": mensagem,
    }
    print(json.dumps(registro, ensure_ascii=False))


# Estado da execução (preenchido ao longo do notebook e gravado na auditoria no fim).
ESTADO_EXEC = {
    "execution_id": str(uuid.uuid4()),
    "inicio": datetime.utcnow(),
    "status": "EM_EXECUCAO",
    "mensagem_erro": None,
    "registros_lidos": 0,
    "pares_avaliados": 0,
    "pares_acima_limiar": 0,
    "grupos_formados": 0,
}

log("INIT", f"Execution ID: {ESTADO_EXEC['execution_id']}.")
log("INIT", f"Tabela origem: {NOME_TABELA_ORIGEM}.")
log("INIT", f"Tabela destino: {NOME_TABELA_DESTINO}.")
log("INIT", f"Tabela auditoria: {NOME_TABELA_AUDITORIA}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Schema da auditoria
# MAGIC
# MAGIC Definido com `StructType` explícito para garantir compatibilidade entre execuções e
# MAGIC permitir gravação mesmo quando a tabela ainda não existe no Unity Catalog.

# COMMAND ----------

SCHEMA_AUDITORIA = StructType(
    [
        StructField("execution_id", StringType(), nullable=False),
        StructField("nome_job", StringType(), nullable=False),
        StructField("inicio", TimestampType(), nullable=False),
        StructField("fim", TimestampType(), nullable=False),
        StructField("duracao_segundos", DoubleType(), nullable=False),
        StructField("status", StringType(), nullable=False),
        StructField("registros_lidos", LongType(), nullable=False),
        StructField("pares_avaliados", LongType(), nullable=False),
        StructField("pares_acima_limiar", LongType(), nullable=False),
        StructField("grupos_formados", LongType(), nullable=False),
        StructField("limiar_match", DoubleType(), nullable=False),
        StructField("peso_cosseno", DoubleType(), nullable=False),
        StructField("peso_jaro_winkler", DoubleType(), nullable=False),
        StructField("modelo_embeddings", StringType(), nullable=False),
        StructField("mensagem_erro", StringType(), nullable=True),
    ]
)


def gravar_auditoria(estado: Dict) -> None:
    """Grava um registro na tabela de auditoria, criando-a se necessário."""
    fim = datetime.utcnow()
    duracao = (fim - estado["inicio"]).total_seconds()

    linha = (
        estado["execution_id"],
        CONFIG["nome_job"],
        estado["inicio"],
        fim,
        float(duracao),
        estado["status"],
        int(estado["registros_lidos"]),
        int(estado["pares_avaliados"]),
        int(estado["pares_acima_limiar"]),
        int(estado["grupos_formados"]),
        float(CONFIG["limiar_match"]),
        float(CONFIG["peso_cosseno"]),
        float(CONFIG["peso_jaro_winkler"]),
        CONFIG["modelo_embeddings"],
        estado["mensagem_erro"],
    )

    sdf = spark.createDataFrame([linha], schema=SCHEMA_AUDITORIA)
    (
        sdf.write.format("delta")
        .mode("append")
        .option("mergeSchema", "false")
        .saveAsTable(NOME_TABELA_AUDITORIA)
    )
    log(
        "AUDITORIA",
        f"Registro de auditoria gravado em {NOME_TABELA_AUDITORIA} "
        f"(status={estado['status']}, duração={duracao:.1f}s).",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Leitura da camada Silver

# COMMAND ----------

try:
    log("LEITURA", f"Lendo tabela {NOME_TABELA_ORIGEM}.")
    df_silver = spark.table(NOME_TABELA_ORIGEM)

    colunas_obrigatorias = {"nome_produto"}
    faltantes = colunas_obrigatorias - set(df_silver.columns)
    if faltantes:
        raise RuntimeError(
            f"Tabela {NOME_TABELA_ORIGEM} não possui colunas obrigatórias: {faltantes}."
        )

    # Mantemos um identificador estável por linha. Se a tabela já tiver um id natural
    # (id_produto, sku etc.) ele será preservado em colunas_origem.
    colunas_origem = [c for c in df_silver.columns if c != "nome_produto"]

    df_silver = df_silver.withColumn(
        "_row_id", F.expr("uuid()")
    ).withColumn(
        "nome_produto", F.coalesce(F.trim(F.col("nome_produto")), F.lit(""))
    )

    # Descarta linhas sem nome de produto utilizável.
    df_silver = df_silver.filter(F.length("nome_produto") > 0)

    pdf = df_silver.toPandas()
    ESTADO_EXEC["registros_lidos"] = int(len(pdf))
    log("LEITURA", f"Registros válidos lidos: {ESTADO_EXEC['registros_lidos']}.")

    if pdf.empty:
        raise RuntimeError(
            "Nenhum registro válido em silver_classificado (nome_produto vazio em todas as linhas)."
        )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha na leitura: {exc}"
    log("LEITURA", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Geração de embeddings (Sentence-BERT)

# COMMAND ----------

try:
    from sentence_transformers import SentenceTransformer

    log("EMBEDDINGS", f"Carregando modelo {CONFIG['modelo_embeddings']}.")
    inicio = time.time()
    modelo = SentenceTransformer(CONFIG["modelo_embeddings"])
    log("EMBEDDINGS", f"Modelo carregado em {time.time() - inicio:.1f}s.")

    nomes = pdf["nome_produto"].astype(str).tolist()
    log("EMBEDDINGS", f"Codificando {len(nomes)} nomes (batch={CONFIG['batch_size_embeddings']}).")
    inicio = time.time()
    matriz = modelo.encode(
        nomes,
        batch_size=CONFIG["batch_size_embeddings"],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # facilita cálculo de cosseno via produto interno
    ).astype(np.float32)
    log(
        "EMBEDDINGS",
        f"Embeddings gerados em {time.time() - inicio:.1f}s. Shape: {matriz.shape}.",
    )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha em embeddings: {exc}"
    log("EMBEDDINGS", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Similaridade de cosseno entre todos os pares
# MAGIC
# MAGIC Como os embeddings já estão L2-normalizados, a similaridade de cosseno é o produto interno.
# MAGIC O cálculo é feito com NumPy no driver para evitar overhead do Spark em uma matriz N×N.
# MAGIC Um teto duro de pares (`max_pares_brutos`) protege contra estouro de memória.

# COMMAND ----------

try:
    n = matriz.shape[0]
    pares_potenciais = n * (n - 1) // 2
    log("COSSENO", f"Total de pares potenciais: {pares_potenciais}.")

    if pares_potenciais > CONFIG["max_pares_brutos"]:
        raise RuntimeError(
            f"Cálculo abortado: {pares_potenciais} pares excedem o limite de "
            f"{CONFIG['max_pares_brutos']}. Aplique blocagem (por categoria/marca) "
            "antes da resolução de entidades."
        )

    log("COSSENO", "Calculando matriz de similaridade NxN.")
    inicio = time.time()
    sim_cosseno = matriz @ matriz.T  # (N, N), valores em [-1, 1] para vetores normalizados
    log("COSSENO", f"Matriz calculada em {time.time() - inicio:.1f}s.")

    # Pré-filtro: só vale a pena avaliar Jaro-Winkler em pares com cosseno razoável.
    # Limiar inferior derivado: se `score = 0.7*cos + 0.3*jw >= 0.88` e `jw <= 1`,
    # então `cos >= (0.88 - 0.30) / 0.70 ≈ 0.829`. Usamos 0.80 com folga.
    limiar_inferior_cosseno = max(
        0.0,
        (CONFIG["limiar_match"] - CONFIG["peso_jaro_winkler"]) / CONFIG["peso_cosseno"]
        - 0.03,
    )
    log("COSSENO", f"Pré-filtro de cosseno: >= {limiar_inferior_cosseno:.3f}.")

    triu_i, triu_j = np.triu_indices(n, k=1)
    valores_cos = sim_cosseno[triu_i, triu_j]
    mascara = valores_cos >= limiar_inferior_cosseno

    pares_i = triu_i[mascara]
    pares_j = triu_j[mascara]
    pares_cos = valores_cos[mascara].astype(np.float32)

    log(
        "COSSENO",
        f"Pares retidos após pré-filtro: {len(pares_i)} de {pares_potenciais}.",
    )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha em cosseno: {exc}"
    log("COSSENO", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Similaridade Jaro-Winkler (Jellyfish) e score combinado

# COMMAND ----------

try:
    import jellyfish
    from unidecode import unidecode

    def _normalizar(texto: str) -> str:
        return unidecode(str(texto)).lower().strip()

    nomes_norm = [_normalizar(x) for x in nomes]

    log("JARO_WINKLER", f"Calculando Jaro-Winkler para {len(pares_i)} pares.")
    inicio = time.time()
    jw_scores = np.empty(len(pares_i), dtype=np.float32)
    for k in range(len(pares_i)):
        a = nomes_norm[int(pares_i[k])]
        b = nomes_norm[int(pares_j[k])]
        jw_scores[k] = jellyfish.jaro_winkler_similarity(a, b)
    log("JARO_WINKLER", f"Jaro-Winkler calculado em {time.time() - inicio:.1f}s.")

    score_combinado = (
        CONFIG["peso_cosseno"] * pares_cos + CONFIG["peso_jaro_winkler"] * jw_scores
    )

    ESTADO_EXEC["pares_avaliados"] = int(len(pares_i))

    mascara_match = score_combinado >= CONFIG["limiar_match"]
    pares_match_i = pares_i[mascara_match]
    pares_match_j = pares_j[mascara_match]
    cos_match = pares_cos[mascara_match]
    jw_match = jw_scores[mascara_match]
    score_match = score_combinado[mascara_match]

    ESTADO_EXEC["pares_acima_limiar"] = int(len(pares_match_i))
    log(
        "SCORE",
        f"Pares com score >= {CONFIG['limiar_match']}: "
        f"{ESTADO_EXEC['pares_acima_limiar']}.",
    )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha em Jaro-Winkler/score: {exc}"
    log("JARO_WINKLER", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Agrupamento por Union-Find
# MAGIC
# MAGIC Cada componente conexo do grafo de pares acima do limiar vira um produto unificado.
# MAGIC Produtos sem nenhum match formam grupos unitários.

# COMMAND ----------

class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


try:
    uf = UnionFind(n)
    for a, b in zip(pares_match_i.tolist(), pares_match_j.tolist()):
        uf.union(int(a), int(b))

    raiz_por_indice = np.array([uf.find(i) for i in range(n)], dtype=np.int64)

    grupos_unicos = np.unique(raiz_por_indice)
    ESTADO_EXEC["grupos_formados"] = int(len(grupos_unicos))
    log("AGRUPAMENTO", f"Grupos formados: {ESTADO_EXEC['grupos_formados']}.")
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha no agrupamento: {exc}"
    log("AGRUPAMENTO", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Construção dos produtos unificados
# MAGIC
# MAGIC Para cada grupo geramos:
# MAGIC - `id_produto_unico`: UUID estável por execução.
# MAGIC - `nome_padrao`: nome com maior comprimento dentro do grupo (desempate alfabético).
# MAGIC - `justificativa`: texto explicando o agrupamento (tamanho do grupo, scores médios e exemplos).

# COMMAND ----------

try:
    # Mapeia raiz -> índices membros
    membros_por_raiz: Dict[int, List[int]] = {}
    for idx, raiz in enumerate(raiz_por_indice.tolist()):
        membros_por_raiz.setdefault(int(raiz), []).append(idx)

    # Mapeia raiz -> lista de scores que conectaram membros do grupo (para a justificativa)
    scores_por_raiz: Dict[int, List[Tuple[float, float, float]]] = {}
    for k in range(len(pares_match_i)):
        raiz = int(uf.find(int(pares_match_i[k])))
        scores_por_raiz.setdefault(raiz, []).append(
            (float(score_match[k]), float(cos_match[k]), float(jw_match[k]))
        )

    registros: List[Dict] = []

    for raiz, indices in membros_por_raiz.items():
        nomes_grupo = [nomes[i] for i in indices]
        # Desempate: maior comprimento; se empatar, menor alfabético
        nome_padrao = sorted(
            nomes_grupo, key=lambda s: (-len(s), s)
        )[0]

        id_unico = str(uuid.uuid4())
        scores = scores_por_raiz.get(raiz, [])

        if len(indices) == 1:
            justificativa = (
                "Produto isolado: nenhum outro registro atingiu o score combinado mínimo "
                f"de {CONFIG['limiar_match']:.2f}."
            )
        else:
            score_medio = float(np.mean([s[0] for s in scores])) if scores else 0.0
            cos_medio = float(np.mean([s[1] for s in scores])) if scores else 0.0
            jw_medio = float(np.mean([s[2] for s in scores])) if scores else 0.0
            exemplos = [x for x in nomes_grupo if x != nome_padrao][:3]
            exemplos_txt = "; ".join(f"'{e}'" for e in exemplos) or "(sem variações)"
            justificativa = (
                f"Agrupados {len(indices)} registros pelo score combinado "
                f"(70% cosseno + 30% Jaro-Winkler) >= {CONFIG['limiar_match']:.2f}. "
                f"Score médio do grupo: {score_medio:.3f} "
                f"(cosseno médio: {cos_medio:.3f}, Jaro-Winkler médio: {jw_medio:.3f}). "
                f"Nome padrão escolhido por maior comprimento: '{nome_padrao}'. "
                f"Variantes: {exemplos_txt}."
            )

        for idx in indices:
            registros.append(
                {
                    "id_produto_unico": id_unico,
                    "nome_padrao": nome_padrao,
                    "nome_produto_origem": nomes[idx],
                    "tamanho_grupo": int(len(indices)),
                    "justificativa": justificativa,
                    "_row_id": pdf.iloc[idx]["_row_id"],
                }
            )

    df_unificado_pd = pd.DataFrame(registros)
    log(
        "UNIFICACAO",
        f"Construídos {len(df_unificado_pd)} registros unificados em "
        f"{ESTADO_EXEC['grupos_formados']} grupos.",
    )
except Exception as exc:  # noqa: BLE001
    ESTADO_EXEC["status"] = "FALHA"
    ESTADO_EXEC["mensagem_erro"] = f"Falha na unificação: {exc}"
    log("UNIFICACAO", ESTADO_EXEC["mensagem_erro"], nivel="ERROR")
    gravar_auditoria(ESTADO_EXEC)
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Persistência em `gold_produtos_unificados`
# MAGIC
# MAGIC Faz join de volta com a Silver original (via `_row_id`) para preservar todos os atributos
# MAGIC de origem (sistema_origem, marca, categoria, etc.) ao lado das colunas de unificação.

# COMMAND ----------

try:
    schema_unificado = StructType(
        [
            StructField("id_produto_unico", StringType(), nullable=False),
            StructField("nome_padrao", StringType(), nullable=False),
            StructField("nome_produto_origem", StringType(), nullable=False),
            StructField("tamanho_grupo", IntegerType(), nullable=False),
            StructField("justificativa", StringType(), nullable=False),
            StructField("_row_id", StringType(), nullable=False),
        ]
    )

    sdf_unificado = spark.createDataFrame(df_unificado_pd, schema=schema_unificado)

    # Junta com a Silver para trazer atributos originais
    sdf_silver_keys = df_silver.select("_row_id", *colunas_origem)
    sdf_final = (
        sdf_unificado.join(sdf_silver_keys, on="_row_id", how="left")
        .drop("_row_id")
        .withColumn("data_processamento", F.current_timestamp())
    )

    log(
        "PERSISTENCIA",
        f"Gravando {sdf_final.count()} linhas em {NOME_TABELA_DESTINO}.",
    )
    (
        sdf_final.write.format("delta")
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
# MAGIC ## 11. Auditoria e fechamento

# COMMAND ----------

try:
    ESTADO_EXEC["status"] = "SUCESSO"
    gravar_auditoria(ESTADO_EXEC)
    log(
        "FIM",
        f"Execução concluída. Lidos={ESTADO_EXEC['registros_lidos']}, "
        f"pares_avaliados={ESTADO_EXEC['pares_avaliados']}, "
        f"pares_match={ESTADO_EXEC['pares_acima_limiar']}, "
        f"grupos={ESTADO_EXEC['grupos_formados']}.",
    )
except Exception as exc:  # noqa: BLE001
    log("AUDITORIA", f"Falha ao registrar auditoria final: {exc}.", nivel="ERROR")
    log("AUDITORIA", traceback.format_exc(), nivel="ERROR")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Amostras

# COMMAND ----------

display(
    spark.table(NOME_TABELA_DESTINO)
    .orderBy(F.col("tamanho_grupo").desc(), "nome_padrao")
    .limit(20)
)

# COMMAND ----------

display(
    spark.table(NOME_TABELA_AUDITORIA)
    .orderBy(F.col("inicio").desc())
    .limit(5)
)