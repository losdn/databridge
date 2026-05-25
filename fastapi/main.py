"""
DataBridge API — Reconciliação e Classificação de Produtos.
Usa a Statement Execution REST API do Databricks (compatível com token PAT no Free Tier).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("databridge.api")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DATABRICKS_HOST: str = os.getenv(
    "DATABRICKS_HOST", "https://dbc-584c3aa2-de74.cloud.databricks.com"
).rstrip("/")
DATABRICKS_TOKEN: str = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_WAREHOUSE_ID: str = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
CATALOG: str = os.getenv("DATABRICKS_CATALOG", "databridge")
SCHEMA: str = os.getenv("DATABRICKS_SCHEMA", "raw_data")
QUERY_TIMEOUT: int = int(os.getenv("DATABRICKS_QUERY_TIMEOUT", "60"))
PAGINACAO_TAMANHO_MAX: int = int(os.getenv("PAGINACAO_TAMANHO_MAX", "200"))

# ---------------------------------------------------------------------------
# Cliente REST para Statement Execution API
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }


def executar_sql(sql: str, params: list[dict] | None = None) -> list[dict[str, Any]]:
    """
    Executa SQL via Statement Execution REST API e retorna lista de dicts.
    Funciona com token PAT no SQL Warehouse Serverless do Free Tier.
    """
    if not DATABRICKS_TOKEN or not DATABRICKS_WAREHOUSE_ID:
        raise RuntimeError("DATABRICKS_TOKEN e DATABRICKS_WAREHOUSE_ID são obrigatórios.")

    url = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    payload: dict[str, Any] = {
        "warehouse_id": DATABRICKS_WAREHOUSE_ID,
        "statement": sql,
        "wait_timeout": "50s",
        "on_wait_timeout": "CONTINUE",
        "format": "JSON_ARRAY",
        "disposition": "INLINE",
    }
    if params:
        payload["parameters"] = params

    logger.info("Executando SQL: %s", sql[:120].replace("\n", " "))

    with httpx.Client(timeout=QUERY_TIMEOUT + 10) as client:
        resp = client.post(url, headers=_headers(), json=payload)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Databricks retornou HTTP {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    state = data.get("status", {}).get("state", "")

    # Polling se a query ainda está rodando (PENDING/RUNNING)
    statement_id = data.get("statement_id")
    poll_url = f"{DATABRICKS_HOST}/api/2.0/sql/statements/{statement_id}"
    with httpx.Client(timeout=30) as client:
        for _ in range(30):
            if state not in ("PENDING", "RUNNING"):
                break
            time.sleep(2)
            pr = client.get(poll_url, headers=_headers())
            data = pr.json()
            state = data.get("status", {}).get("state", "")

    if state == "FAILED":
        err = data.get("status", {}).get("error", {})
        raise RuntimeError(f"Query falhou: {err.get('message', str(err))}")

    if state == "CANCELED":
        raise RuntimeError("Query cancelada por timeout.")

    # Monta lista de dicts a partir do schema + data
    schema = data.get("manifest", {}).get("schema", {}).get("columns", [])
    col_names = [c["name"] for c in schema]
    rows_raw = (data.get("result") or {}).get("data_array") or []

    rows = []
    for row in rows_raw:
        rows.append(dict(zip(col_names, row)))

    logger.info("Query retornou %d linha(s).", len(rows))
    return rows


def _tabela(nome: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{nome}"


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DataBridge API",
    description=(
        "API de reconciliação de produtos e classificação NCM "
        "sobre dados do Lakehouse Databricks (Unity Catalog)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://databridge-panel.losdn.workers.dev",
        "https://losdn.github.io",
        "http://localhost:8000",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas Pydantic v2
# ---------------------------------------------------------------------------

class ReconciliarRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"nome_produto": "paraf. sext. zinc. M2"}})
    nome_produto: str


class ReconciliarResponse(BaseModel):
    nome_entrada: str
    nome_padrao: str | None
    id_produto_unico: str | None
    score_similaridade: float | None
    top_3_similares: list[dict[str, Any]]


class ClassificarNcmRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"nome_produto": "parafuso sextavado M2"}})
    nome_produto: str


class ClassificarNcmResponse(BaseModel):
    nome_entrada: str
    nome_padrao: str | None
    top_ncm_1: str | None
    score_ncm_1: float | None
    descricao_ncm_1: str | None
    top_ncm_2: str | None
    score_ncm_2: float | None
    descricao_ncm_2: str | None
    top_ncm_3: str | None
    score_ncm_3: float | None
    descricao_ncm_3: str | None
    ncm_confianca: float | None


class ProdutoItem(BaseModel):
    model_config = ConfigDict(extra="allow")


class AuditoriaResponse(BaseModel):
    qtd_bronze: int | None
    qtd_gold: int | None
    qtd_deduplicada: int | None
    tempo_pipeline_segundos: int | None
    detalhes: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    api: str
    databricks: str
    warehouse_id: str
    host: str
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Sistema"], summary="Health check")
def health():
    """Retorna status da API e conectividade com o Databricks."""
    import datetime

    ts = datetime.datetime.utcnow().isoformat() + "Z"

    if not DATABRICKS_TOKEN or not DATABRICKS_WAREHOUSE_ID:
        return HealthResponse(
            status="degradado",
            api="ok",
            databricks="nao_configurado",
            warehouse_id=DATABRICKS_WAREHOUSE_ID or "(vazio)",
            host=DATABRICKS_HOST,
            timestamp=ts,
        )

    try:
        executar_sql("SELECT 1 AS ping")
        db_status = "ok"
        status = "ok"
    except Exception as exc:
        logger.warning("Health check Databricks falhou: %s", exc)
        db_status = f"erro: {str(exc)[:120]}"
        status = "degradado"

    return HealthResponse(
        status=status,
        api="ok",
        databricks=db_status,
        warehouse_id=DATABRICKS_WAREHOUSE_ID,
        host=DATABRICKS_HOST,
        timestamp=ts,
    )


@app.post("/reconciliar", response_model=ReconciliarResponse, tags=["Reconciliação"],
          summary="Reconcilia um nome livre contra produtos unificados.")
def reconciliar(req: ReconciliarRequest):
    """
    Recebe um nome de produto em formato livre e retorna o produto unificado
    mais próximo, usando similaridade textual (levenshtein normalizado) sobre
    a tabela `gold_produtos_unificados`.
    """
    nome = req.nome_produto.strip()
    if not nome:
        raise HTTPException(status_code=422, detail="nome_produto não pode ser vazio.")

    sql = f"""
    SELECT
        id_produto_unico,
        nome_padrao,
        (1.0 - levenshtein(lower(nome_padrao), lower(:p_nome)) * 1.0
            / GREATEST(length(nome_padrao), length(:p_nome), 1)
        ) AS score
    FROM {_tabela('gold_produtos_unificados')}
    ORDER BY score DESC
    LIMIT 10
    """
    try:
        rows = executar_sql(sql, [{"name": "p_nome", "value": nome, "type": "STRING"}])
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if not rows:
        raise HTTPException(status_code=404, detail="Nenhum produto encontrado na tabela gold_produtos_unificados.")

    melhor = rows[0]
    top3 = rows[:3]

    return ReconciliarResponse(
        nome_entrada=nome,
        nome_padrao=melhor.get("nome_padrao"),
        id_produto_unico=melhor.get("id_produto_unico"),
        score_similaridade=float(melhor["score"]) if melhor.get("score") is not None else None,
        top_3_similares=[
            {
                "nome_padrao": r.get("nome_padrao"),
                "id_produto_unico": r.get("id_produto_unico"),
                "score": float(r["score"]) if r.get("score") is not None else None,
            }
            for r in top3
        ],
    )


@app.post("/classificar-ncm", response_model=ClassificarNcmResponse, tags=["NCM"],
          summary="Retorna a classificação NCM (top 3).")
def classificar_ncm(req: ClassificarNcmRequest):
    """
    Recebe um nome de produto e retorna os 3 NCMs mais prováveis
    a partir da tabela `gold_produtos_com_ncm`.
    """
    nome = req.nome_produto.strip()
    if not nome:
        raise HTTPException(status_code=422, detail="nome_produto não pode ser vazio.")

    sql = f"""
    SELECT
        nome_padrao,
        top_ncm_1, score_ncm_1, descricao_ncm_1,
        top_ncm_2, score_ncm_2, descricao_ncm_2,
        top_ncm_3, score_ncm_3, descricao_ncm_3,
        ncm_confianca,
        (1.0 - levenshtein(lower(nome_padrao), lower(:p_nome)) * 1.0
            / GREATEST(length(nome_padrao), length(:p_nome), 1)
        ) AS score_texto
    FROM {_tabela('gold_produtos_com_ncm')}
    ORDER BY score_texto DESC
    LIMIT 1
    """
    try:
        rows = executar_sql(sql, [{"name": "p_nome", "value": nome, "type": "STRING"}])
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if not rows:
        raise HTTPException(status_code=404, detail="Nenhum produto encontrado na tabela gold_produtos_com_ncm.")

    r = rows[0]

    def _float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return ClassificarNcmResponse(
        nome_entrada=nome,
        nome_padrao=r.get("nome_padrao"),
        top_ncm_1=r.get("top_ncm_1"),
        score_ncm_1=_float(r.get("score_ncm_1")),
        descricao_ncm_1=r.get("descricao_ncm_1"),
        top_ncm_2=r.get("top_ncm_2"),
        score_ncm_2=_float(r.get("score_ncm_2")),
        descricao_ncm_2=r.get("descricao_ncm_2"),
        top_ncm_3=r.get("top_ncm_3"),
        score_ncm_3=_float(r.get("score_ncm_3")),
        descricao_ncm_3=r.get("descricao_ncm_3"),
        ncm_confianca=_float(r.get("ncm_confianca")),
    )


@app.get("/produtos", tags=["Produtos"], summary="Lista paginada de produtos.")
def listar_produtos(
    pagina: int = Query(1, ge=1, description="Número da página (começa em 1)"),
    tamanho: int = Query(20, ge=1, le=200, description="Itens por página (máx 200)"),
):
    """
    Retorna produtos de `gold_produtos_com_ncm` com paginação simples.
    """
    offset = (pagina - 1) * tamanho

    sql_total = f"SELECT COUNT(*) AS total FROM {_tabela('gold_produtos_com_ncm')}"
    sql_dados = f"""
    SELECT *
    FROM {_tabela('gold_produtos_com_ncm')}
    ORDER BY nome_padrao
    LIMIT {int(tamanho)} OFFSET {int(offset)}
    """

    try:
        total_rows = executar_sql(sql_total)
        total = int(total_rows[0]["total"]) if total_rows else 0
        dados = executar_sql(sql_dados)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "pagina": pagina,
        "tamanho": tamanho,
        "total": total,
        "paginas": (total + tamanho - 1) // tamanho if tamanho else 0,
        "produtos": dados,
    }


@app.get("/auditoria", response_model=AuditoriaResponse, tags=["Auditoria"],
         summary="Resumo do pipeline.")
def auditoria():
    """
    Retorna o resumo consolidado do pipeline DataBridge a partir de
    `gold_audit_consolidado`. Se a tabela não existir, faz fallback em `audit_logs`.
    """
    # Tenta gold_audit_consolidado primeiro
    try:
        rows = executar_sql(f"SELECT * FROM {_tabela('gold_audit_consolidado')} LIMIT 1")
        if rows:
            r = rows[0]
            detalhes_rows = executar_sql(
                f"SELECT * FROM {_tabela('gold_audit_consolidado')} LIMIT 50"
            )
            return AuditoriaResponse(
                qtd_bronze=int(r["qtd_bronze"]) if r.get("qtd_bronze") is not None else None,
                qtd_gold=int(r["qtd_gold"]) if r.get("qtd_gold") is not None else None,
                qtd_deduplicada=int(r["qtd_deduplicada"]) if r.get("qtd_deduplicada") is not None else None,
                tempo_pipeline_segundos=int(r["tempo_pipeline_segundos"]) if r.get("tempo_pipeline_segundos") is not None else None,
                detalhes=detalhes_rows,
            )
    except Exception as exc:
        logger.warning("gold_audit_consolidado indisponível, usando fallback: %s", exc)

    # Fallback: resume a partir de audit_logs
    try:
        rows = executar_sql(
            f"SELECT nome_job, status, registros_lidos FROM {_tabela('audit_logs')} ORDER BY inicio"
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return AuditoriaResponse(
        qtd_bronze=None,
        qtd_gold=None,
        qtd_deduplicada=None,
        tempo_pipeline_segundos=None,
        detalhes=rows,
    )