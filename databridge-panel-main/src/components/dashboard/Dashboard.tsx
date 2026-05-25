import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Boxes,
  CheckCircle2,
  Copy,
  Layers,
  Loader2,
  Search,
  Sparkles,
  Timer,
  Zap,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

// ---------------------------------------------------------------------------
// Configuração da API
// ---------------------------------------------------------------------------
const API_BASE =
  (import.meta as any).env?.VITE_API_BASE ?? "https://databridge.up.railway.app";

async function apiRequest<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detalhe = "";
    try {
      const body = await res.json();
      detalhe = body?.detalhe ?? body?.detail ?? body?.erro ?? JSON.stringify(body);
    } catch {
      detalhe = await res.text();
    }
    throw new Error(`[${res.status}] ${detalhe || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Tipos da API
// ---------------------------------------------------------------------------
type ProdutoSimilarApi = {
  nome_padrao: string;
  id_produto_unico: string | null;
  score_similaridade: number;
};

type ReconciliarResponse = {
  nome_consultado: string;
  nome_padrao: string | null;
  id_produto_unico: string | null;
  score_similaridade: number | null;
  top_3_similares: ProdutoSimilarApi[];
};

type ClassificarNcmResponse = {
  nome_consultado: string;
  nome_padrao: string | null;
  top_ncm_1: string | null;
  score_ncm_1: number | null;
  descricao_ncm_1: string | null;
  top_ncm_2: string | null;
  score_ncm_2: number | null;
  descricao_ncm_2: string | null;
  top_ncm_3: string | null;
  score_ncm_3: number | null;
  descricao_ncm_3: string | null;
  ncm_confianca: number | null;
};

type AuditoriaResponse = {
  qtd_bronze: number | null;
  qtd_gold: number | null;
  qtd_deduplicada: number | null;
  tempo_pipeline: number | null;
  // alguns deploys expõem o campo já em segundos com nome explícito
  tempo_pipeline_segundos?: number | null;
  detalhes: Record<string, unknown>[];
};

type ProdutoItemApi = {
  nome_padrao: string | null;
  id_produto_unico?: string | null;
  top_ncm_1?: string | null;
  score_ncm_1?: number | null;
  descricao_ncm_1?: string | null;
  ncm_confianca?: number | null;
  [key: string]: unknown;
};

type ProdutosResponse = {
  pagina: number;
  tamanho: number;
  total_retornado: number;
  itens: ProdutoItemApi[];
};

// ---------------------------------------------------------------------------
// Tipos para a UI (mantém compatibilidade visual)
// ---------------------------------------------------------------------------
type ReconcileViewModel = {
  canonical: string;
  similars: { name: string; score: number }[];
};

type NcmViewModel = { code: string; desc: string; conf: number }[];

type ProdutoLinha = { name: string; ncm: string; desc: string; conf: number };

type SummaryCard = {
  label: string;
  value: string;
  icon: typeof Boxes;
  accent: string;
};

const PLACEHOLDER_VALOR = "—";

const ICONES_SUMMARY = {
  bronze: { icon: Boxes, accent: "from-fuchsia-500 to-purple-600" },
  gold: { icon: CheckCircle2, accent: "from-purple-500 to-violet-700" },
  deduplicada: { icon: Copy, accent: "from-pink-500 to-fuchsia-600" },
  tempo: { icon: Timer, accent: "from-violet-500 to-purple-700" },
} as const;

function formatarInteiro(valor: number | null | undefined): string {
  if (valor === null || valor === undefined || Number.isNaN(valor)) {
    return PLACEHOLDER_VALOR;
  }
  return new Intl.NumberFormat("pt-BR").format(valor);
}

function formatarTempo(segundos: number | null | undefined): string {
  if (segundos === null || segundos === undefined || Number.isNaN(segundos)) {
    return PLACEHOLDER_VALOR;
  }
  if (segundos < 60) return `${segundos.toFixed(0)}s`;
  const min = Math.floor(segundos / 60);
  const sec = Math.round(segundos % 60);
  return `${min}m ${sec.toString().padStart(2, "0")}s`;
}

function montarSummary(audit: AuditoriaResponse | null): SummaryCard[] {
  const tempo = audit?.tempo_pipeline_segundos ?? audit?.tempo_pipeline ?? null;
  return [
    {
      label: "Produtos no Pipeline",
      value: formatarInteiro(audit?.qtd_bronze),
      icon: ICONES_SUMMARY.bronze.icon,
      accent: ICONES_SUMMARY.bronze.accent,
    },
    {
      label: "Produtos Padronizados",
      value: formatarInteiro(audit?.qtd_gold),
      icon: ICONES_SUMMARY.gold.icon,
      accent: ICONES_SUMMARY.gold.accent,
    },
    {
      label: "Duplicatas Removidas",
      value: formatarInteiro(audit?.qtd_deduplicada),
      icon: ICONES_SUMMARY.deduplicada.icon,
      accent: ICONES_SUMMARY.deduplicada.accent,
    },
    {
      label: "Última Execução",
      value: formatarTempo(tempo),
      icon: ICONES_SUMMARY.tempo.icon,
      accent: ICONES_SUMMARY.tempo.accent,
    },
  ];
}

function mapearProduto(item: ProdutoItemApi): ProdutoLinha {
  return {
    name: item.nome_padrao ?? PLACEHOLDER_VALOR,
    ncm: item.top_ncm_1 ?? PLACEHOLDER_VALOR,
    desc: item.descricao_ncm_1 ?? "",
    conf: typeof item.ncm_confianca === "number"
      ? item.ncm_confianca
      : typeof item.score_ncm_1 === "number"
        ? item.score_ncm_1
        : 0,
  };
}

// ---------------------------------------------------------------------------
// Componente
// ---------------------------------------------------------------------------
export function Dashboard() {
  const [recQuery, setRecQuery] = useState("");
  const [recResult, setRecResult] = useState<ReconcileViewModel | null>(null);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);

  const [ncmQuery, setNcmQuery] = useState("");
  const [ncmResult, setNcmResult] = useState<NcmViewModel | null>(null);
  const [ncmLoading, setNcmLoading] = useState(false);
  const [ncmError, setNcmError] = useState<string | null>(null);

  const [audit, setAudit] = useState<AuditoriaResponse | null>(null);
  const [auditLoading, setAuditLoading] = useState(true);
  const [auditError, setAuditError] = useState<string | null>(null);

  const [page, setPage] = useState(1);
  const pageSize = 5;
  const [paged, setPaged] = useState<ProdutoLinha[]>([]);
  const [produtosLoading, setProdutosLoading] = useState(true);
  const [produtosError, setProdutosError] = useState<string | null>(null);
  // Como o backend não retorna total, inferimos se há próxima pela quantidade retornada.
  const [hasNext, setHasNext] = useState(false);

  const [healthOnline, setHealthOnline] = useState(true);

  const summary = useMemo(() => montarSummary(audit), [audit]);

  // Carregamento inicial: auditoria + health
  useEffect(() => {
    let ativo = true;
    (async () => {
      setAuditLoading(true);
      setAuditError(null);
      try {
        const data = await apiRequest<AuditoriaResponse>("/auditoria");
        if (ativo) setAudit(data);
      } catch (err) {
        if (ativo) setAuditError((err as Error).message);
      } finally {
        if (ativo) setAuditLoading(false);
      }
    })();

    (async () => {
      try {
        const data = await apiRequest<{ status: string; databricks: string }>(
          "/health",
        );
        if (ativo) setHealthOnline(data.status === "ok" && data.databricks === "ok");
      } catch {
        if (ativo) setHealthOnline(false);
      }
    })();

    return () => {
      ativo = false;
    };
  }, []);

  // Carregamento dos produtos com paginação real
  useEffect(() => {
    let ativo = true;
    (async () => {
      setProdutosLoading(true);
      setProdutosError(null);
      try {
        const data = await apiRequest<ProdutosResponse>(
          `/produtos?pagina=${page}&tamanho=${pageSize}`,
        );
        if (!ativo) return;
        setPaged(data.itens.map(mapearProduto));
        setHasNext(data.total_retornado >= pageSize);
      } catch (err) {
        if (ativo) {
          setProdutosError((err as Error).message);
          setPaged([]);
          setHasNext(false);
        }
      } finally {
        if (ativo) setProdutosLoading(false);
      }
    })();
    return () => {
      ativo = false;
    };
  }, [page]);

  async function executarReconciliar() {
    if (!recQuery.trim() || recLoading) return;
    setRecLoading(true);
    setRecError(null);
    setRecResult(null);
    try {
      const data = await apiRequest<ReconciliarResponse>("/reconciliar", {
        method: "POST",
        body: JSON.stringify({ nome_produto: recQuery.trim() }),
      });
      setRecResult({
        canonical: data.nome_padrao ?? "Sem correspondência encontrada",
        similars: data.top_3_similares.map((s) => ({
          name: s.nome_padrao,
          score: s.score_similaridade,
        })),
      });
    } catch (err) {
      setRecError((err as Error).message);
    } finally {
      setRecLoading(false);
    }
  }

  async function executarClassificarNcm() {
    if (!ncmQuery.trim() || ncmLoading) return;
    setNcmLoading(true);
    setNcmError(null);
    setNcmResult(null);
    try {
      const data = await apiRequest<ClassificarNcmResponse>("/classificar-ncm", {
        method: "POST",
        body: JSON.stringify({ nome_produto: ncmQuery.trim() }),
      });
      const sugestoes: NcmViewModel = [];
      const slots: [string | null, number | null, string | null][] = [
        [data.top_ncm_1, data.score_ncm_1, data.descricao_ncm_1],
        [data.top_ncm_2, data.score_ncm_2, data.descricao_ncm_2],
        [data.top_ncm_3, data.score_ncm_3, data.descricao_ncm_3],
      ];
      for (const [code, score, desc] of slots) {
        if (code) {
          sugestoes.push({
            code,
            desc: desc ?? "",
            conf: typeof score === "number" ? score : 0,
          });
        }
      }
      setNcmResult(sugestoes);
    } catch (err) {
      setNcmError((err as Error).message);
    } finally {
      setNcmLoading(false);
    }
  }

  return (
    <div className="min-h-screen text-foreground">
      {/* Top bar */}
      <header className="glass sticky top-0 z-30 border-b border-[color:var(--glass-border)]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-fuchsia-500 to-purple-700 animate-pulse-neon">
              <Zap className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight gradient-text">DataBridge</h1>
              <p className="text-xs text-muted-foreground">Padronização inteligente de produtos</p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full glass px-4 py-2">
            <span className="relative flex h-2.5 w-2.5">
              <span
                className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${
                  healthOnline ? "bg-emerald-400" : "bg-rose-400"
                }`}
              />
              <span
                className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
                  healthOnline ? "bg-emerald-400" : "bg-rose-400"
                }`}
              />
            </span>
            <span className="text-sm font-medium">
              {healthOnline ? "Conexão online" : "Conexão indisponível"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-8 px-6 py-10">
        {/* Hero */}
        <section className="space-y-2">
          <h2 className="text-3xl font-bold sm:text-4xl">
            Visão geral do <span className="gradient-text">DataBridge</span>
          </h2>
          <p className="text-muted-foreground">Monitore, reconcilie e classifique os produtos em tempo real.</p>
        </section>

        {/* Summary cards */}
        <section className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {summary.map((c) => (
            <div
              key={c.label}
              className="group relative overflow-hidden rounded-2xl glass p-5 transition-all hover:-translate-y-1 hover:neon-border"
            >
              <div className={`absolute -right-10 -top-10 h-32 w-32 rounded-full bg-gradient-to-br ${c.accent} opacity-30 blur-2xl transition-opacity group-hover:opacity-60`} />
              <div className="relative flex items-start justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">{c.label}</p>
                  <p className="mt-2 text-3xl font-bold tracking-tight">
                    {auditLoading ? (
                      <Loader2 className="h-6 w-6 animate-spin text-fuchsia-300" />
                    ) : (
                      c.value
                    )}
                  </p>
                </div>
                <div className={`flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br ${c.accent} shadow-lg`}>
                  <c.icon className="h-5 w-5 text-white" />
                </div>
              </div>
              <div className="mt-4 flex items-center gap-1 text-xs text-muted-foreground">
                <Activity className="h-3 w-3 text-fuchsia-400" />
                <span>{auditError ? "Falha ao atualizar" : "Atualizado agora"}</span>
              </div>
            </div>
          ))}
        </section>

        {auditError && (
          <p className="text-xs text-rose-300">
            Não foi possível carregar a auditoria: {auditError}
          </p>
        )}

        {/* Reconciliation + NCM */}
        <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Reconciliation */}
          <div className="rounded-2xl glass p-6">
            <div className="mb-4 flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-fuchsia-500 to-purple-600">
                <Sparkles className="h-4 w-4 text-white" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Reconciliação</h3>
                <p className="text-xs text-muted-foreground">Visão do comprador</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Digite o nome livre do produto..."
                value={recQuery}
                onChange={(e) => setRecQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && executarReconciliar()}
                className="glass border-[color:var(--glass-border)] bg-transparent"
              />
              <Button
                onClick={executarReconciliar}
                disabled={!recQuery.trim() || recLoading}
                className="bg-gradient-to-r from-fuchsia-500 to-purple-600 shadow-[0_0_15px_oklch(0.62_0.27_305/0.6)] hover:opacity-90"
              >
                {recLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Search className="mr-2 h-4 w-4" />
                )}
                Padronizar
              </Button>
            </div>

            {recError && (
              <p className="mt-3 text-xs text-rose-300">{recError}</p>
            )}

            {recResult && (
              <div className="mt-5 space-y-3">
                <div className="rounded-xl border border-fuchsia-500/40 bg-fuchsia-500/5 p-4">
                  <p className="text-xs uppercase tracking-wider text-fuchsia-300">Nome padronizado</p>
                  <p className="mt-1 font-semibold neon-text">{recResult.canonical}</p>
                </div>
                <p className="text-xs uppercase tracking-wider text-muted-foreground">Similares</p>
                {recResult.similars.length === 0 ? (
                  <p className="text-sm text-muted-foreground">Nenhum similar encontrado.</p>
                ) : (
                  <ul className="space-y-2">
                    {recResult.similars.map((s, i) => (
                      <li
                        key={`${s.name}-${i}`}
                        className="flex items-center justify-between rounded-lg glass px-3 py-2"
                      >
                        <span className="text-sm">{s.name}</span>
                        <ScoreBadge score={s.score} />
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>

          {/* NCM */}
          <div className="rounded-2xl glass p-6">
            <div className="mb-4 flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-pink-600">
                <Layers className="h-4 w-4 text-white" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Classificação NCM</h3>
                <p className="text-xs text-muted-foreground">Visão do fiscal</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Digite o produto..."
                value={ncmQuery}
                onChange={(e) => setNcmQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && executarClassificarNcm()}
                className="glass border-[color:var(--glass-border)] bg-transparent"
              />
              <Button
                onClick={executarClassificarNcm}
                disabled={!ncmQuery.trim() || ncmLoading}
                className="bg-gradient-to-r from-purple-500 to-pink-600 shadow-[0_0_15px_oklch(0.65_0.3_340/0.6)] hover:opacity-90"
              >
                {ncmLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Search className="mr-2 h-4 w-4" />
                )}
                Classificar
              </Button>
            </div>

            {ncmError && (
              <p className="mt-3 text-xs text-rose-300">{ncmError}</p>
            )}

            {ncmResult && (
              <div className="mt-5 space-y-3">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">Top 3 sugestões</p>
                {ncmResult.length === 0 ? (
                  <p className="text-sm text-muted-foreground">Nenhuma sugestão de NCM.</p>
                ) : (
                  ncmResult.map((n, i) => (
                    <div
                      key={`${n.code}-${i}`}
                      className="rounded-xl glass p-4 transition-all hover:neon-border"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="rounded bg-fuchsia-500/20 px-2 py-0.5 font-mono text-xs text-fuchsia-200">
                              #{i + 1}
                            </span>
                            <span className="font-mono font-semibold">{n.code}</span>
                          </div>
                          <p className="mt-1 text-sm text-muted-foreground">{n.desc}</p>
                        </div>
                        <ScoreBadge score={n.conf} />
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </section>

        {/* Table */}
        <section className="rounded-2xl glass p-6">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-lg font-semibold">Produtos padronizados</h3>
              <p className="text-xs text-muted-foreground">Visão geral do catálogo</p>
            </div>
            <Badge className="bg-fuchsia-500/20 text-fuchsia-200 hover:bg-fuchsia-500/30">
              {audit?.qtd_gold != null
                ? `${formatarInteiro(audit.qtd_gold)} produtos`
                : `${paged.length} nesta página`}
            </Badge>
          </div>
          <div className="overflow-hidden rounded-xl border border-[color:var(--glass-border)]">
            <Table>
              <TableHeader>
                <TableRow className="border-[color:var(--glass-border)] hover:bg-transparent">
                  <TableHead>Nome padronizado</TableHead>
                  <TableHead>NCM</TableHead>
                  <TableHead>Descrição NCM</TableHead>
                  <TableHead className="text-right">Confiança</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {produtosLoading ? (
                  <TableRow className="border-[color:var(--glass-border)]">
                    <TableCell colSpan={4} className="text-center text-sm text-muted-foreground">
                      <span className="inline-flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Carregando produtos...
                      </span>
                    </TableCell>
                  </TableRow>
                ) : produtosError ? (
                  <TableRow className="border-[color:var(--glass-border)]">
                    <TableCell colSpan={4} className="text-center text-sm text-rose-300">
                      Falha ao carregar produtos: {produtosError}
                    </TableCell>
                  </TableRow>
                ) : paged.length === 0 ? (
                  <TableRow className="border-[color:var(--glass-border)]">
                    <TableCell colSpan={4} className="text-center text-sm text-muted-foreground">
                      Sem produtos para exibir.
                    </TableCell>
                  </TableRow>
                ) : (
                  paged.map((p, i) => (
                    <TableRow
                      key={`${p.name}-${i}`}
                      className="border-[color:var(--glass-border)]"
                    >
                      <TableCell className="font-medium">{p.name}</TableCell>
                      <TableCell className="font-mono text-fuchsia-200">{p.ncm}</TableCell>
                      <TableCell className="max-w-md text-sm text-muted-foreground">
                        {p.desc}
                      </TableCell>
                      <TableCell className="text-right">
                        <ScoreBadge score={p.conf} />
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          <div className="mt-4 flex items-center justify-between">
            <p className="text-xs text-muted-foreground">
              Página {page}
              {hasNext ? "" : " (final)"}
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1 || produtosLoading}
                className="glass border-[color:var(--glass-border)]"
              >
                Anterior
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => p + 1)}
                disabled={!hasNext || produtosLoading}
                className="glass border-[color:var(--glass-border)]"
              >
                Próxima
              </Button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const safe = Number.isFinite(score) ? score : 0;
  const pct = Math.max(0, Math.min(100, Math.round(safe * 100)));
  const tone =
    pct >= 90
      ? "from-emerald-400 to-emerald-600 shadow-[0_0_12px_rgba(16,185,129,0.6)]"
      : pct >= 75
      ? "from-fuchsia-400 to-purple-600 shadow-[0_0_12px_oklch(0.62_0.27_305/0.6)]"
      : "from-amber-400 to-orange-600 shadow-[0_0_12px_rgba(251,146,60,0.6)]";
  return (
    <span className={`inline-flex items-center rounded-full bg-gradient-to-r ${tone} px-2.5 py-0.5 text-xs font-semibold text-white`}>
      {pct}%
    </span>
  );
}
