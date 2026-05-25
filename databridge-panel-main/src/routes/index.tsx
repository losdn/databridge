import { createFileRoute } from "@tanstack/react-router";
import { Dashboard } from "@/components/dashboard/Dashboard";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "DataBridge — Painel de Padronização" },
      { name: "description", content: "Painel DataBridge para padronização de produtos, reconciliação e classificação NCM." },
    ],
  }),
});

function Index() {
  return <Dashboard />;
}
