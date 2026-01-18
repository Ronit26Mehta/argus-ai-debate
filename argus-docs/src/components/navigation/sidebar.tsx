"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  BookOpen, 
  Terminal, 
  Cpu, 
  Network, 
  Shield, 
  Database, 
  BarChart3, 
  Users,
  Layers,
  Zap,
  CheckCircle2,
  Code2
} from "lucide-react";

const sidebarNav = [
  {
    title: "Getting Started",
    items: [
      { title: "Introduction", href: "/docs", icon: BookOpen },
      { title: "Installation", href: "/docs/installation", icon: Terminal },
      { title: "Quick Start", href: "/docs/quick-start", icon: Zap },
    ],
  },
  {
    title: "Core Concepts",
    items: [
      { title: "Research Debate Chain", href: "/docs/concepts/rdc", icon: Layers },
      { title: "Conceptual Debate Graph", href: "/docs/concepts/cdag", icon: Network },
      { title: "EDDO Algorithm", href: "/docs/concepts/eddo", icon: Cpu },
      { title: "Bayesian Belief Prop", href: "/docs/concepts/bayesian", icon: BarChart3 },
    ],
  },
  {
    title: "Agents",
    items: [
      { title: "Overview", href: "/docs/agents", icon: Users },
      { title: "Moderator", href: "/docs/agents/moderator", icon: Shield },
      { title: "Specialist", href: "/docs/agents/specialist", icon: Database },
      { title: "Refuter", href: "/docs/agents/refuter", icon: Zap },
      { title: "Jury", href: "/docs/agents/jury", icon: CheckCircle2 },
    ],
  },
  {
    title: "Knowledge & RAG",
    items: [
      { title: "Hybrid Retrieval", href: "/docs/knowledge/retrieval", icon: Database },
      { title: "External Connectors", href: "/docs/knowledge/connectors", icon: Network },
    ],
  },
  {
    title: "Decision & Planning",
    items: [
      { title: "Value of Info (VoI)", href: "/docs/decision/voi", icon: BarChart3 },
      { title: "Calibration", href: "/docs/decision/calibration", icon: Cpu },
    ],
  },
  {
    title: "API Reference",
    items: [
      { title: "Orchestrator", href: "/docs/api/orchestrator", icon: Code2 },
      { title: "CDAG", href: "/docs/api/cdag", icon: Code2 },
      { title: "LLM Providers", href: "/docs/api/llm", icon: Code2 },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed top-16 z-30 -ml-2 hidden h-[calc(100vh-4rem)] w-full shrink-0 md:sticky md:block">
      <ScrollArea className="h-full py-6 pr-6 lg:py-8">
        <div className="flex flex-col gap-8">
          {sidebarNav.map((section) => (
            <div key={section.title} className="px-2">
              <h4 className="mb-2 px-2 text-sm font-semibold tracking-tight text-foreground/70 uppercase">
                {section.title}
              </h4>
              <div className="flex flex-col gap-1">
                {section.items.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm font-medium transition-all hover:bg-muted group",
                      pathname === item.href
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    <item.icon className={cn(
                      "h-4 w-4 shrink-0 transition-colors",
                      pathname === item.href ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                    )} />
                    {item.title}
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </aside>
  );
}
