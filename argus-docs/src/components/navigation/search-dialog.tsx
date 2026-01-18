"use client";

import * as React from "react";
import { Search as SearchIcon, FileText, Cpu, Shield, Zap } from "lucide-react";
import { useRouter } from "next/navigation";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

const searchResults = [
  { title: "Introduction", href: "/docs", icon: FileText, category: "Getting Started" },
  { title: "Installation", href: "/docs/installation", icon: Zap, category: "Getting Started" },
  { title: "Quick Start", href: "/docs/quick-start", icon: Zap, category: "Getting Started" },
  { title: "Conceptual Debate Graph", href: "/docs/concepts/cdag", icon: Cpu, category: "Core Concepts" },
  { title: "Research Debate Chain", href: "/docs/concepts/rdc", icon: Cpu, category: "Core Concepts" },
  { title: "Moderator Agent", href: "/docs/agents/moderator", icon: Shield, category: "Agents" },
  { title: "Bayesian Belief Prop", href: "/docs/concepts/bayesian", icon: Cpu, category: "Core Concepts" },
  { title: "API: RDCOrchestrator", href: "/docs/api/orchestrator", icon: FileText, category: "API Reference" },
];

export function SearchDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [query, setQuery] = React.useState("");
  const router = useRouter();

  const filteredResults = searchResults.filter((result) =>
    result.title.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[550px] p-0 gap-0 overflow-hidden border-border/50 bg-background/95 backdrop-blur-xl">
        <DialogHeader className="p-4 border-b border-border/50">
          <div className="flex items-center gap-3">
            <SearchIcon className="h-5 w-5 text-muted-foreground" />
            <Input
              placeholder="Search documentation..."
              className="border-0 focus-visible:ring-0 text-lg bg-transparent p-0"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoFocus
            />
          </div>
        </DialogHeader>
        <ScrollArea className="max-h-[400px]">
          <div className="p-2">
            {filteredResults.length > 0 ? (
              filteredResults.map((result) => (
                <button
                  key={result.href}
                  className="w-full flex items-center justify-between p-3 rounded-lg hover:bg-primary/10 group transition-all text-left"
                  onClick={() => {
                    router.push(result.href);
                    onOpenChange(false);
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div className="h-8 w-8 rounded-md bg-muted flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                      <result.icon className="h-4 w-4 text-muted-foreground group-hover:text-primary" />
                    </div>
                    <div>
                      <div className="text-sm font-medium">{result.title}</div>
                      <div className="text-[10px] text-muted-foreground uppercase tracking-widest">{result.category}</div>
                    </div>
                  </div>
                  <div className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                    Enter
                  </div>
                </button>
              ))
            ) : (
              <div className="p-8 text-center text-muted-foreground">
                No results found for "{query}"
              </div>
            )}
          </div>
        </ScrollArea>
        <div className="p-3 border-t border-border/50 bg-muted/30 flex items-center justify-between text-[10px] text-muted-foreground">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1"><kbd className="bg-muted px-1 rounded border border-border">↑↓</kbd> to navigate</span>
            <span className="flex items-center gap-1"><kbd className="bg-muted px-1 rounded border border-border">↵</kbd> to select</span>
          </div>
          <div className="flex items-center gap-1">
            <kbd className="bg-muted px-1 rounded border border-border">ESC</kbd> to close
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
