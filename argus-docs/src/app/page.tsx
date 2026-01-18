"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { 
  Shield, 
  Cpu, 
  Database, 
  Network, 
  Zap, 
  Search, 
  ArrowRight, 
  CheckCircle2, 
  Layers, 
  BarChart3,
  Terminal,
  Code2
} from "lucide-react";
import { Button } from "@/components/ui/button";

const features = [
  {
    title: "Conceptual Debate Graph (C-DAG)",
    description: "Structured Bayesian reasoning where claims and evidence form a directed graph of influence.",
    icon: Network,
    color: "text-blue-500",
    bg: "bg-blue-500/10",
  },
  {
    title: "Multi-Agent Orchestration",
    description: "Specialized agents (Moderator, Specialist, Refuter, Jury) competing to find the truth.",
    icon: Cpu,
    color: "text-purple-500",
    bg: "bg-purple-500/10",
  },
  {
    title: "Full Provenance Tracking",
    description: "PROV-O compatible ledger with hash-chain integrity for complete auditability.",
    icon: Shield,
    color: "text-emerald-500",
    bg: "bg-emerald-500/10",
  },
  {
    title: "Hybrid Retrieval System",
    description: "State-of-the-art BM25 and FAISS dense retrieval with reciprocal rank fusion.",
    icon: Database,
    color: "text-orange-500",
    bg: "bg-orange-500/10",
  },
];

const agents = [
  {
    name: "Moderator",
    role: "The Architect",
    task: "Creates debate agendas, manages rounds, and evaluates stopping criteria.",
    icon: Layers,
  },
  {
    name: "Specialist",
    role: "The Researcher",
    task: "Gathers evidence from external sources using hybrid retrieval and tools.",
    icon: Search,
  },
  {
    name: "Refuter",
    role: "The Critic",
    task: "Challenges claims, finds logical fallacies, and generates counter-evidence.",
    icon: Zap,
  },
  {
    name: "Jury",
    role: "The Judge",
    task: "Renders verdicts using Bayesian aggregation and confidence calibration.",
    icon: Shield,
  },
];

const stats = [
  { label: "LLM Providers", value: "27+" },
  { label: "Pre-built Tools", value: "19+" },
  { label: "Embedding Models", value: "16+" },
  { label: "Novel Metrics", value: "8" },
];

export default function LandingPage() {
  return (
    <div className="relative overflow-hidden">
      {/* Hero Section */}
      <section className="relative pt-20 pb-32 md:pt-32 md:pb-48">
        {/* Background Gradients */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full -z-10 pointer-events-none">
          <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-primary/20 rounded-full blur-[120px] opacity-50" />
          <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-blue-500/10 rounded-full blur-[120px] opacity-50" />
        </div>

        <div className="container mx-auto px-4 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <span className="px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium inline-flex items-center gap-2 mb-6">
              <Zap className="h-4 w-4" />
              v1.4.2 is now stable
            </span>
            <h1 className="text-5xl md:text-8xl font-black tracking-tight mb-8">
              Reasoning Beyond <br />
              <span className="text-gradient">Inference.</span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-12 leading-relaxed">
              ARGUS is a debate-native, multi-agent AI framework for evidence-based reasoning. 
              Calibrated truth through structured argumentation and Bayesian belief propagation.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Button size="lg" className="rounded-full px-8 py-6 text-lg h-auto group" asChild>
                <Link href="/docs">
                  Get Started 
                  <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="rounded-full px-8 py-6 text-lg h-auto" asChild>
                <Link href="https://github.com/Ronit26Mehta/argus-ai-debate" target="_blank">
                  View on GitHub
                </Link>
              </Button>
            </div>
          </motion.div>

          {/* Hero Image / Illustration Placeholder */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="mt-20 relative mx-auto max-w-5xl rounded-2xl border border-border/50 bg-muted/30 p-4 shadow-2xl overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent z-10" />
            <div className="flex items-center gap-2 mb-4 px-2">
              <div className="flex gap-1.5">
                <div className="h-3 w-3 rounded-full bg-red-500/50" />
                <div className="h-3 w-3 rounded-full bg-yellow-500/50" />
                <div className="h-3 w-3 rounded-full bg-green-500/50" />
              </div>
              <div className="text-xs text-muted-foreground font-mono ml-4">argus --debate "Metformin efficacy"</div>
            </div>
            <div className="aspect-[16/9] bg-background/50 rounded-lg flex items-center justify-center border border-border/50 relative overflow-hidden">
               {/* Terminal Mockup */}
               <div className="w-full h-full p-8 text-left font-mono text-sm overflow-hidden">
                  <div className="text-primary mb-2">➜ argus debate "Metformin reduces HbA1c by >1% in Type 2 diabetes"</div>
                  <div className="text-muted-foreground mb-4">🎯 Debating: Metformin reduces HbA1c by >1% in Type 2 diabetes</div>
                  <div className="text-blue-400 mb-1">[ROUND 1] Moderator creates agenda...</div>
                  <div className="text-emerald-400 mb-1">[ROUND 1] Specialist found 12 clinical trials via PubMed/arXiv</div>
                  <div className="text-orange-400 mb-4">[ROUND 1] Refuter challenged 3 trials for sample bias</div>
                  <div className="text-blue-400 mb-1">[ROUND 2] Specialist gathered counter-evidence...</div>
                  <div className="text-purple-400 mb-4">[ROUND 3] Convergence detected (Δp < 0.01)</div>
                  <div className="border border-primary/30 bg-primary/5 p-4 rounded-lg">
                    <div className="text-primary font-bold mb-1">📊 VERDICT: SUPPORTED</div>
                    <div className="flex gap-8 mb-2">
                      <span>Posterior: 0.842</span>
                      <span>Confidence: 0.910</span>
                      <span>Rounds: 3</span>
                    </div>
                    <div className="text-muted-foreground italic">
                      "Bayesian aggregation of 18 evidence nodes suggests strong consensus across clinical literature..."
                    </div>
                  </div>
               </div>
               
               {/* Decorative C-DAG Nodes */}
               <div className="absolute inset-0 pointer-events-none opacity-20">
                 <div className="absolute top-1/4 left-1/4 h-2 w-2 rounded-full bg-primary animate-ping" />
                 <div className="absolute top-1/2 right-1/3 h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                 <div className="absolute bottom-1/4 left-1/2 h-2 w-2 rounded-full bg-orange-500 animate-bounce" />
               </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 border-y border-border/40 bg-muted/10">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-4xl md:text-5xl font-black text-foreground mb-2">{stat.value}</div>
                <div className="text-sm text-muted-foreground uppercase tracking-widest font-semibold">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-32">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-16">
            Engineered for <span className="text-gradient">Precision.</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, idx) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="group p-8 rounded-2xl border border-border/50 bg-background hover:bg-muted/30 hover:border-primary/30 transition-all text-left"
              >
                <div className={cn("h-12 w-12 rounded-xl flex items-center justify-center mb-6 transition-transform group-hover:scale-110", feature.bg)}>
                  <feature.icon className={cn("h-6 w-6", feature.color)} />
                </div>
                <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Agents / Workflow Section */}
      <section className="py-32 bg-muted/10 relative">
        <div className="container mx-auto px-4">
          <div className="flex flex-col lg:flex-row items-center gap-20">
            <div className="lg:w-1/2">
              <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-8">
                The Multi-Agent <br />
                <span className="text-primary text-gradient">Debate Chain</span>
              </h2>
              <p className="text-lg text-muted-foreground mb-12">
                Instead of single-pass inference, ARGUS orchestrates a competitive environment where specialized agents collaborate to refine beliefs through structured argumentation.
              </p>
              <div className="space-y-6">
                {agents.map((agent, idx) => (
                  <div key={agent.name} className="flex gap-4">
                    <div className="flex flex-col items-center">
                      <div className="h-10 w-10 rounded-full bg-background border border-border flex items-center justify-center shrink-0">
                        <agent.icon className="h-5 w-5 text-primary" />
                      </div>
                      {idx !== agents.length - 1 && <div className="w-px h-full bg-border mt-2" />}
                    </div>
                    <div className="pb-8">
                      <h4 className="text-lg font-bold flex items-center gap-2">
                        {agent.name}
                        <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-muted border border-border text-muted-foreground">
                          {agent.role}
                        </span>
                      </h4>
                      <p className="text-muted-foreground text-sm mt-1">{agent.task}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="lg:w-1/2 relative">
               {/* Visual Diagram Placeholder */}
               <div className="aspect-square bg-background rounded-3xl border border-border shadow-2xl p-8 flex items-center justify-center relative overflow-hidden">
                  <div className="absolute inset-0 bg-primary/5 pattern-grid-lg opacity-20" />
                  <div className="relative z-10 w-full h-full flex flex-col items-center justify-center gap-8">
                     {/* Diagram Mock */}
                     <div className="flex flex-col items-center gap-4">
                        <div className="px-6 py-3 bg-primary text-background font-bold rounded-lg shadow-lg">Moderator</div>
                        <div className="h-8 w-px bg-border dashed" />
                        <div className="flex gap-8">
                          <div className="px-4 py-3 bg-emerald-500 text-background font-bold rounded-lg shadow-lg">Specialist</div>
                          <div className="px-4 py-3 bg-red-500 text-background font-bold rounded-lg shadow-lg">Refuter</div>
                        </div>
                        <div className="h-8 w-px bg-border dashed" />
                        <div className="px-6 py-3 bg-purple-500 text-background font-bold rounded-lg shadow-lg">Jury</div>
                     </div>
                     <div className="text-xs text-muted-foreground font-mono bg-muted px-4 py-2 rounded-full border border-border">
                        Evidence Directed Debate Orchestration (EDDO)
                     </div>
                  </div>
               </div>
            </div>
          </div>
        </div>
      </section>

      {/* Code / Implementation Section */}
      <section className="py-32">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto rounded-3xl overflow-hidden border border-border shadow-2xl">
            <div className="bg-muted px-6 py-4 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Code2 className="h-5 w-5 text-primary" />
                <span className="text-sm font-bold tracking-tight">quick_start.py</span>
              </div>
              <Button variant="ghost" size="sm" className="h-8">Copy</Button>
            </div>
            <div className="bg-background p-8 font-mono text-sm sm:text-base overflow-x-auto leading-relaxed">
              <pre>
                <code className="block">
<span className="text-purple-400">from</span> argus <span className="text-purple-400">import</span> RDCOrchestrator, get_llm<br/><br/>
<span className="text-muted-foreground"># Initialize with any supported LLM</span><br/>
llm = <span className="text-blue-400">get_llm</span>(<span className="text-emerald-400">"openai"</span>, model=<span className="text-emerald-400">"gpt-4o"</span>)<br/><br/>
<span className="text-muted-foreground"># Run a debate on a proposition</span><br/>
orchestrator = <span className="text-blue-400">RDCOrchestrator</span>(llm=llm, max_rounds=<span className="text-orange-400">5</span>)<br/>
result = orchestrator.<span className="text-blue-400">debate</span>(<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-emerald-400">"Metformin increases longevity in non-diabetics"</span>,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;prior=<span className="text-orange-400">0.5</span><br/>
)<br/><br/>
<span className="text-blue-400">print</span>(<span className="text-emerald-400">f"Verdict: <span className="text-orange-400">{"{"}</span>result.verdict.label<span className="text-orange-400">{"}"}</span>"</span>)<br/>
<span className="text-blue-400">print</span>(<span className="text-emerald-400">f"Posterior: <span className="text-orange-400">{"{"}</span>result.verdict.posterior:.3f<span className="text-orange-400">{"}"}</span>"</span>)
                </code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Comparison Section */}
      <section className="py-32 bg-primary/5">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-16">
            Beyond the <span className="text-primary">Standard.</span>
          </h2>
          <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8 text-left">
            <div className="p-8 rounded-2xl border border-border bg-background">
              <h3 className="text-xl font-bold mb-4">Vs. LangChain</h3>
              <ul className="space-y-4 text-sm text-muted-foreground">
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" /> Opinionated debate structure</li>
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" /> Built-in Bayesian math</li>
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" /> Native decision-theory</li>
              </ul>
            </div>
            <div className="p-8 rounded-2xl border border-border bg-background">
              <h3 className="text-xl font-bold mb-4">Vs. LangGraph</h3>
              <ul className="space-y-4 text-sm text-muted-foreground">
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" /> Research-centric optimization</li>
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" /> Deterministic belief prop</li>
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" /> Calibrated uncertainty</li>
              </ul>
            </div>
            <div className="p-8 rounded-2xl border border-border bg-background">
              <h3 className="text-xl font-bold mb-4">Vs. AutoGen</h3>
              <ul className="space-y-4 text-sm text-muted-foreground">
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" /> Rigid provenance chain</li>
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" /> Evidence-directed flow</li>
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" /> Scientific rigor by default</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-32 relative">
        <div className="absolute inset-0 bg-primary/10 -z-10" />
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-6xl font-bold tracking-tight mb-8">
            Ready to <span className="text-gradient">Debate?</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-12">
            Build production-grade agentic systems that don't just "chat", but reason with evidence.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button size="lg" className="rounded-full px-12 py-8 text-xl h-auto group" asChild>
              <Link href="/docs">
                Start Building
                <ArrowRight className="ml-2 h-6 w-6 group-hover:translate-x-1 transition-transform" />
              </Link>
            </Button>
            <Button size="lg" variant="ghost" className="rounded-full px-12 py-8 text-xl h-auto" asChild>
              <Link href="/docs/concepts">Explore Concepts</Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
