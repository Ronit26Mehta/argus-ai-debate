'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { Navbar } from '@/components/navbar'
import { Button } from '@/components/ui/button'
import { CodeBlock } from '@/components/ui/code-block'
import { 
  ArrowRight, 
  Zap, 
  Brain, 
  Shield, 
  GitBranch, 
  Layers, 
  BarChart3,
  Users,
  Database,
  Cpu,
  CheckCircle2,
  Terminal,
  Workflow,
  Network,
  Target,
  Gauge,
  BookOpen,
  Github,
  Twitter,
  MessageSquare
} from 'lucide-react'

const quickStartCode = `from argus import Orchestrator, ArgusConfig
from argus.agents import SpecialistAgent, RefuterAgent, ModeratorAgent
from argus.decision import BayesianAggregator

# Initialize the orchestrator with your preferred LLM
config = ArgusConfig(
    llm_provider="openai",
    model="gpt-4-turbo",
    temperature=0.7
)

orchestrator = Orchestrator(config)

# Create a multi-agent debate for complex reasoning
result = await orchestrator.debate(
    question="Should we invest in quantum computing stocks?",
    agents=[
        SpecialistAgent(domain="finance"),
        SpecialistAgent(domain="technology"),
        RefuterAgent(),
        ModeratorAgent()
    ],
    rounds=3,
    aggregator=BayesianAggregator()
)

print(f"Decision: {result.decision}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Reasoning: {result.reasoning}")`

const features = [
  {
    icon: Brain,
    title: 'Cognitive DAG (CDAG)',
    description: 'Structured reasoning through directed acyclic graphs. Track claims, evidence, and logical dependencies with full provenance.',
    color: 'from-violet-500 to-purple-500',
  },
  {
    icon: Users,
    title: 'Multi-Agent Debate',
    description: 'Specialists, refuters, and moderators engage in adversarial deliberation to stress-test conclusions and uncover blind spots.',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    icon: BarChart3,
    title: 'Bayesian Aggregation',
    description: 'Combine agent opinions using principled probabilistic methods. Get calibrated confidence scores, not just answers.',
    color: 'from-emerald-500 to-green-500',
  },
  {
    icon: Database,
    title: 'RAG Integration',
    description: 'Built-in retrieval-augmented generation with semantic search, hybrid retrieval, and automatic context management.',
    color: 'from-orange-500 to-amber-500',
  },
  {
    icon: Shield,
    title: 'Human-in-the-Loop',
    description: 'Configurable checkpoints for human oversight. Review, approve, or redirect reasoning at critical decision points.',
    color: 'from-pink-500 to-rose-500',
  },
  {
    icon: Gauge,
    title: 'Evaluation Framework',
    description: 'Comprehensive benchmarks and metrics to measure reasoning quality, calibration, and decision accuracy.',
    color: 'from-indigo-500 to-blue-500',
  },
]

const architectureFeatures = [
  {
    icon: Network,
    title: 'EDDO Architecture',
    description: 'Evidence-Driven Dialectical Orchestration ensures every claim is backed by retrievable evidence.',
  },
  {
    icon: GitBranch,
    title: 'RDC Strategy',
    description: 'Retrieve-Debate-Consolidate pattern for systematic multi-hop reasoning across complex domains.',
  },
  {
    icon: Layers,
    title: 'Modular Design',
    description: 'Swap LLM providers, embedding models, and vector stores without changing your application code.',
  },
  {
    icon: Workflow,
    title: 'Durable Execution',
    description: 'Checkpoint and resume long-running debates. Never lose progress on complex reasoning tasks.',
  },
]

const comparisonData = [
  { feature: 'Multi-agent debate', argus: true, langchain: false, autogen: true, crewai: false },
  { feature: 'Bayesian aggregation', argus: true, langchain: false, autogen: false, crewai: false },
  { feature: 'Cognitive graphs (CDAG)', argus: true, langchain: false, autogen: false, crewai: false },
  { feature: 'Built-in RAG', argus: true, langchain: true, autogen: false, crewai: false },
  { feature: 'Human-in-the-loop', argus: true, langchain: false, autogen: true, crewai: true },
  { feature: 'Evaluation framework', argus: true, langchain: true, autogen: false, crewai: false },
  { feature: 'Provenance tracking', argus: true, langchain: false, autogen: false, crewai: false },
  { feature: 'Confidence calibration', argus: true, langchain: false, autogen: false, crewai: false },
]

const stats = [
  { value: '40%', label: 'Better accuracy on complex reasoning' },
  { value: '< 100ms', label: 'Agent coordination latency' },
  { value: '15+', label: 'Built-in tool integrations' },
  { value: '100%', label: 'Type-safe Python with Pydantic' },
]

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid opacity-50" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-primary/10 blur-[120px] rounded-full" />
        
        <div className="container relative px-4 md:px-8 pt-20 pb-32">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center max-w-4xl mx-auto"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-sm text-primary mb-8">
              <Zap className="w-4 h-4" />
              <span>v1.0.0 — Production Ready</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6">
              <span className="text-gradient">Adversarial Reasoning</span>
              <br />
              <span className="text-foreground">for AI Systems</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground mb-10 max-w-2xl mx-auto leading-relaxed">
              Build AI that thinks through debate. Multi-agent deliberation, Bayesian aggregation, 
              and cognitive graphs for decisions you can trust.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Button asChild variant="glow" size="xl">
                <Link href="/docs/getting-started" className="gap-2">
                  Get Started
                  <ArrowRight className="w-5 h-5" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="xl">
                <Link href="/examples" className="gap-2">
                  <Terminal className="w-5 h-5" />
                  View Examples
                </Link>
              </Button>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-20 max-w-4xl mx-auto"
          >
            <CodeBlock 
              code={quickStartCode} 
              language="python"
              filename="quickstart.py"
            />
          </motion.div>
        </div>
      </section>

      <section className="py-20 border-t border-border/40">
        <div className="container px-4 md:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="text-4xl md:text-5xl font-bold text-gradient mb-2">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24 bg-muted/30">
        <div className="container px-4 md:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-5xl font-bold mb-4">
              Everything you need for
              <span className="text-gradient"> intelligent reasoning</span>
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              A complete toolkit for building AI systems that reason through structured debate 
              and produce calibrated, trustworthy decisions.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="group relative p-6 rounded-2xl bg-card border border-border/50 hover:border-primary/50 transition-all hover:shadow-lg hover:shadow-primary/5"
              >
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${feature.color} mb-4`}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24">
        <div className="container px-4 md:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-5xl font-bold mb-4">
              Built on
              <span className="text-gradient"> proven principles</span>
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Argus implements cutting-edge research in multi-agent systems, 
              Bayesian inference, and structured reasoning.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            {architectureFeatures.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="flex gap-4 p-6 rounded-xl bg-card border border-border/50"
              >
                <div className="flex-shrink-0 p-3 rounded-lg bg-primary/10">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold mb-1">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24 bg-muted/30">
        <div className="container px-4 md:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-5xl font-bold mb-4">
              Why choose
              <span className="text-gradient"> Argus?</span>
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              See how Argus compares to other popular AI frameworks.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            viewport={{ once: true }}
            className="max-w-5xl mx-auto overflow-x-auto"
          >
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-4 px-4 font-medium">Feature</th>
                  <th className="text-center py-4 px-4 font-bold text-primary">Argus</th>
                  <th className="text-center py-4 px-4 font-medium text-muted-foreground">LangChain</th>
                  <th className="text-center py-4 px-4 font-medium text-muted-foreground">AutoGen</th>
                  <th className="text-center py-4 px-4 font-medium text-muted-foreground">CrewAI</th>
                </tr>
              </thead>
              <tbody>
                {comparisonData.map((row, index) => (
                  <tr key={row.feature} className="border-b border-border/50">
                    <td className="py-4 px-4 text-sm">{row.feature}</td>
                    <td className="text-center py-4 px-4">
                      {row.argus ? (
                        <CheckCircle2 className="w-5 h-5 text-green-500 mx-auto" />
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="text-center py-4 px-4">
                      {row.langchain ? (
                        <CheckCircle2 className="w-5 h-5 text-green-500 mx-auto" />
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="text-center py-4 px-4">
                      {row.autogen ? (
                        <CheckCircle2 className="w-5 h-5 text-green-500 mx-auto" />
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="text-center py-4 px-4">
                      {row.crewai ? (
                        <CheckCircle2 className="w-5 h-5 text-green-500 mx-auto" />
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </motion.div>
        </div>
      </section>

      <section className="py-24">
        <div className="container px-4 md:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="relative rounded-3xl bg-gradient-to-br from-violet-600 to-purple-600 p-12 md:p-20 text-center overflow-hidden"
          >
            <div className="absolute inset-0 bg-grid opacity-20" />
            <div className="relative">
              <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
                Ready to build smarter AI?
              </h2>
              <p className="text-lg md:text-xl text-white/80 mb-10 max-w-2xl mx-auto">
                Get started with Argus in minutes. Install with pip and build your first 
                multi-agent debate system today.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button asChild size="xl" className="bg-white text-primary hover:bg-white/90 shadow-xl">
                  <Link href="/docs/getting-started" className="gap-2">
                    <BookOpen className="w-5 h-5" />
                    Read the Docs
                  </Link>
                </Button>
                <Button asChild size="xl" variant="outline" className="border-white/30 text-white hover:bg-white/10">
                  <Link href="https://github.com/argus-ai/argus" className="gap-2">
                    <Github className="w-5 h-5" />
                    Star on GitHub
                  </Link>
                </Button>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      <footer className="border-t border-border/40 py-12">
        <div className="container px-4 md:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-purple-600">
                <svg viewBox="0 0 24 24" className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="3" />
                  <circle cx="12" cy="5" r="2" />
                  <circle cx="12" cy="19" r="2" />
                  <circle cx="5" cy="12" r="2" />
                  <circle cx="19" cy="12" r="2" />
                </svg>
              </div>
              <span className="font-semibold">Argus</span>
            </div>
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <Link href="/docs/getting-started" className="hover:text-foreground transition-colors">Documentation</Link>
              <Link href="/examples" className="hover:text-foreground transition-colors">Examples</Link>
              <Link href="https://github.com/argus-ai/argus" className="hover:text-foreground transition-colors">GitHub</Link>
            </div>
            <div className="flex items-center gap-4">
              <Link href="https://github.com/argus-ai/argus" className="text-muted-foreground hover:text-foreground transition-colors">
                <Github className="w-5 h-5" />
              </Link>
              <Link href="https://twitter.com/argus_ai" className="text-muted-foreground hover:text-foreground transition-colors">
                <Twitter className="w-5 h-5" />
              </Link>
              <Link href="https://discord.gg/argus" className="text-muted-foreground hover:text-foreground transition-colors">
                <MessageSquare className="w-5 h-5" />
              </Link>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-border/40 text-center text-sm text-muted-foreground">
            <p>© 2024 Argus. Open source under MIT License.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
