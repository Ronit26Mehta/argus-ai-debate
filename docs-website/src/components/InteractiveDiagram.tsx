'use client'

import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface InteractiveDiagramProps {
    title?: string
    children: React.ReactNode
    className?: string
}

export function InteractiveDiagram({ title, children, className }: InteractiveDiagramProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className={cn(
                'my-8 rounded-xl border bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-8 shadow-lg',
                className
            )}
        >
            {title && (
                <h3 className="mb-6 text-xl font-semibold text-center gradient-text">
                    {title}
                </h3>
            )}
            <div className="flex items-center justify-center">{children}</div>
        </motion.div>
    )
}

// Mermaid diagram component
interface MermaidDiagramProps {
    chart: string
    title?: string
}

export function MermaidDiagram({ chart, title }: MermaidDiagramProps) {
    return (
        <InteractiveDiagram title={title}>
            <div className="mermaid w-full">{chart}</div>
        </InteractiveDiagram>
    )
}

// Simple flow diagram component
interface FlowNode {
    id: string
    label: string
    type?: 'start' | 'process' | 'decision' | 'end'
}

interface FlowEdge {
    from: string
    to: string
    label?: string
}

interface FlowDiagramProps {
    nodes: FlowNode[]
    edges: FlowEdge[]
    title?: string
}

export function FlowDiagram({ nodes, edges, title }: FlowDiagramProps) {
    const getNodeColor = (type?: string) => {
        switch (type) {
            case 'start':
                return 'bg-green-100 dark:bg-green-900/30 border-green-500'
            case 'end':
                return 'bg-red-100 dark:bg-red-900/30 border-red-500'
            case 'decision':
                return 'bg-amber-100 dark:bg-amber-900/30 border-amber-500'
            default:
                return 'bg-blue-100 dark:bg-blue-900/30 border-blue-500'
        }
    }

    return (
        <InteractiveDiagram title={title}>
            <div className="flex flex-col gap-4 w-full max-w-2xl">
                {nodes.map((node, idx) => (
                    <motion.div
                        key={node.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className={cn(
                            'rounded-lg border-2 px-6 py-4 text-center font-medium shadow-md transition-all hover:shadow-lg',
                            getNodeColor(node.type)
                        )}
                    >
                        {node.label}
                    </motion.div>
                ))}
            </div>
        </InteractiveDiagram>
    )
}
