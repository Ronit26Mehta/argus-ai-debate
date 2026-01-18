'use client'

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Copy, Check } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'

interface CodeBlockProps {
  code: string
  language?: string
  filename?: string
  showLineNumbers?: boolean
  className?: string
}

export function CodeBlock({ 
  code, 
  language = 'python', 
  filename,
  showLineNumbers = true,
  className 
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={cn('relative group rounded-lg overflow-hidden border border-border/50', className)}>
      {filename && (
        <div className="flex items-center justify-between px-4 py-2 bg-muted/50 border-b border-border/50">
          <span className="text-sm text-muted-foreground font-mono">{filename}</span>
          <span className="text-xs text-muted-foreground uppercase">{language}</span>
        </div>
      )}
      <button
        onClick={copyToClipboard}
        className="absolute top-2 right-2 p-2 rounded-md bg-muted/80 hover:bg-muted opacity-0 group-hover:opacity-100 transition-opacity z-10"
        aria-label="Copy code"
      >
        {copied ? (
          <Check className="h-4 w-4 text-green-500" />
        ) : (
          <Copy className="h-4 w-4 text-muted-foreground" />
        )}
      </button>
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        showLineNumbers={showLineNumbers}
        customStyle={{
          margin: 0,
          padding: '1rem',
          background: 'hsl(224 71% 4%)',
          fontSize: '0.875rem',
        }}
        lineNumberStyle={{
          minWidth: '2.5em',
          paddingRight: '1em',
          color: 'hsl(215 16% 47%)',
          userSelect: 'none',
        }}
      >
        {code.trim()}
      </SyntaxHighlighter>
    </div>
  )
}

export function InlineCode({ children }: { children: React.ReactNode }) {
  return (
    <code className="px-1.5 py-0.5 rounded bg-muted font-mono text-sm text-primary">
      {children}
    </code>
  )
}
