'use client'

import React, { useState } from 'react'
import { Check, Copy } from 'lucide-react'
import { codeToHtml } from 'shiki'

interface CodeBlockProps {
    code: string
    language: string
    filename?: string
    showLineNumbers?: boolean
    highlightLines?: number[]
}

export function CodeBlock({
    code,
    language,
    filename,
    showLineNumbers = false,
    highlightLines = [],
}: CodeBlockProps) {
    const [copied, setCopied] = useState(false)
    const [html, setHtml] = useState<string>('')

    // Highlight code with Shiki
    React.useEffect(() => {
        async function highlightCode() {
            const highlighted = await codeToHtml(code, {
                lang: language,
                theme: 'github-dark',
            })
            setHtml(highlighted)
        }
        highlightCode()
    }, [code, language])

    const copyToClipboard = async () => {
        await navigator.clipboard.writeText(code)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
    }

    return (
        <div className="group relative my-6">
            {/* Header with filename and copy button */}
            {(filename || true) && (
                <div className="flex items-center justify-between rounded-t-xl bg-slate-800/90 px-4 py-2 border-b border-slate-700/50">
                    {filename && (
                        <span className="text-sm font-mono text-slate-300">
                            {filename}
                        </span>
                    )}
                    <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-slate-400 uppercase">
                            {language}
                        </span>
                        <button
                            onClick={copyToClipboard}
                            className="flex items-center gap-1.5 rounded-md bg-slate-700/50 px-2.5 py-1.5 text-xs font-medium text-slate-300 transition-all hover:bg-slate-700 hover:text-white opacity-0 group-hover:opacity-100"
                            aria-label="Copy code"
                        >
                            {copied ? (
                                <>
                                    <Check className="h-3.5 w-3.5" />
                                    Copied!
                                </>
                            ) : (
                                <>
                                    <Copy className="h-3.5 w-3.5" />
                                    Copy
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}

            {/* Code content */}
            <div className="relative overflow-x-auto">
                {html ? (
                    <div
                        dangerouslySetInnerHTML={{ __html: html }}
                        className="code-block-content"
                    />
                ) : (
                    <pre className={filename ? 'rounded-t-none' : ''}>
                        <code className={`language-${language}`}>{code}</code>
                    </pre>
                )}
            </div>
        </div>
    )
}

// Simple inline code component
export function InlineCode({ children }: { children: React.ReactNode }) {
    return <code className="inline-code">{children}</code>
}
