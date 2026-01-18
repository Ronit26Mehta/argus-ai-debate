'use client'

import { AlertCircle, AlertTriangle, CheckCircle2, Info, Lightbulb, Zap } from 'lucide-react'
import { cn } from '@/lib/utils'

type CalloutVariant = 'info' | 'warning' | 'danger' | 'success' | 'tip' | 'note'

interface CalloutProps {
    variant?: CalloutVariant
    title?: string
    children: React.ReactNode
    className?: string
}

const variantConfig = {
    info: {
        icon: Info,
        className: 'bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-900/50',
        iconClassName: 'text-blue-600 dark:text-blue-400',
        titleClassName: 'text-blue-900 dark:text-blue-300',
    },
    warning: {
        icon: AlertTriangle,
        className: 'bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-900/50',
        iconClassName: 'text-amber-600 dark:text-amber-400',
        titleClassName: 'text-amber-900 dark:text-amber-300',
    },
    danger: {
        icon: AlertCircle,
        className: 'bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-900/50',
        iconClassName: 'text-red-600 dark:text-red-400',
        titleClassName: 'text-red-900 dark:text-red-300',
    },
    success: {
        icon: CheckCircle2,
        className: 'bg-green-50 dark:bg-green-950/30 border-green-200 dark:border-green-900/50',
        iconClassName: 'text-green-600 dark:text-green-400',
        titleClassName: 'text-green-900 dark:text-green-300',
    },
    tip: {
        icon: Lightbulb,
        className: 'bg-purple-50 dark:bg-purple-950/30 border-purple-200 dark:border-purple-900/50',
        iconClassName: 'text-purple-600 dark:text-purple-400',
        titleClassName: 'text-purple-900 dark:text-purple-300',
    },
    note: {
        icon: Zap,
        className: 'bg-indigo-50 dark:bg-indigo-950/30 border-indigo-200 dark:border-indigo-900/50',
        iconClassName: 'text-indigo-600 dark:text-indigo-400',
        titleClassName: 'text-indigo-900 dark:text-indigo-300',
    },
}

export function Callout({
    variant = 'info',
    title,
    children,
    className,
}: CalloutProps) {
    const config = variantConfig[variant]
    const Icon = config.icon

    return (
        <div
            className={cn(
                'my-6 rounded-xl border-l-4 p-6 shadow-sm transition-all duration-300 hover:shadow-md',
                config.className,
                className
            )}
        >
            <div className="flex gap-4">
                <div className="flex-shrink-0">
                    <Icon className={cn('h-6 w-6', config.iconClassName)} />
                </div>
                <div className="flex-1 space-y-2">
                    {title && (
                        <h4 className={cn('font-semibold text-lg', config.titleClassName)}>
                            {title}
                        </h4>
                    )}
                    <div className="text-sm leading-relaxed text-foreground/80">
                        {children}
                    </div>
                </div>
            </div>
        </div>
    )
}
