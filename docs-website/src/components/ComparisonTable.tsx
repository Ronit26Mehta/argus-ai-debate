'use client'

import { Check, X, Minus } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ComparisonColumn {
    name: string
    highlight?: boolean
}

export interface ComparisonRow {
    feature: string
    values: (boolean | string | number)[]
    description?: string
}

interface ComparisonTableProps {
    columns: ComparisonColumn[]
    rows: ComparisonRow[]
    className?: string
}

export function ComparisonTable({ columns, rows, className }: ComparisonTableProps) {
    const renderValue = (value: boolean | string | number) => {
        if (typeof value === 'boolean') {
            return value ? (
                <Check className="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
            ) : (
                <X className="h-5 w-5 text-red-500 dark:text-red-400 mx-auto" />
            )
        }
        if (value === '-' || value === 'N/A') {
            return <Minus className="h-5 w-5 text-muted-foreground mx-auto" />
        }
        return <span className="text-sm font-medium">{value}</span>
    }

    return (
        <div className={cn('my-8 overflow-hidden rounded-xl border shadow-lg', className)}>
            <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                    <thead>
                        <tr className="bg-muted/50">
                            <th className="px-6 py-4 text-left text-sm font-semibold">
                                Feature
                            </th>
                            {columns.map((column, idx) => (
                                <th
                                    key={idx}
                                    className={cn(
                                        'px-6 py-4 text-center text-sm font-semibold',
                                        column.highlight &&
                                        'bg-primary/10 dark:bg-primary/20 relative'
                                    )}
                                >
                                    {column.highlight && (
                                        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary to-purple-600" />
                                    )}
                                    <div className={cn(column.highlight && 'text-primary')}>
                                        {column.name}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row, rowIdx) => (
                            <tr
                                key={rowIdx}
                                className="border-t transition-colors hover:bg-muted/30"
                            >
                                <td className="px-6 py-4">
                                    <div>
                                        <div className="font-medium text-sm">{row.feature}</div>
                                        {row.description && (
                                            <div className="text-xs text-muted-foreground mt-1">
                                                {row.description}
                                            </div>
                                        )}
                                    </div>
                                </td>
                                {row.values.map((value, colIdx) => (
                                    <td
                                        key={colIdx}
                                        className={cn(
                                            'px-6 py-4 text-center',
                                            columns[colIdx].highlight &&
                                            'bg-primary/5 dark:bg-primary/10'
                                        )}
                                    >
                                        {renderValue(value)}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
