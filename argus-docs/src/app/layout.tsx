import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-geist-sans',
})

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-geist-mono',
})

export const metadata: Metadata = {
  title: 'Argus - Adversarial Multi-Agent Debate Framework',
  description: 'Production-ready framework for building AI systems that reason through structured debate, Bayesian aggregation, and cognitive graphs. Achieve superior accuracy through multi-agent deliberation.',
  keywords: ['AI', 'multi-agent', 'debate', 'LLM', 'reasoning', 'Bayesian', 'cognitive graph', 'Python'],
  authors: [{ name: 'Argus Team' }],
  openGraph: {
    title: 'Argus - Adversarial Multi-Agent Debate Framework',
    description: 'Build AI systems that reason through structured debate and cognitive graphs',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased`}>
        {children}
      </body>
    </html>
  )
}
