import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { Header } from "@/components/layout/Header"
import { Footer } from "@/components/layout/Footer"
import { ThemeProvider } from "@/components/theme-provider"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
    title: "ARGUS - Agentic Research & Governance Unified System",
    description: "Production-ready multi-agent AI debate framework for evidence-based reasoning with structured argumentation, decision-theoretic planning, and full provenance tracking.",
    keywords: ["AI", "multi-agent", "debate", "reasoning", "LLM", "machine learning", "research"],
    authors: [{ name: "ARGUS Team" }],
    openGraph: {
        title: "ARGUS - Multi-Agent AI Debate Framework",
        description: "Production-ready framework for evidence-based reasoning",
        type: "website",
    },
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en" suppressHydrationWarning>
            <body className={inter.className}>
                <ThemeProvider
                    attribute="class"
                    defaultTheme="system"
                    enableSystem
                    disableTransitionOnChange
                >
                    <div className="flex min-h-screen flex-col">
                        <Header />
                        <main className="flex-1">{children}</main>
                        <Footer />
                    </div>
                </ThemeProvider>
            </body>
        </html>
    )
}
