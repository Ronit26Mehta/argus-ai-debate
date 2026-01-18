"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, X, Github, Search } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/theme-toggle"

export function Header() {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
    const pathname = usePathname()

    const navigation = [
        { name: "Docs", href: "/docs/getting-started" },
        { name: "API Reference", href: "/api-reference" },
        { name: "Tutorials", href: "/tutorials" },
        { name: "Comparison", href: "/comparison" },
    ]

    return (
        <header className="sticky top-0 z-50 w-full border-b glass">
            <div className="container flex h-16 items-center justify-between">
                {/* Logo */}
                <Link href="/" className="flex items-center space-x-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-purple-600">
                        <span className="text-lg font-bold text-white">A</span>
                    </div>
                    <span className="text-xl font-bold gradient-text">ARGUS</span>
                </Link>

                {/* Desktop Navigation */}
                <nav className="hidden md:flex items-center space-x-6">
                    {navigation.map((item) => (
                        <Link
                            key={item.name}
                            href={item.href}
                            className={`text-sm font-medium transition-colors hover:text-primary ${pathname?.startsWith(item.href)
                                    ? "text-foreground"
                                    : "text-muted-foreground"
                                }`}
                        >
                            {item.name}
                        </Link>
                    ))}
                </nav>

                {/* Right Actions */}
                <div className="flex items-center space-x-4">
                    <Button variant="ghost" size="icon" className="hidden md:flex">
                        <Search className="h-5 w-5" />
                    </Button>

                    <Link
                        href="https://github.com/Ronit26Mehta/argus-ai-debate"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        <Button variant="ghost" size="icon">
                            <Github className="h-5 w-5" />
                        </Button>
                    </Link>

                    <ThemeToggle />

                    {/* Mobile Menu Button */}
                    <Button
                        variant="ghost"
                        size="icon"
                        className="md:hidden"
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    >
                        {mobileMenuOpen ? (
                            <X className="h-5 w-5" />
                        ) : (
                            <Menu className="h-5 w-5" />
                        )}
                    </Button>
                </div>
            </div>

            {/* Mobile Menu */}
            {mobileMenuOpen && (
                <div className="md:hidden border-t glass">
                    <nav className="container py-4 flex flex-col space-y-3">
                        {navigation.map((item) => (
                            <Link
                                key={item.name}
                                href={item.href}
                                className={`text-sm font-medium transition-colors hover:text-primary ${pathname?.startsWith(item.href)
                                        ? "text-foreground"
                                        : "text-muted-foreground"
                                    }`}
                                onClick={() => setMobileMenuOpen(false)}
                            >
                                {item.name}
                            </Link>
                        ))}
                    </nav>
                </div>
            )}
        </header>
    )
}
