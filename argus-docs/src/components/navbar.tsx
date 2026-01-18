'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { 
  Menu, 
  X, 
  Github, 
  BookOpen, 
  Sparkles, 
  ExternalLink,
  Command
} from 'lucide-react'

const navItems = [
  { href: '/docs/getting-started', label: 'Docs', icon: BookOpen },
  { href: '/docs/api-reference', label: 'API', icon: Command },
  { href: '/examples', label: 'Examples', icon: Sparkles },
]

export function Navbar() {
  const pathname = usePathname()
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4 md:px-8">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="relative">
            <div className="absolute inset-0 bg-primary/20 blur-xl rounded-full group-hover:bg-primary/30 transition-colors" />
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-violet-600 to-purple-600 shadow-lg shadow-primary/25">
              <svg viewBox="0 0 24 24" className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <circle cx="12" cy="5" r="2" />
                <circle cx="12" cy="19" r="2" />
                <circle cx="5" cy="12" r="2" />
                <circle cx="19" cy="12" r="2" />
                <line x1="12" y1="7" x2="12" y2="9" />
                <line x1="12" y1="15" x2="12" y2="17" />
                <line x1="7" y1="12" x2="9" y2="12" />
                <line x1="15" y1="12" x2="17" y2="12" />
              </svg>
            </div>
          </div>
          <span className="font-bold text-xl tracking-tight">Argus</span>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname.startsWith(item.href)
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  isActive 
                    ? 'bg-primary/10 text-primary' 
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                )}
              >
                <Icon className="w-4 h-4" />
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="flex items-center gap-3">
          <Link
            href="https://github.com/argus-ai/argus"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          >
            <Github className="w-4 h-4" />
            <span className="hidden lg:inline">GitHub</span>
            <ExternalLink className="w-3 h-3 opacity-50" />
          </Link>
          
          <Button asChild variant="glow" size="sm">
            <Link href="/docs/getting-started">
              Get Started
            </Link>
          </Button>

          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
      </div>

      {mobileOpen && (
        <div className="md:hidden border-t border-border/40 bg-background/95 backdrop-blur-xl">
          <nav className="container py-4 px-4 flex flex-col gap-2">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = pathname.startsWith(item.href)
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className={cn(
                    'flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors',
                    isActive 
                      ? 'bg-primary/10 text-primary' 
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  )}
                >
                  <Icon className="w-5 h-5" />
                  {item.label}
                </Link>
              )
            })}
            <Link
              href="https://github.com/argus-ai/argus"
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => setMobileOpen(false)}
              className="flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            >
              <Github className="w-5 h-5" />
              GitHub
              <ExternalLink className="w-3 h-3 opacity-50 ml-auto" />
            </Link>
          </nav>
        </div>
      )}
    </header>
  )
}
