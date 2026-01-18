"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Search as SearchIcon, Github, Menu, X, Shield, Cpu, BookOpen } from "lucide-react";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "./theme-toggle";
import { SearchDialog } from "./search-dialog";

const navLinks = [
  { name: "Documentation", href: "/docs", icon: BookOpen },
  { name: "Core Concepts", href: "/docs/concepts/rdc", icon: Cpu },
  { name: "API Reference", href: "/docs/api/orchestrator", icon: Shield },
];

export function Navbar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);

  // Keyboard shortcut for search
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setSearchOpen((open) => !open);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/60 backdrop-blur-xl">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="relative h-8 w-8 bg-primary rounded-lg flex items-center justify-center overflow-hidden group-hover:scale-110 transition-transform">
              <Shield className="h-5 w-5 text-background" />
              <motion.div
                className="absolute inset-0 bg-white/20"
                initial={{ x: "-100%" }}
                animate={{ x: "100%" }}
                transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
              />
            </div>
            <span className="text-xl font-bold tracking-tighter text-foreground group-hover:text-primary transition-colors">
              ARGUS
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "px-4 py-2 text-sm font-medium rounded-full transition-all hover:bg-muted",
                  pathname.startsWith(link.href) ? "text-primary bg-primary/10" : "text-muted-foreground hover:text-foreground"
                )}
              >
                {link.name}
              </Link>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div 
            className="hidden md:flex relative group cursor-pointer"
            onClick={() => setSearchOpen(true)}
          >
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
            <div className="pl-10 pr-4 py-1.5 w-64 bg-muted/50 border border-border/50 rounded-full text-sm text-muted-foreground flex justify-between items-center group-hover:border-primary/30 transition-all">
              <span>Search...</span>
              <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono border border-border/50">
                ⌘K
              </kbd>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Button variant="ghost" size="icon" className="rounded-full" asChild>
              <Link href="https://github.com/Ronit26Mehta/argus-ai-debate" target="_blank">
                <Github className="h-5 w-5" />
              </Link>
            </Button>
            <Button className="rounded-full hidden sm:flex" asChild>
              <Link href="/docs">Get Started</Link>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden rounded-full"
              onClick={() => setIsOpen(!isOpen)}
            >
              {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>
        </div>
      </div>

      <SearchDialog open={searchOpen} onOpenChange={setSearchOpen} />

      {/* Mobile Menu */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="md:hidden border-b border-border/40 bg-background/95 backdrop-blur-xl px-4 py-6 flex flex-col gap-4"
        >
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="flex items-center gap-3 text-lg font-medium p-2 rounded-lg hover:bg-muted"
              onClick={() => setIsOpen(false)}
            >
              <link.icon className="h-5 w-5 text-primary" />
              {link.name}
            </Link>
          ))}
          <Button 
            variant="outline" 
            className="w-full py-6 text-lg rounded-xl flex items-center justify-between"
            onClick={() => {
              setIsOpen(false);
              setSearchOpen(true);
            }}
          >
            <span className="flex items-center gap-2"><SearchIcon className="h-5 w-5" /> Search</span>
            <kbd className="text-xs opacity-50">⌘K</kbd>
          </Button>
          <Button className="w-full py-6 text-lg rounded-xl mt-2" asChild>
            <Link href="/docs">Get Started</Link>
          </Button>
        </motion.div>
      )}
    </nav>
  );
}
