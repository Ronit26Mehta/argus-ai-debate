# ARGUS Documentation Website

Official documentation website for **ARGUS** - Agentic Research & Governance Unified System.

## Overview

This is a Next.js 14 documentation website built with:
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui (Radix UI + Tailwind)
- **Theme**: Dark/Light mode support with next-themes
- **Typography**: Inter font family

## Features

- 🎨 **Stunning Design**: Modern UI with glassmorphism effects, gradients, and smooth animations
- 📱 **Responsive**: Mobile-first design that works on all devices
- 🌓 **Dark Mode**: Full dark mode support with theme toggle
- 🔍 **Comprehensive Docs**: Complete documentation for all 17 modules
- 💻 **Code Examples**: Syntax-highlighted code blocks with copy functionality
- 🚀 **Fast**: Optimized for performance with static export
- ♿ **Accessible**: Built with accessibility in mind using Radix UI primitives

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. **Install dependencies**:
```bash
npm install
```

2. **Run development server**:
```bash
npm run dev
```

3. **Open your browser**:
Navigate to [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
```

This creates an optimized production build in the `out/` directory.

### Type Checking

```bash
npm run type-check
```

### Linting

```bash
npm run lint
```

## Project Structure

```
docs-website/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── docs/              # Documentation pages
│   │   │   ├── getting-started/
│   │   │   ├── core-concepts/
│   │   │   ├── modules/       # Module documentation
│   │   │   └── ...
│   │   ├── api-reference/     # API reference pages
│   │   ├── tutorials/         # Tutorial pages
│   │   ├── comparison/        # Comparison page
│   │   ├── layout.tsx         # Root layout
│   │   ├── page.tsx           # Landing page
│   │   └── globals.css        # Global styles
│   ├── components/
│   │   ├── layout/            # Layout components
│   │   │   ├── Header.tsx
│   │   │   ├── Footer.tsx
│   │   │   └── DocsSidebar.tsx
│   │   ├── ui/                # shadcn/ui components
│   │   └── ...
│   └── lib/
│       ├── utils.ts           # Utility functions
│       └── design-system.ts   # Design tokens
├── public/                    # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── next.config.js
└── README.md
```

## Documentation Structure

The documentation is organized into several main sections:

1. **Getting Started**: Installation, quick start, configuration
2. **Core Concepts**: RDC, C-DAG, multi-agent systems
3. **Modules**: Documentation for all 17 core modules
4. **Integrations**: LLM providers, embeddings, tools
5. **API Reference**: Complete API documentation
6. **Tutorials**: Step-by-step guides
7. **Comparison**: Comparison with other frameworks

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions for:
- Vercel (recommended)
- Netlify
- Cloudflare Pages
- GitHub Pages

## Contributing

Contributions are welcome! Please see the main [CONTRIBUTING.md](https://github.com/Ronit26Mehta/argus-ai-debate/blob/main/CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](https://github.com/Ronit26Mehta/argus-ai-debate/blob/main/LICENSE)

## Links

- **Main Repository**: https://github.com/Ronit26Mehta/argus-ai-debate
- **PyPI Package**: https://pypi.org/project/argus-debate-ai/
- **Documentation**: https://argus-docs.vercel.app (after deployment)

## Version

Current version: **1.0.0** (Documentation website)
ARGUS version: **2.0.0**
