# Deployment Guide

This guide covers deploying the ARGUS documentation website to various free hosting platforms.

## Table of Contents

- [Vercel (Recommended)](#vercel-recommended)
- [Netlify](#netlify)
- [Cloudflare Pages](#cloudflare-pages)
- [GitHub Pages](#github-pages)
- [Custom Domain Setup](#custom-domain-setup)

---

## Vercel (Recommended)

Vercel is the recommended platform as it's created by the Next.js team and offers the best integration.

### Prerequisites
- GitHub account
- Vercel account (free tier available)

### Deployment Steps

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/argus-docs.git
   git push -u origin main
   ```

2. **Import to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect Next.js settings

3. **Configure Build Settings** (auto-detected):
   - **Framework Preset**: Next.js
   - **Build Command**: `npm run build`
   - **Output Directory**: `out`
   - **Install Command**: `npm install`

4. **Deploy**:
   - Click "Deploy"
   - Wait for deployment to complete
   - Your site will be live at `https://your-project.vercel.app`

### Continuous Deployment

Vercel automatically deploys:
- **Production**: Pushes to `main` branch
- **Preview**: Pull requests and other branches

### Custom Domain

1. Go to Project Settings → Domains
2. Add your custom domain
3. Configure DNS records as instructed
4. SSL certificate is automatically provisioned

---

## Netlify

Netlify is another excellent option with a generous free tier.

### Deployment Steps

1. **Push to GitHub** (same as Vercel)

2. **Import to Netlify**:
   - Go to [netlify.com](https://netlify.com)
   - Click "Add new site" → "Import an existing project"
   - Connect to GitHub and select your repository

3. **Configure Build Settings**:
   - **Build command**: `npm run build`
   - **Publish directory**: `out`
   - **Node version**: 18 or higher

4. **Deploy**:
   - Click "Deploy site"
   - Your site will be live at `https://random-name.netlify.app`

### netlify.toml Configuration

Create a `netlify.toml` file in your project root:

```toml
[build]
  command = "npm run build"
  publish = "out"

[build.environment]
  NODE_VERSION = "18"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

### Custom Domain

1. Go to Site settings → Domain management
2. Add custom domain
3. Configure DNS records
4. SSL is automatically provisioned

---

## Cloudflare Pages

Cloudflare Pages offers excellent performance with their global CDN.

### Deployment Steps

1. **Push to GitHub** (same as above)

2. **Create Pages Project**:
   - Go to [Cloudflare Dashboard](https://dash.cloudflare.com)
   - Navigate to Pages
   - Click "Create a project"
   - Connect to GitHub

3. **Configure Build**:
   - **Framework preset**: Next.js (Static HTML Export)
   - **Build command**: `npm run build`
   - **Build output directory**: `out`

4. **Deploy**:
   - Click "Save and Deploy"
   - Your site will be live at `https://your-project.pages.dev`

### Environment Variables

If needed, add environment variables in:
Settings → Environment variables

### Custom Domain

1. Go to Custom domains
2. Add your domain
3. Cloudflare automatically handles DNS and SSL

---

## GitHub Pages

GitHub Pages is completely free but has some limitations with Next.js.

### Prerequisites

- GitHub repository
- Static export enabled in `next.config.js` (already configured)

### Deployment Steps

1. **Install gh-pages**:
   ```bash
   npm install --save-dev gh-pages
   ```

2. **Update package.json**:
   ```json
   {
     "scripts": {
       "deploy": "next build && touch out/.nojekyll && gh-pages -d out -t true"
     }
   }
   ```

3. **Deploy**:
   ```bash
   npm run deploy
   ```

4. **Configure GitHub Pages**:
   - Go to repository Settings → Pages
   - Source: Deploy from branch `gh-pages`
   - Your site will be live at `https://YOUR_USERNAME.github.io/argus-docs/`

### Base Path Configuration

If deploying to a repository (not user/org site), update `next.config.js`:

```javascript
const nextConfig = {
  output: 'export',
  basePath: '/argus-docs', // Your repo name
  images: {
    unoptimized: true,
  },
}
```

---

## Custom Domain Setup

### DNS Configuration

For all platforms, you'll need to configure DNS records:

#### Apex Domain (example.com)

**Vercel**:
```
A     @     76.76.21.21
```

**Netlify**:
```
A     @     75.2.60.5
```

**Cloudflare Pages**:
Cloudflare handles this automatically if your domain is on Cloudflare.

#### Subdomain (docs.example.com)

**All Platforms**:
```
CNAME docs your-project.platform.app
```

### SSL Certificates

All platforms automatically provision and renew SSL certificates via Let's Encrypt.

---

## CI/CD with GitHub Actions

For automated deployments, create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Vercel

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build
        run: npm run build
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
```

---

## Performance Optimization

### Lighthouse Scores

Target scores:
- **Performance**: 95+
- **Accessibility**: 100
- **Best Practices**: 100
- **SEO**: 100

### Optimization Tips

1. **Images**: Use Next.js Image component (already configured)
2. **Fonts**: Use `next/font` for optimal font loading
3. **Code Splitting**: Automatic with Next.js App Router
4. **Static Generation**: All pages are pre-rendered
5. **CDN**: All platforms provide global CDN

---

## Monitoring

### Vercel Analytics

Enable in Project Settings → Analytics (free tier available)

### Google Analytics

Add to `src/app/layout.tsx`:

```tsx
import Script from 'next/script'

export default function RootLayout({ children }) {
  return (
    <html>
      <head>
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'GA_MEASUREMENT_ID');
          `}
        </Script>
      </head>
      <body>{children}</body>
    </html>
  )
}
```

---

## Troubleshooting

### Build Failures

1. **Check Node version**: Ensure Node 18+ is being used
2. **Clear cache**: Delete `.next` and `node_modules`, reinstall
3. **Check logs**: Review build logs for specific errors

### 404 Errors

1. **Verify output directory**: Should be `out/`
2. **Check redirects**: Ensure proper redirect configuration
3. **Base path**: Verify `basePath` in `next.config.js` if using subdirectory

### Slow Builds

1. **Enable caching**: Most platforms cache `node_modules`
2. **Optimize images**: Compress images before committing
3. **Reduce dependencies**: Remove unused packages

---

## Support

For deployment issues:
- **Vercel**: [Vercel Support](https://vercel.com/support)
- **Netlify**: [Netlify Support](https://www.netlify.com/support/)
- **Cloudflare**: [Cloudflare Community](https://community.cloudflare.com/)

For ARGUS documentation issues:
- [GitHub Issues](https://github.com/Ronit26Mehta/argus-ai-debate/issues)
