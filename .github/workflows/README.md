# GitHub Pages Deployment

This directory contains the GitHub Actions workflow for deploying the ARGUS documentation website to GitHub Pages.

## Workflow: deploy-docs.yml

Automatically deploys the docs-website to GitHub Pages when changes are pushed to the main branch.

### Triggers
- Push to `main` or `master` branch (only when files in `docs-website/` change)
- Manual workflow dispatch

### What it does
1. Checks out the repository
2. Sets up Node.js 20
3. Installs dependencies
4. Builds the Next.js static site
5. Uploads the build artifact
6. Deploys to GitHub Pages

## Setup Instructions

### 1. Enable GitHub Pages
1. Go to your repository settings
2. Navigate to "Pages" in the left sidebar
3. Under "Build and deployment":
   - Source: **GitHub Actions**
4. Save the settings

### 2. Push the workflow
```bash
git add .github/workflows/deploy-docs.yml
git commit -m "Add GitHub Pages deployment workflow"
git push origin main
```

### 3. Monitor deployment
- Go to the "Actions" tab in your repository
- Watch the "Deploy to GitHub Pages" workflow run
- Once complete, your site will be available at: `https://<username>.github.io/<repository>/`

## Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file to `docs-website/public/`:
   ```
   docs.yoursite.com
   ```

2. Configure DNS:
   - Add a CNAME record pointing to `<username>.github.io`
   - Or add A records for GitHub Pages IPs

3. In repository settings > Pages, add your custom domain

## Troubleshooting

### Build fails
- Check the Actions tab for error logs
- Ensure all dependencies are in `package.json`
- Test build locally: `cd docs-website && npm run build`

### 404 errors
- Ensure `output: 'export'` is in `next.config.js`
- Check that `trailingSlash: true` is set
- Verify the base path if using a repository subdirectory

### Assets not loading
- Ensure `images.unoptimized: true` in `next.config.js`
- Check that all asset paths are relative

## Local Testing

Test the production build locally:

```bash
cd docs-website
npm run build
npx serve@latest out
```

Visit `http://localhost:3000` to preview the static site.
