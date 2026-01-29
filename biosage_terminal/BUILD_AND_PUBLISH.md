# BioSage Terminal - Build and Publish Guide

This guide provides step-by-step instructions for building and publishing the BioSage Terminal package to PyPI.

## Prerequisites

1. **Install build tools** (if not already installed):
   ```bash
   pip install --upgrade build twine
   ```

2. **PyPI Account Setup**:
   - Create an account at https://pypi.org
   - Create an account at https://test.pypi.org (for testing)
   - Generate an API token at https://pypi.org/manage/account/token/

3. **Configure PyPI credentials**:
   
   Create or update `~/.pypirc` (Linux/Mac) or `%USERPROFILE%\.pypirc` (Windows):
   ```ini
   [pypi]
   username = __token__
   password = pypi-YourActualTokenHere

   [testpypi]
   username = __token__
   password = pypi-YourTestTokenHere
   ```

## Step-by-Step Build and Publish Process

### Step 1: Pre-Build Checklist

Navigate to the project directory:
```bash
cd c:\ingester_ops\argus\biosage_terminal
```

Verify the following:
- [ ] `pyproject.toml` is properly configured
- [ ] `README.md` is up to date
- [ ] Version number in `pyproject.toml` and `biosage_terminal/__init__.py` match
- [ ] All dependencies are correctly specified
- [ ] LICENSE file exists (if applicable)

### Step 2: Clean Previous Builds

Remove old build artifacts:
```bash
# Windows PowerShell
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Windows CMD
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist biosage_terminal.egg-info rmdir /s /q biosage_terminal.egg-info

# Linux/Mac
rm -rf dist/ build/ *.egg-info
```

### Step 3: Build the Package

Build both source distribution and wheel:
```bash
python -m build
```

This creates:
- `dist/biosage-terminal-1.0.0.tar.gz` (source distribution)
- `dist/biosage_terminal-1.0.0-py3-none-any.whl` (wheel)

### Step 4: Verify the Build

Check the contents of the distribution:
```bash
# Windows
tar -tzf dist\biosage-terminal-1.0.0.tar.gz

# Linux/Mac
tar -tzf dist/biosage-terminal-1.0.0.tar.gz
```

Verify package metadata:
```bash
twine check dist/*
```

Expected output:
```
Checking dist/biosage-terminal-1.0.0.tar.gz: PASSED
Checking dist/biosage_terminal-1.0.0-py3-none-any.whl: PASSED
```

### Step 5: Test Upload (Recommended)

Upload to TestPyPI first:
```bash
twine upload --repository testpypi dist/*
```

Test installation from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ biosage-terminal
```

Note: `--extra-index-url` allows dependencies to be pulled from PyPI if not on TestPyPI.

### Step 6: Upload to Production PyPI

After successful testing, upload to the real PyPI:
```bash
twine upload dist/*
```

You'll see output like:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading biosage-terminal-1.0.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 kB • 00:01 • ?
Uploading biosage_terminal-1.0.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.0/45.0 kB • 00:01 • ?

View at:
https://pypi.org/project/biosage-terminal/1.0.0/
```

### Step 7: Verify Installation

Test the published package:
```bash
pip install biosage-terminal

# Or with LLM provider
pip install biosage-terminal[gemini]
```

Verify the installation:
```bash
biosage --version
biosage --check-api
```

## Alternative Upload Methods

### Using API Token Directly (No .pypirc)

```bash
twine upload --repository pypi dist/* --username __token__ --password pypi-YourActualTokenHere
```

### Interactive Upload (Prompt for Credentials)

```bash
twine upload dist/*
```

Twine will prompt for username and password.

## Common Issues and Solutions

### Issue 1: "File already exists"
**Solution**: You cannot re-upload the same version. Increment the version number in:
- `pyproject.toml`
- `biosage_terminal/__init__.py`

Then rebuild and upload.

### Issue 2: "Invalid distribution"
**Solution**: Run `twine check dist/*` to see specific errors. Common causes:
- Missing or malformed README.md
- Invalid classifiers
- Incorrect metadata in pyproject.toml

### Issue 3: "HTTPError: 403 Forbidden"
**Solution**: 
- Verify your API token is correct
- Ensure the token has upload permissions
- Check if the package name is already taken

### Issue 4: Dependencies not found during installation
**Solution**: Ensure all dependencies exist on PyPI and version constraints are correct.

## Version Management

### Semantic Versioning
Follow [SemVer](https://semver.org/):
- **MAJOR** (1.x.x): Breaking changes
- **MINOR** (x.1.x): New features, backward compatible
- **PATCH** (x.x.1): Bug fixes, backward compatible

### Updating Version

1. Update in `pyproject.toml`:
   ```toml
   version = "1.0.1"
   ```

2. Update in `biosage_terminal/__init__.py`:
   ```python
   __version__ = "1.0.1"
   ```

3. Rebuild and republish

## Continuous Integration (Future Enhancement)

Consider setting up GitHub Actions for automated publishing:

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Post-Publication Checklist

- [ ] Verify package appears on PyPI: https://pypi.org/project/biosage-terminal/
- [ ] Test installation in a fresh virtual environment
- [ ] Update project documentation with installation instructions
- [ ] Create a GitHub release (if using GitHub)
- [ ] Announce the release to users/team
- [ ] Tag the release in git: `git tag v1.0.0 && git push origin v1.0.0`

## Quick Reference Commands

```bash
# Complete build and upload workflow
cd c:\ingester_ops\argus\biosage_terminal
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
python -m build
twine check dist/*
twine upload --repository testpypi dist/*  # Test first
twine upload dist/*                         # Production
```

## Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PyPI Help](https://pypi.org/help/)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)

## Support

For issues specific to BioSage Terminal packaging:
- GitHub Issues: https://github.com/biosage/biosage-terminal/issues
- Email: team@biosage.ai
