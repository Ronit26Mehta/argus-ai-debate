# BioSage Terminal - Quick Build & Publish Steps

## 🚀 Prerequisites (One-Time Setup)

```bash
# Install build tools
pip install --upgrade build twine

# Get PyPI API token from: https://pypi.org/manage/account/token/
# Save in ~/.pypirc (Linux/Mac) or %USERPROFILE%\.pypirc (Windows)
```

## 📦 Build & Publish Commands

### Option 1: Automated Script (Recommended)

**Windows PowerShell:**
```powershell
cd c:\ingester_ops\argus\biosage_terminal
.\build_package.ps1
```

**Windows CMD:**
```cmd
cd c:\ingester_ops\argus\biosage_terminal
build_package.bat
```

Then upload:
```bash
# Test upload first
twine upload --repository testpypi dist/*

# Production upload
twine upload dist/*
```

### Option 2: Manual Commands

```bash
# 1. Navigate to project
cd c:\ingester_ops\argus\biosage_terminal

# 2. Clean previous builds
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# 3. Build the package
python -m build

# 4. Verify the build
twine check dist/*

# 5. Upload to TestPyPI (optional but recommended)
twine upload --repository testpypi dist/*

# 6. Upload to PyPI
twine upload dist/*
```

## 🔍 Verification

```bash
# Test installation
pip install biosage-terminal

# Or with Gemini provider
pip install biosage-terminal[gemini]

# Verify it works
biosage --version
biosage --check-api
```

## 📋 Expected Output

After running `python -m build`:
```
dist/
  ├── biosage-terminal-1.0.0.tar.gz
  └── biosage_terminal-1.0.0-py3-none-any.whl
```

After `twine check dist/*`:
```
Checking dist/biosage-terminal-1.0.0.tar.gz: PASSED
Checking dist/biosage_terminal-1.0.0-py3-none-any.whl: PASSED
```

After `twine upload dist/*`:
```
Uploading biosage-terminal-1.0.0.tar.gz
Uploading biosage_terminal-1.0.0-py3-none-any.whl

View at: https://pypi.org/project/biosage-terminal/1.0.0/
```

## ⚠️ Important Notes

1. **Version numbers** in `pyproject.toml` and `biosage_terminal/__init__.py` must match
2. **Cannot re-upload** the same version - increment version if needed
3. **Test first** using TestPyPI before production upload
4. **API token** must have upload permissions

## 🔧 Troubleshooting

**"File already exists"**
→ Increment version in both files and rebuild

**"Invalid distribution"**
→ Run `twine check dist/*` for details

**"403 Forbidden"**
→ Verify API token and permissions

## 📚 Full Documentation

See `BUILD_AND_PUBLISH.md` for complete documentation.
