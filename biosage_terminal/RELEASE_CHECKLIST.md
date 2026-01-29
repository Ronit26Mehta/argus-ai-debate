# BioSage Terminal - Pre-Release Checklist

## 📋 Pre-Build Verification

- [ ] Version numbers match:
  - [ ] `pyproject.toml` → `version = "1.0.0"`
  - [ ] `biosage_terminal/__init__.py` → `__version__ = "1.0.0"`

- [ ] Documentation is up-to-date:
  - [ ] README.md reflects current features
  - [ ] Installation instructions are correct
  - [ ] Dependencies are listed accurately

- [ ] Metadata is correct:
  - [ ] Author information in `pyproject.toml`
  - [ ] Project URLs are valid
  - [ ] Keywords are relevant
  - [ ] Classifiers are appropriate

- [ ] Code quality:
  - [ ] All imports are working
  - [ ] No syntax errors
  - [ ] CLI entry point works (`biosage --version`)

- [ ] Files are properly included:
  - [ ] LICENSE file exists (if applicable)
  - [ ] MANIFEST.in includes all necessary files
  - [ ] No sensitive data in package

## 🔧 Build Process

- [ ] Navigate to project directory:
  ```bash
  cd c:\ingester_ops\argus\biosage_terminal
  ```

- [ ] Clean previous builds:
  ```powershell
  Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
  ```

- [ ] Build package:
  ```bash
  python -m build
  ```

- [ ] Verify build output:
  - [ ] `dist/biosage-terminal-1.0.0.tar.gz` created
  - [ ] `dist/biosage_terminal-1.0.0-py3-none-any.whl` created

- [ ] Check distribution:
  ```bash
  twine check dist/*
  ```
  - [ ] All checks PASSED

## 🧪 Testing

- [ ] Upload to TestPyPI:
  ```bash
  twine upload --repository testpypi dist/*
  ```

- [ ] Test installation from TestPyPI:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ biosage-terminal
  ```

- [ ] Verify installation works:
  ```bash
  biosage --version
  biosage --check-api
  ```

- [ ] Test basic functionality:
  - [ ] Application launches without errors
  - [ ] Main screens are accessible
  - [ ] No import errors

- [ ] Uninstall test version:
  ```bash
  pip uninstall biosage-terminal
  ```

## 🚀 Production Release

- [ ] Upload to PyPI:
  ```bash
  twine upload dist/*
  ```

- [ ] Verify on PyPI:
  - [ ] Visit: https://pypi.org/project/biosage-terminal/
  - [ ] Package information is correct
  - [ ] README renders properly

- [ ] Test production installation:
  ```bash
  pip install biosage-terminal
  biosage --version
  ```

## 📝 Post-Release

- [ ] Update CHANGELOG.md with release notes
- [ ] Create Git tag:
  ```bash
  git tag v1.0.0
  git push origin v1.0.0
  ```

- [ ] Create GitHub Release (if using GitHub)

- [ ] Update project documentation

- [ ] Announce release:
  - [ ] Team notification
  - [ ] Social media (if applicable)
  - [ ] Documentation site update

## 🔄 For Future Releases

- [ ] Increment version number (follow SemVer):
  - Major (1.x.x) - Breaking changes
  - Minor (x.1.x) - New features
  - Patch (x.x.1) - Bug fixes

- [ ] Update both version locations:
  - `pyproject.toml`
  - `biosage_terminal/__init__.py`

- [ ] Document changes in CHANGELOG.md

- [ ] Repeat this checklist

## 📞 Support

Issues or questions:
- GitHub: https://github.com/biosage/biosage-terminal/issues
- Email: team@biosage.ai

## ✅ Quick Status

Current Version: **1.0.0**
Last Build: _____________
Last Upload: _____________
PyPI Status: _____________
