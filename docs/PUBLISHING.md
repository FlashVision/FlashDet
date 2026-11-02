# Publishing to PyPI from GitHub

The repository can publish to PyPI automatically after CI passes.

## How it works

`.github/workflows/publish.yml` runs when:

- a `CI` workflow finishes successfully on `main`
- a GitHub release is published
- the workflow is started manually from the Actions tab

Before publishing it reads the version from `pyproject.toml` and checks PyPI.
If that exact version already exists, the publish job is skipped. This makes
normal pushes safe: only a version bump triggers a new PyPI release.

Automatic publishing is opt-in behind a repository variable so the workflow
does not fail before PyPI is configured. After completing one of the setup
options below, add:

| Type | Name | Value |
|---|---|---|
| Repository variable | `PYPI_AUTO_PUBLISH` | `true` |

Without the variable you can still publish manually from the Actions tab.

## One-time setup

Choose one of these two options.

### Option A — PyPI trusted publishing (recommended, no token)

In PyPI, open the `flashdet` project → **Publishing** → **Add a new publisher**
and enter:

| Field | Value |
|---|---|
| Owner | `FlashVision` |
| Repository name | `FlashDet` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Then create a GitHub environment named `pypi` in
**FlashVision/FlashDet → Settings → Environments** (no secrets are required).

The workflow uses OpenID Connect, so no long-lived PyPI token is stored in
GitHub.

### Option B — API token

Add a repository secret:

**Settings → Secrets and variables → Actions → New repository secret**

| Name | Value |
|---|---|
| `PYPI_API_TOKEN` | a PyPI API token with upload access to `flashdet` |

The workflow uses the token automatically when it is present. Otherwise it
falls back to trusted publishing.

## Release a new version

```bash
# 1. Bump the version
#    pyproject.toml:  project.version
#    flashdet/__init__.py: fallback __version__

# 2. Update docs/CHANGELOG.md

# 3. Commit and push to main
git add pyproject.toml flashdet/__init__.py docs/CHANGELOG.md
git commit -m "Release 1.3.1"
git push origin main
```

Then:

1. GitHub runs CI.
2. If CI passes, `Publish to PyPI` checks `flashdet==1.3.1`.
3. If the version is new, it builds the wheel/sdist and uploads it.
4. PyPI and `pip install flashdet` update automatically.

You can also run the publish workflow manually from
**Actions → Publish to PyPI → Run workflow**.

## Important

- PyPI versions are immutable. Always bump the version before publishing.
- Existing versions are skipped, so re-running the workflow is safe.
- If you use trusted publishing, the environment name must match on PyPI and in
  the workflow (`pypi`).
