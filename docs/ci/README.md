# GitHub Actions template

[github-actions.yml](github-actions.yml) defines a Linux regression matrix for
Python 3.9, 3.10, 3.11, and 3.12. It installs the pinned requirements, checks the
release file manifest, and runs all 28 software regressions. Official actions
are pinned to their verified release commits, and the workflow token has only
read access to repository contents.

The template is not active in v0.2.0. The credential used to publish this release
can push source code but cannot create workflows because it lacks the OAuth
`workflow` scope. No Linux CI result is claimed. The recorded local tests use
Python 3.9.6 on macOS arm64.

To enable CI with an appropriately authorized account:

1. Copy `docs/ci/github-actions.yml` to `.github/workflows/tests.yml`.
2. Refresh `SHA256SUMS.txt` for changed tracked files and include the new workflow
   if it is part of the release manifest.
3. Commit and push the workflow, then inspect every matrix job in the Actions tab.

Only successful runs establish support for those environments. The source
release remains usable without GitHub Actions.
