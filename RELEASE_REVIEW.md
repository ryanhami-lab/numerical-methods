# Private review copy

This project is prepared for review before any public release. Repository
visibility should remain **private** until its owner chooses to make it public.
No license has been selected; no license grant is implied by this review copy.

## Start here

- [README.md](README.md): installation, usage, supported scope and limitations.
- [VALIDATION.md](VALIDATION.md): requirement-to-evidence map and study findings.
- [COMPLETION.md](COMPLETION.md): verification outcomes and reproduction commands.
- [results/verification_summary.json](results/verification_summary.json): shareable validation summary.

## Privacy preparation

Personal filesystem paths have been replaced with portable setup commands.
Raw local logs, build archives, cache directories, notebook checkpoints and
private review working files are excluded from Git. Raw verification logs and
private working files are also excluded from source distributions. The legacy
notebook is included with saved outputs and execution counts cleared.

`python scripts/audit_release.py` checks the exact staged snapshot for common
credential formats, private keys, credential assignments, authenticated URLs,
email addresses, personal filesystem paths, notebook outputs and local-only
files. It reports locations without printing matched values. Pattern checks
cannot establish that every possible secret is absent; review new files before
future uploads. Scientific environment metadata (versions, CPU, BLAS and thread
settings) is retained to interpret the measurements.

## Before making the project public

1. Review the examples, scope and measured claims.
2. Choose a license and confirm the copyright holder.
3. Review the GitHub Actions results after the private upload.
4. Explicitly decide to make the repository public; that is a separate action.

This preparation does not create a package-index publication, a public release,
or a release tag.
