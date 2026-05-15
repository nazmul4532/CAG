# Legacy Workspace

This folder contains the original notebook-era workspace and local artifacts.
Nothing was deleted; the old files were grouped here so the repository root can
stay focused on the new `mail_cag_project/` workspace.

## Layout

- `notebook_era/`: original notebooks, reports, archive, and patched attacker.
- `artifacts/`: local raw data, experiment outputs, model checkpoints, and
  reference documents. This folder is ignored by Git.
- `caches/`: local TextAttack/NLTK/Torch caches. This folder is ignored by Git.
- `root_links/`: compatibility symlinks for old root-level paths.

The new clean workspace reads existing legacy outputs directly from
`legacy_workspace/artifacts/`.
