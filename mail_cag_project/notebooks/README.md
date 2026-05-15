# Notebooks

Keep notebooks here for analysis, plots, tables, and paper figures.

Preferred pattern:

1. Load a config from `../configs/`.
2. Call reusable code from `../src/mail_cag/`.
3. Save figures or tables under a run/report folder.

Avoid putting the whole training loop directly inside a notebook again. That made
the earlier work hard to resume, compare, and trust.
