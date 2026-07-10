# Compile Instructions

The ThesisUESTC template is XeLaTeX-based.

Recommended local environment:

- TeX Live with XeLaTeX
- latexmk
- Chinese fonts required by the template, or the font package recommended by ThesisUESTC

Compile from the thesis directory:

```bash
cd thesis_latex
latexmk main.tex
```

Fallback command:

```bash
cd thesis_latex
xelatex main.tex
```

If bibliography entries are later added and `\thesisbibliography{refs/references}` is enabled, compile with XeLaTeX/BibTeX/XeLaTeX/XeLaTeX or use `latexmk`.

Current draft intentionally keeps `refs/references.bib` as TODO comments only and uses an empty manual bibliography environment to avoid fabricated citations.

Do not commit compile artifacts such as `.aux`, `.log`, `.pdf`, `.out`, `.toc`, `.fls`, `.fdb_latexmk`, or `.synctex.gz`.

## Local Check In This Environment

Checked on 2026-07-10:

- `latexmk main.tex`: not available, `/bin/bash: latexmk: command not found`
- `xelatex main.tex`: not available, `/bin/bash: xelatex: command not found`

The manuscript files are generated, but PDF compilation still needs a local TeX Live or equivalent XeLaTeX environment.
