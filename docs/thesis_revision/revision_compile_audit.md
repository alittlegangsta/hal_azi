# Revision compile audit

## Environment

- Date: 2026-07-14
- Platform: Codex in WSL2, Windows TeX Live 2026
- TeX Live binary directory: `C:\texlive\2026\bin\windows`
- Source directory: `C:\Users\Administrator\Desktop\Hal\hal_azi_thesis\thesis_latex`
- Engine: XeLaTeX through latexmk 4.88 and BibTeX 0.99e

The TeX Live binary directory was not present in the inherited Windows `PATH`. It was added only to the individual `cmd.exe` process; no system environment setting was changed.

## Compile rounds

1. `latexmk -r latexmkrc main.tex` reached XeLaTeX but could not overwrite `main.pdf`, which was locked by another Windows process. The existing PDF was not closed, deleted, or overwritten.
2. The same source was compiled with the isolated job name `revision_validation`. BibTeX exposed an unescaped underscore in the verified issue string `5_Supplement`.
3. The issue string was normalized to `5 Supplement`. XeLaTeX, BibTeX, directory generation, references, and citations then converged. One 5.31824 pt table overfull remained.
4. The overview table spacing was reduced locally and the document was compiled again. latexmk reported all targets up to date.

Successful validation command:

```bat
set PATH=C:\texlive\2026\bin\windows;%PATH%
pushd C:\Users\Administrator\Desktop\Hal\hal_azi_thesis\thesis_latex
latexmk -r latexmkrc -jobname=revision_validation main.tex
```

## Final result

- Compile status: success
- Validation PDF: 55 pages, A4, PDF 1.7
- Undefined citations: 0
- Undefined references: 0
- Duplicate labels: 0 by source audit
- Missing figures: 0
- BibTeX fatal errors: 0
- Overfull boxes: 0
- Underfull boxes: 2 nonfatal warnings
- Other warning: the template does not define `CJKttdefault`; XeCJK ignored the unknown monospaced CJK family

The two underfull warnings occur in front matter and a narrow table cell. They do not indicate clipped or overlapping content. No class-file change was made to suppress the XeCJK warning.

## Cleanup

The validation PDF and all auxiliary build files are build artifacts. They are excluded from Git and are removed with latexmk cleanup after the audit data have been recorded. The pre-existing locked `main.pdf` is not staged.
