# Final Compile Audit

## Compile environment

- Repository: `/mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis`
- Thesis directory: `C:\Users\Administrator\Desktop\Hal\hal_azi_thesis\thesis_latex`
- Build command: `latexmk -r latexmkrc main.tex`
- TeX distribution: Windows TeX Live 2026
- Audit date: 2026-07-15

The build was executed through `cmd.exe` from WSL so that the previously verified Windows TeX Live toolchain and the repository-local `latexmkrc` were used. No class or bibliography-style file was modified by this task.

## Build rounds

1. The first final build succeeded and exposed one `3.31607 pt` overfull box in the English abstract.
2. The English abstract sentence was shortened without changing its scientific meaning. The next build reduced the overfull count to zero.
3. BibTeX surname particles and the `blockCV` title capitalization were protected, followed by a final successful rebuild.

## Final result

| Check | Result |
|---|---:|
| Build status | success |
| PDF pages | 57 |
| Paper size | A4 |
| Undefined citations | 0 |
| Undefined references | 0 |
| Duplicate labels | 0 |
| Missing figures | 0 |
| BibTeX warnings | 0 |
| Overfull boxes | 0 |
| Underfull boxes | 2 |
| Font warnings | 1 |

The two underfull boxes occur in a cover-information area with administrative placeholders and in a data-description table. They do not hide or overlap content. The remaining font warning is the template-level xeCJK warning `Unknown CJK family \CJKttdefault`; no change was made to `thesis-uestc.cls` to suppress it.

## Content checks on the generated PDF

- Chinese and English abstracts were generated and include the confirmed Halliburton project source.
- The table of contents, six chapters, references, appendix and acknowledgements were generated.
- The reference list contains 48 entries, matching the 48 distinct citation keys used by the thesis source.
- The PDF text preserves the conclusion boundary that the method cannot recover absolute azimuth and has not established multi-well generalization.
- No Grad-CAM figure of unverified provenance appears in the main text.

## Artifact handling

The generated `main.pdf` was used only for this audit. In accordance with the task constraints, `latexmk -C` was run after verification, and the PDF and compilation cache are not included in Git.
