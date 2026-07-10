# UESTC LaTeX Compile Audit

Generated: 2026-07-10. Scope: `thesis_latex/` first real compile audit.

## Compile Commands

Initial clean:

```bash
cmd.exe /c 'pushd \\wsl.localhost\Ubuntu\home\xiaoj\hal_azi\thesis_latex && C:\texlive\2026\bin\windows\latexmk.exe -C'
```

Final validation:

```bash
cmd.exe /c 'pushd \\wsl.localhost\Ubuntu\home\xiaoj\hal_azi\thesis_latex && C:\texlive\2026\bin\windows\latexmk.exe -C && C:\texlive\2026\bin\windows\latexmk.exe -r latexmkrc main.tex'
```

`latexmk -r latexmkrc main.tex` reads both `./latexmkrc` and the explicitly supplied `latexmkrc`, producing a harmless duplicate `run_makeglossaries` message. The compile still completes.

## Result

| item | status |
| --- | --- |
| PDF generated | yes |
| output file | `thesis_latex/main.pdf` |
| pages | 42 |
| final LaTeX exit code | 0 |
| committed PDF | no |
| committed aux/log/toc/out/fls/fdb artifacts | no |

Final log evidence:

```text
Output written on main.pdf (42 pages).
Latexmk: All targets (main.pdf) are up-to-date
```

## Fixes Applied

| file | fix | reason |
| --- | --- | --- |
| `.gitignore` | Added LaTeX build artifact ignore rules and `thesis_latex/main.pdf` | Prevent compile outputs from entering git while preserving `thesis_latex/pic/*.pdf` template resources. |
| `thesis_latex/chapters/chapter2_data_problem.tex` | Moved long `depth_heldout_split_confirmed` status string into a centered small `texttt` line | Removed an overfull paragraph from a long unbreakable status token. |
| `thesis_latex/chapters/chapter4_experiments.tex` | Narrowed EXP-006 table columns and reduced `tabcolsep` | Removed the large table overfull warning while keeping all metrics unchanged. |

No experiment metric values were changed.

## Warning Audit

| category | issue | status |
| --- | --- | --- |
| `must_fix_now` | Fatal `main.w18` / UNC cwd issue when directly invoking Windows TeX from WSL workdir | fixed by compiling through `cmd.exe pushd`; no template change required. |
| `must_fix_now` | Large EXP-006 table overfull | fixed. |
| `acceptable_for_draft` | Several underfull hbox warnings in dense metric/table text | accepted for current manuscript draft; can be polished during final layout. |
| `acceptable_for_draft` | `xeCJK` warning for undefined CJK monospaced family | accepted for current draft; main SimSun/SimHei/Times New Roman fonts load. |
| `requires_manual_content` | Cover metadata remains TODO | author, student number, advisor, school, major, dates, classification number, UDC must be filled manually. |
| `requires_real_bibliography` | Empty `thebibliography` environment | intentional to avoid fabricated references; replace with real BibTeX entries later. |
| `requires_real_figure` | All figures are placeholder boxes | replace with manually redrawn figures from `docs/thesis_plot_data/`. |

## Remaining Log Warnings

Final `main.log` warning summary:

```text
Package xeCJK Warning: Unknown CJK family `\CJKttdefault' is being ignored.
Underfull \hbox ... chapter3_method.tex lines 85--86
Underfull \hbox ... chapter4_experiments.tex line 133
Underfull \hbox ... chapter5_discussion.tex lines 47--48
LaTeX Warning: Empty `thebibliography' environment on input line 72.
```

No final undefined references were present after the second latexmk run.

## Hygiene

After final validation, compile artifacts were removed using `latexmk -C` before git staging. The following artifact classes are ignored:

- `*.aux`
- `*.log`
- `*.out`
- `*.toc`
- `*.lof`
- `*.lot`
- `*.fls`
- `*.fdb_latexmk`
- `*.synctex.gz`
- `*.bbl`
- `*.blg`
- `thesis_latex/main.pdf`

The template resource PDFs under `thesis_latex/pic/` are intentionally retained and are not compile outputs.
