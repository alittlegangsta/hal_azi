# UESTC LaTeX Environment Audit

Generated: 2026-07-10. Scope: `thesis_latex/` compile environment only. No model training, remote job, raw data, processed data, Windows results, or experiment metrics were modified.

## Tool Discovery

The Linux/WSL `PATH` did not expose TeX tools directly:

| command | WSL PATH result |
| --- | --- |
| `xelatex --version` | `command not found` |
| `latexmk --version` | `command not found` |
| `bibtex --version` | `command not found` |
| `kpsewhich thesis-uestc.cls` | `command not found` |

TeX Live was found on the Windows side:

```text
/mnt/c/texlive/2026/bin/windows/xelatex.exe
/mnt/c/texlive/2026/bin/windows/latexmk.exe
/mnt/c/texlive/2026/bin/windows/bibtex.exe
/mnt/c/texlive/2026/bin/windows/kpsewhich.exe
```

## Version Checks

| tool | command used | result |
| --- | --- | --- |
| XeLaTeX | `/mnt/c/texlive/2026/bin/windows/xelatex.exe --version` | XeTeX `3.141592653-2.6-0.999998`, TeX Live 2026, kpathsea `6.4.2` |
| latexmk | `/mnt/c/texlive/2026/bin/windows/latexmk.exe --version` | Latexmk version `4.88`, 2026-03-09 |
| BibTeX | `/mnt/c/texlive/2026/bin/windows/bibtex.exe --version` | BibTeX `0.99e`, TeX Live 2026, kpathsea `6.4.2` |
| template lookup | `/mnt/c/texlive/2026/bin/windows/kpsewhich.exe thesis-uestc.cls` from `thesis_latex/` | `./thesis-uestc.cls` |

## Path Handling

Directly invoking Windows TeX executables from the WSL repo directory failed because Windows `cmd.exe` does not accept a UNC path as the current directory. The initial failure produced:

```text
Package catchfile Error: File `main.w18' not found.
```

Root cause: `ifplatform` shell-escape detection ran from a WSL UNC current directory. The successful compile used Windows `cmd.exe pushd` to map the WSL UNC directory to a temporary Windows drive:

```bash
cmd.exe /c 'pushd \\wsl.localhost\Ubuntu\home\xiaoj\hal_azi\thesis_latex && C:\texlive\2026\bin\windows\latexmk.exe -r latexmkrc main.tex'
```

## Font Availability From Compile Log

The final successful log shows that the template found the expected Windows fonts:

| font | log evidence | status |
| --- | --- | --- |
| SimSun | `Font family 'SimSun(0)' created for font 'SimSun'` | available |
| SimHei | `Font family 'SimHei(0)' created for font 'SimHei'`; later `SimHei(1)` for CJK script | available |
| Times New Roman | `Font family 'TimesNewRoman(0)' created for font 'Times New Roman'` | available |

Residual font warning:

```text
Package xeCJK Warning: Unknown CJK family `\CJKttdefault' is being ignored.
```

This is classified as acceptable for draft because PDF generation succeeds and the main Chinese/English fonts are available.
