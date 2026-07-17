# Contributing to POROSIM

Thanks for your interest in POROSIM — an open-source research code for
axisymmetric Poisson–Nernst–Planck (PNP) simulation of ionic transport in
nanopores. Bug reports, questions, and code contributions are all welcome.

> **Heads-up:** POROSIM is developed by a single maintainer alongside research
> work, so replies may take some time. Thanks for your patience.

## Reporting a bug

Please open a
**[GitHub Issue](https://github.com/huamaniluciano/porosim/issues)**. A good
report lets the problem be reproduced quickly — please include:

- **What happened vs. what you expected.**
- **Which pillar** is involved: mesher (`1_mesher`), solver (`2_solver`), or
  extractor (`3_extractor`).
- **How to reproduce it**: the exact command, and the input files you used
  (the mesher's `geom.json`, the solver's `params.json`, or the mesh / solution
  folder). Small inputs are ideal.
- **The full error message / traceback** (copy-paste the text, not a screenshot).
- **Your environment**: operating system and how you installed the dependencies
  (e.g. `conda env create -f environment.yml`). The output of `conda list`
  helps.

## Asking for help

If it is not a bug but a usage question ("how do I…?"), you have two options:

- Open a **GitHub Issue** — it is public, so the answer helps the next person
  with the same question.
- Or email the maintainer directly: **huamani.luciano@quimica.unlp.edu.ar**.

## Contributing code

1. **Fork** the repository and create a branch for your change.
2. Make your change, following the style of the surrounding code. (POROSIM mixes
   English documentation and comments with Spanish identifiers — please keep
   that convention.)
3. **Run the test suite** and make sure it stays green:
   ```bash
   python -m pytest
   ```
   The same suite runs automatically in continuous integration on every pull
   request, so a PR is ready to review once CI is green.
4. Open a **Pull Request** describing what you changed and why.

By contributing, you agree that your contributions are licensed under the
project's **[MIT License](LICENSE)**.

## Code of Conduct

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful in all interactions.

## Contact

Maintainer and project lead: **Angel Luciano Huamani** —
huamani.luciano@quimica.unlp.edu.ar
