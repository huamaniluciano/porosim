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

## Got a new module or some custom physics? We want it!

I designed POROSIM from the ground up to be super easy to expand. If you've been using it for your research and added something new, please consider sharing it with the community! You don't need to be a software engineer to contribute (I'm certainly not one! 😅).

### 1. Expanding the Extractor (Pillar 3)
Did you write a custom script to compute a new observable, extract a specific integral, or create a new kind of plot? You can drop your `.py` file right into `3_extractor/modulos/` (just make sure it follows the [Standard Module Contract](3_extractor/modulos/MODULE_CONTRACT.md)). 
The best part? The Extractor will automatically find it and plug it into both the console menu and the web GUI. You don't have to touch any GUI code at all!

### 2. Adding Custom Physics (Pillar 2)
If you simulated a system with parameters that aren't in our defaults—like a different type of electrolyte or salt—you can simply add the new JSON object into `2_solver/sales.json`. That's it! The solver will pick it up automatically without needing any Python code changes.

Don't be shy—if you built something cool, open a Pull Request. We'd love to see what you're doing with POROSIM!

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
