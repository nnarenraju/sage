# Contributing to Sage

Thank you for your interest in Sage.

## Asking questions

Open a [GitHub Discussion](https://github.com/nnarenraju/sage/discussions) for usage questions, ideas, or general feedback. Reserve the issue tracker for confirmed bugs and concrete feature requests.

## Reporting bugs

1. Search existing issues to avoid duplicates.
2. Open an issue with:
   - A minimal reproducible example.
   - Your Python, PyTorch, and CUDA versions (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`).
   - The full traceback.

## Submitting changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Add or update tests for any behaviour that changes.
3. Run the syntax and import checks locally:
   ```bash
   python -m py_compile $(find sage -name '*.py')
   python -c "import sage.dsp; print('OK')"
   pytest tests/ -v
   ```
4. Update `CHANGELOG.md` under `[Unreleased]` with a brief description.
5. Open a pull request against `main` with a clear description of the motivation and approach.

## Code style

- `black` (line length 88) for formatting.
- No comments that restate what the code does — only add a comment when the *why* is non-obvious.
- Keep the hot path in GPU-facing modules free of Python loops; use pre-computed integer tuples and registered buffers.

## Citation

If Sage contributes to published work, please cite the paper and the Zenodo release — details are in [`CITATION.cff`](CITATION.cff) and the repository README.
