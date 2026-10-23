#!/bin/bash
# Run the test suite from the repository root; extra arguments go to pytest.
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)" || exit 1
python -m pytest tests "$@"
