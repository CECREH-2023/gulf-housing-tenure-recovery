# Software requirements

Use a separate Python environment and install the profile shown in [REPRODUCING.md](REPRODUCING.md). That profile names the packages used by the documented statistical entry point. It is not a pinned environment for every acquisition, mapping, or sensitivity module. The full [pyproject.toml](../pyproject.toml) includes retrieval and model-provider tooling beyond the two statistical entry points.

The commands and statistical checks actually executed are recorded in [VALIDATION.json](../VALIDATION.json). Installing a dependency specification alone does not demonstrate end-to-end reproduction.
