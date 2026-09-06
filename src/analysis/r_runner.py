"""R script runner using CENTAUR multilanguage support.

This module provides utilities for running R scripts through the
centaur-platform R engine, enabling seamless integration of R-based
statistical analysis with the Python-based CENTAUR infrastructure.

Usage:
    from src.analysis.r_runner import run_r_script, get_r_engine

    # Run a specific R script
    result = run_r_script(
        script_name="run_cluster_inference.R",
        data_path="data/processed/regression/regression_dataset.csv",
    )

    # Get the R engine directly
    engine = get_r_engine()
    is_valid, message = engine.validate_installation()
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from centaur_link import ensure_centaur_available, get_centaur_component, CENTAUR_PATH
from config.settings import get_settings


@dataclass
class RScriptResult:
    """Result from running an R script."""

    success: bool
    script_name: str
    output: str = ""
    error: str = ""
    result_data: Optional[dict] = None
    execution_time: float = 0.0


def get_r_engine():
    """Get the CENTAUR R engine.

    Returns:
        REngine instance from centaur-platform.

    Raises:
        CENTAURImportError: If R engine cannot be loaded.
    """
    ensure_centaur_available()
    REngine = get_centaur_component("src.analysis.engines.r_engine.REngine")
    return REngine()


def validate_r_installation() -> tuple[bool, str]:
    """Validate R installation for Gulf Hazard analysis.

    Returns:
        Tuple of (is_valid, message).
    """
    try:
        engine = get_r_engine()
        return engine.validate_installation()
    except Exception as e:
        return False, f"R engine not available: {e}"


def run_r_script(
    script_name: str,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    timeout: int = 3600,
) -> RScriptResult:
    """Run an R script from the r_analysis directory.

    Args:
        script_name: Name of the R script (e.g., "run_cluster_inference.R").
        data_path: Optional path to input data file.
        output_dir: Optional output directory for results.
        extra_args: Optional additional arguments to pass to script.
        timeout: Script timeout in seconds.

    Returns:
        RScriptResult with execution details.
    """
    import time

    settings = get_settings()

    # Locate script
    script_path = settings.project_root / "r_analysis" / script_name

    if not script_path.exists():
        return RScriptResult(
            success=False,
            script_name=script_name,
            error=f"Script not found: {script_path}",
        )

    # Build command
    cmd = ["Rscript", str(script_path)]

    if data_path:
        cmd.append(str(data_path))
    if output_dir:
        cmd.append(str(output_dir))
    if extra_args:
        cmd.extend(extra_args)

    # Run script
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(settings.project_root),
        )

        execution_time = time.time() - start_time

        if result.returncode != 0:
            return RScriptResult(
                success=False,
                script_name=script_name,
                output=result.stdout,
                error=result.stderr or f"Script exited with code {result.returncode}",
                execution_time=execution_time,
            )

        return RScriptResult(
            success=True,
            script_name=script_name,
            output=result.stdout,
            error=result.stderr,
            execution_time=execution_time,
        )

    except subprocess.TimeoutExpired:
        return RScriptResult(
            success=False,
            script_name=script_name,
            error=f"Script timed out after {timeout} seconds",
            execution_time=timeout,
        )
    except Exception as e:
        return RScriptResult(
            success=False,
            script_name=script_name,
            error=str(e),
            execution_time=time.time() - start_time,
        )


def run_cluster_inference(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> RScriptResult:
    """Run the cluster inference R script.

    This script performs event-clustered renter disadvantages analysis
    using CR2 odds ratios and wild-cluster bootstrap risk differences.

    Args:
        data_path: Path to regression dataset. Defaults to processed/regression/.
        output_dir: Output directory. Defaults to data/processed/results/cluster_inference/.

    Returns:
        RScriptResult with inference results.
    """
    settings = get_settings()

    if data_path is None:
        data_path = str(settings.processed_dir / "regression" / "regression_dataset_with_bert_themes.csv")

    if output_dir is None:
        output_dir = str(settings.processed_dir / "results" / "cluster_inference")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return run_r_script(
        script_name="run_cluster_inference.R",
        data_path=data_path,
        output_dir=output_dir,
    )


def run_missingness_analysis(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> RScriptResult:
    """Run the missingness and multiple imputation R script.

    This script performs missing-data rates by tenure analysis and
    multiple-imputation (20x MICE) displacement odds ratios.

    Args:
        data_path: Path to input dataset.
        output_dir: Output directory.

    Returns:
        RScriptResult with missingness analysis results.
    """
    settings = get_settings()

    if data_path is None:
        data_path = str(settings.processed_dir / "regression" / "regression_dataset_with_bert_themes.csv")

    if output_dir is None:
        output_dir = str(settings.processed_dir / "results" / "missingness")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return run_r_script(
        script_name="run_missingness_and_mi.R",
        data_path=data_path,
        output_dir=output_dir,
    )


def run_measurement_invariance(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> RScriptResult:
    """Run measurement invariance R script.

    Tests measurement invariance of factor structures across tenure groups.

    Args:
        data_path: Path to input dataset.
        output_dir: Output directory.

    Returns:
        RScriptResult with invariance test results.
    """
    settings = get_settings()

    if output_dir is None:
        output_dir = str(settings.processed_dir / "results" / "invariance")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return run_r_script(
        script_name="run_measurement_invariance.R",
        data_path=data_path,
        output_dir=output_dir,
    )


def run_hurricane_coverage_sensitivity(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> RScriptResult:
    """Run hurricane coverage sensitivity analysis R script.

    Analyzes coverage rates and odds ratios under hurricane-specific recodes.

    Args:
        data_path: Path to insurance dataset.
        output_dir: Output directory.

    Returns:
        RScriptResult with sensitivity analysis results.
    """
    settings = get_settings()

    if data_path is None:
        data_path = str(settings.processed_dir / "insurance" / "insurance_model_dataset.csv")

    if output_dir is None:
        output_dir = str(settings.processed_dir / "results" / "insurance")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return run_r_script(
        script_name="run_hurricane_coverage_sensitivity.R",
        data_path=data_path,
        output_dir=output_dir,
    )


# Available R analysis scripts
R_SCRIPTS = {
    "cluster_inference": {
        "script": "run_cluster_inference.R",
        "runner": run_cluster_inference,
        "description": "Event-clustered renter disadvantages (CR2 ORs, wild-cluster bootstrap)",
    },
    "missingness": {
        "script": "run_missingness_and_mi.R",
        "runner": run_missingness_analysis,
        "description": "Missing-data rates by tenure and multiple imputation analysis",
    },
    "measurement_invariance": {
        "script": "run_measurement_invariance.R",
        "runner": run_measurement_invariance,
        "description": "Measurement invariance tests across tenure groups",
    },
    "hurricane_sensitivity": {
        "script": "run_hurricane_coverage_sensitivity.R",
        "runner": run_hurricane_coverage_sensitivity,
        "description": "Hurricane coverage sensitivity analysis",
    },
    "insurance_models": {
        "script": "analyze_insurance_models.R",
        "runner": lambda: run_r_script("analyze_insurance_models.R"),
        "description": "Insurance model analysis",
    },
}


def list_r_analyses() -> list[dict]:
    """List available R analysis scripts.

    Returns:
        List of available R analysis configurations.
    """
    return [
        {"name": name, **config}
        for name, config in R_SCRIPTS.items()
    ]


def run_analysis(name: str, **kwargs) -> RScriptResult:
    """Run a named R analysis.

    Args:
        name: Analysis name from R_SCRIPTS.
        **kwargs: Arguments to pass to the runner.

    Returns:
        RScriptResult with analysis results.

    Raises:
        ValueError: If analysis name is not recognized.
    """
    if name not in R_SCRIPTS:
        available = ", ".join(R_SCRIPTS.keys())
        raise ValueError(f"Unknown analysis: {name}. Available: {available}")

    runner = R_SCRIPTS[name]["runner"]
    return runner(**kwargs)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core functions
    "get_r_engine",
    "validate_r_installation",
    "run_r_script",

    # Specific analyses
    "run_cluster_inference",
    "run_missingness_analysis",
    "run_measurement_invariance",
    "run_hurricane_coverage_sensitivity",

    # Registry
    "R_SCRIPTS",
    "list_r_analyses",
    "run_analysis",

    # Data classes
    "RScriptResult",
]
