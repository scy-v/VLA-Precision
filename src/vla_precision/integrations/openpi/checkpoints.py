"""Resolve OpenPI checkpoint locations at the extension boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenPICheckpoint:
    """One OpenPI checkpoint in both policy- and weight-loader form."""

    directory: str
    params_path: str
    step: int | None


@dataclass(frozen=True)
class OpenPIAssets:
    """Normalization assets associated with one OpenPI checkpoint."""

    directory: str
    asset_id: str
    norm_stats_path: str | None


def resolve_openpi_checkpoint(
    source: str | Path,
    *,
    requested_step: int | None = None,
) -> OpenPICheckpoint:
    """Resolve a manager root, numeric step, or ``params`` child.

    OpenPI policy construction consumes the numeric checkpoint directory,
    whereas ``CheckpointWeightLoader`` consumes its ``params`` child. Remote
    URIs cannot be enumerated, so they are retained as given and only the two
    corresponding views are derived.
    """
    requested_step = None if requested_step is None else int(requested_step)
    if requested_step is not None and requested_step < 0:
        raise ValueError(f"requested_step must be non-negative, got {requested_step}")

    source_text = str(source)
    if "://" in source_text:
        normalized = source_text.rstrip("/")
        if normalized.endswith("/params"):
            directory = normalized.removesuffix("/params")
            params_path = normalized
        else:
            directory = normalized
            params_path = f"{normalized}/params"
        return OpenPICheckpoint(
            directory=directory,
            params_path=params_path,
            step=requested_step,
        )

    path = Path(source).expanduser().resolve()
    if path.name == "params":
        path = path.parent

    if (path / "params").exists():
        step = int(path.name) if path.name.isdigit() else None
        if requested_step is not None and step is not None and requested_step != step:
            raise FileNotFoundError(
                f"checkpoint path selects step {step}, but requested_step={requested_step}"
            )
        return OpenPICheckpoint(
            directory=str(path),
            params_path=str(path / "params"),
            step=step,
        )

    available = tuple(
        sorted(
            int(candidate.name)
            for candidate in path.iterdir()
            if candidate.is_dir()
            and candidate.name.isdigit()
            and (candidate / "params").exists()
        )
    ) if path.is_dir() else ()
    if not available:
        raise FileNotFoundError(f"no OpenPI checkpoint containing params was found at {path}")

    step = available[-1] if requested_step is None else requested_step
    if step not in available:
        raise FileNotFoundError(
            f"checkpoint step {step} is unavailable at {path}; found {list(available)}"
        )
    directory = path / str(step)
    return OpenPICheckpoint(
        directory=str(directory),
        params_path=str(directory / "params"),
        step=step,
    )


def resolve_openpi_assets(
    source: str | Path,
    *,
    default_asset_id: str,
    requested_step: int | None = None,
) -> OpenPIAssets | None:
    """Locate the norm stats saved with a Stage-I OpenPI checkpoint.

    Local checkpoints retain the paper implementation's behavior: use the
    first sorted ``norm_stats.json`` below the checkpoint assets, and keep the
    configured assets path when none exists. Remote URIs cannot be enumerated,
    so OpenPI receives the conventional ``assets/<repo-id>`` location.
    """
    checkpoint = resolve_openpi_checkpoint(source, requested_step=requested_step)
    if "://" in checkpoint.directory:
        assets_directory = f"{checkpoint.directory.rstrip('/')}/assets"
        asset_id = str(default_asset_id)
        return OpenPIAssets(
            directory=assets_directory,
            asset_id=asset_id,
            norm_stats_path=f"{assets_directory}/{asset_id}/norm_stats.json",
        )

    resolved_directory = Path(checkpoint.directory)
    source_path = Path(source).expanduser().resolve()
    if source_path.name == "params":
        source_path = source_path.parent
    candidates = [resolved_directory / "assets"]
    if source_path != resolved_directory:
        candidates.append(source_path / "assets")

    for assets_directory in candidates:
        norm_stats_files = sorted(assets_directory.rglob("norm_stats.json")) if assets_directory.exists() else []
        if not norm_stats_files:
            continue
        norm_stats_path = norm_stats_files[0]
        return OpenPIAssets(
            directory=str(assets_directory),
            asset_id=norm_stats_path.parent.relative_to(assets_directory).as_posix(),
            norm_stats_path=str(norm_stats_path),
        )
    return None
