"""Unified command-line entrypoint for VLA-Precision."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from omegaconf import OmegaConf

from vla_precision.config import (
    ResolvedConfig,
    ResolvedStage1Config,
    load_config,
    load_stage1_config,
)
from vla_precision.logging import configure_logging, log_startup_config


def _add_config_arguments(parser: argparse.ArgumentParser, *, allow_deployment: bool = True) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Stage-specific task YAML (for Stage II: configs/stage2/tasks/<task>.yaml).",
    )
    if allow_deployment:
        parser.add_argument(
            "--deployment",
            type=Path,
            help="Shared Stage-II deployment YAML (normally configs/stage2/deployments/<name>.yaml).",
        )
    parser.add_argument(
        "--resolved-config-out",
        type=Path,
        help="Optional path for the fully resolved configuration.",
    )


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vla-precision")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="Inspect the unified configuration.")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    resolve = config_commands.add_parser("resolve")
    _add_config_arguments(resolve)
    resolve.add_argument("--stage", choices=("stage1", "stage2"), default="stage2")

    preprocess = commands.add_parser("preprocess", help="Precompute indexed demonstrations and context.")
    _add_config_arguments(preprocess)

    norm_stats = commands.add_parser("norm-stats", help="Compute OpenPI state/action normalization statistics.")
    _add_config_arguments(norm_stats, allow_deployment=False)
    norm_stats.add_argument("--max-frames", type=int)

    train = commands.add_parser("train", help="Run Stage I or Stage II training.")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    train_vla = train_commands.add_parser("vla", help="Stage I OpenPI full-parameter fine-tuning.")
    _add_config_arguments(train_vla, allow_deployment=False)
    train_acob = train_commands.add_parser("acob", help="Stage II ACoB online post-training.")
    _add_config_arguments(train_acob)
    train_acob.add_argument("--role", choices=("actor", "learner"), required=True)

    evaluate = commands.add_parser("evaluate", help="Evaluate a Stage-I or Stage-II checkpoint.")
    evaluate_commands = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_vla = evaluate_commands.add_parser("vla", help="Evaluate a Stage-I full VLA checkpoint.")
    _add_config_arguments(evaluate_vla)
    evaluate_acob = evaluate_commands.add_parser("acob", help="Evaluate a Stage-II ACoB actor checkpoint.")
    _add_config_arguments(evaluate_acob)

    openpi_inference = commands.add_parser(
        "openpi-inference",
        help="Run Stage-I inference without ACoB-Stream or Pyro.",
    )
    openpi_commands = openpi_inference.add_subparsers(
        dest="openpi_inference_command", required=True
    )
    openpi_run = openpi_commands.add_parser("run", help="Run on the robot-side machine.")
    openpi_run.add_argument("--config", type=Path, required=True)
    openpi_serve = openpi_commands.add_parser(
        "serve-policy", help="Serve the full OpenPI policy over WebSocket."
    )
    openpi_serve.add_argument("--config", type=Path, required=True)

    serve = commands.add_parser("serve", help="Run a service used by distributed training.")
    serve_commands = serve.add_subparsers(dest="serve_command", required=True)
    environment = serve_commands.add_parser(
        "robot-agent-bridge",
        help="Bridge the local robot system to a remote Actor or evaluator over Pyro.",
    )
    _add_config_arguments(environment)
    robot = serve_commands.add_parser("robot", help="Run the low-level UR RTDE/HTTP service.")
    _add_config_arguments(robot)

    return parser


def _is_stage1_command(args: argparse.Namespace) -> bool:
    return (
        args.command == "norm-stats"
        or (args.command == "train" and args.train_command == "vla")
        or (
            args.command == "config"
            and args.config_command == "resolve"
            and args.stage == "stage1"
        )
    )


def _resolve(
    args: argparse.Namespace,
    overrides: Sequence[str],
    *,
    role: str,
) -> ResolvedConfig | ResolvedStage1Config:
    if _is_stage1_command(args):
        if getattr(args, "deployment", None) is not None:
            raise ValueError("Stage I uses one self-contained config and does not accept --deployment")
        resolved = load_stage1_config(args.config, cli_args=overrides)
        os.environ["CUDA_VISIBLE_DEVICES"] = resolved.config.cuda_visible_devices
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
            resolved.config.xla_memory_fraction
        )
        configure_logging(
            "INFO",
            debug=resolved.config.debug,
            experiment=resolved.config.openpi.exp_name,
            role=role,
        )
        if resolved.config.debug:
            logging.getLogger(__name__).debug(
                "Stage-I config: name=%s exp_name=%s action_dim=%s action_horizon=%s config_sha256=%s",
                resolved.config.openpi.name,
                resolved.config.openpi.exp_name,
                resolved.config.openpi.model.action_dim,
                resolved.config.openpi.model.action_horizon,
                resolved.config_sha256,
            )
    else:
        resolved = load_config(
            args.config,
            deployment=getattr(args, "deployment", None),
            cli_args=overrides,
        )
        if role in ("actor", "vla", "acob"):
            os.environ["CUDA_VISIBLE_DEVICES"] = resolved.config.actor.cuda_visible_devices
        elif role == "learner":
            os.environ["CUDA_VISIBLE_DEVICES"] = resolved.config.learner.cuda_visible_devices
        elif role == "preprocess":
            os.environ["CUDA_VISIBLE_DEVICES"] = resolved.config.preprocess.cuda_visible_devices
        configure_logging(
            resolved.config.logging.level,
            debug=resolved.config.debug,
            experiment=resolved.config.experiment.name,
            role=role,
        )
        log_startup_config(resolved, role=role)
    if args.resolved_config_out:
        resolved.save(args.resolved_config_out)
    return resolved


def _dispatch(
    args: argparse.Namespace,
    resolved: ResolvedConfig | ResolvedStage1Config,
) -> None:
    runner: Callable[[ResolvedConfig], None]
    if args.command == "preprocess":
        from vla_precision.data.preprocess import run_preprocess

        runner = run_preprocess
    elif args.command == "norm-stats":
        from vla_precision.integrations.openpi.norm_stats import run_norm_stats

        runner = lambda config: run_norm_stats(config, max_frames=args.max_frames)
    elif args.command == "train" and args.train_command == "vla":
        from vla_precision.integrations.openpi.train import run_stage1_training

        runner = run_stage1_training
    elif args.command == "train" and args.train_command == "acob":
        if args.role == "actor":
            from vla_precision.acob_stream.actor import run_actor

            runner = run_actor
        else:
            from vla_precision.acob_stream.learner import run_learner

            runner = run_learner
    elif args.command == "evaluate":
        from vla_precision.evaluation import run_evaluation

        runner = lambda config: run_evaluation(config, mode=args.evaluate_command)
    elif args.command == "serve" and args.serve_command == "robot-agent-bridge":
        from vla_precision.robotics.environments.server import run_environment_server

        runner = run_environment_server
    elif args.command == "serve" and args.serve_command == "robot":
        from vla_precision.robotics.servers.ur import run_robot_server

        runner = run_robot_server
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    runner(resolved)


def _run_command(argv: Sequence[str] | None = None) -> None:
    """Run the normalized internal command selected by the public CLI."""
    parser = _command_parser()
    args, overrides = parser.parse_known_args(argv)
    if args.command == "openpi-inference":
        if overrides:
            parser.error("OpenPI native inference does not accept unknown command-line arguments")
        from vla_precision.integrations.openpi.inference import load_native_inference_config

        native = load_native_inference_config(args.config)
        configure_logging("INFO", debug=False)
        if args.openpi_inference_command == "run":
            from vla_precision.integrations.openpi.inference.runtime import run_native_inference

            run_native_inference(native)
        else:
            from vla_precision.integrations.openpi.inference.runtime import serve_native_policy

            serve_native_policy(native)
        return
    role = (
        getattr(args, "role", None)
        or getattr(args, "train_command", None)
        or getattr(args, "evaluate_command", None)
        or getattr(args, "serve_command", None)
        or args.command
    )
    resolved = _resolve(args, overrides, role=role)

    if args.command == "config":
        print(OmegaConf.to_yaml(OmegaConf.structured(resolved.config), resolve=True))
        if isinstance(resolved, ResolvedStage1Config):
            print(f"config_sha256: {resolved.config_sha256}")
        else:
            print(f"shared_config_sha256: {resolved.shared_sha256}")
            print(f"preprocess_config_sha256: {resolved.preprocess_sha256}")
        return

    experiment_name = (
        resolved.config.openpi.exp_name
        if isinstance(resolved, ResolvedStage1Config)
        else resolved.config.experiment.name
    )
    logging.getLogger(__name__).info("starting %s for %s", role, experiment_name)
    _dispatch(args, resolved)


def _parser() -> argparse.ArgumentParser:
    """Build the single user-facing ``--stage/--mode`` parser."""
    parser = argparse.ArgumentParser(description="Run VLA-Precision")
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "resolve-config",
            "norm-stats",
            "preprocess",
            "train",
            "evaluate",
            "openpi-inference",
            "serve-openpi-policy",
            "robot-agent-bridge",
            "serve-robot",
        ),
        required=True,
    )
    parser.add_argument("--role", choices=("actor", "learner"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--deployment")
    parser.add_argument("--resolved-config-out")
    parser.add_argument("--max-frames", type=int)
    return parser


def _append_common(target: list[str], args: argparse.Namespace) -> None:
    target.extend(("--config", args.config))
    if args.deployment is not None:
        target.extend(("--deployment", args.deployment))
    if args.resolved_config_out is not None:
        target.extend(("--resolved-config-out", args.resolved_config_out))


def _command_arguments(args: argparse.Namespace, overrides: Sequence[str]) -> list[str]:
    """Normalize the public stage/mode selection for the internal dispatcher."""
    if args.mode == "resolve-config":
        result = ["config", "resolve", "--stage", args.stage]
    elif args.mode == "norm-stats" and args.stage == "stage1":
        result = ["norm-stats"]
        if args.max_frames is not None:
            result.extend(("--max-frames", str(args.max_frames)))
    elif args.mode == "preprocess" and args.stage == "stage2":
        result = ["preprocess"]
    elif args.mode == "train" and args.stage == "stage1":
        result = ["train", "vla"]
    elif args.mode == "train" and args.stage == "stage2" and args.role is not None:
        result = ["train", "acob", "--role", args.role]
    elif args.mode == "evaluate":
        result = ["evaluate", "vla" if args.stage == "stage1" else "acob"]
    elif args.mode == "openpi-inference" and args.stage == "stage1":
        result = ["openpi-inference", "run"]
    elif args.mode == "serve-openpi-policy" and args.stage == "stage1":
        result = ["openpi-inference", "serve-policy"]
    elif args.mode == "robot-agent-bridge" and args.stage == "stage2":
        result = ["serve", "robot-agent-bridge"]
    elif args.mode == "serve-robot" and args.stage == "stage2":
        result = ["serve", "robot"]
    else:
        raise ValueError(f"Unsupported stage={args.stage!r}, mode={args.mode!r}, role={args.role!r} combination")
    _append_common(result, args)
    result.extend(overrides)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public VLA-Precision command-line interface."""
    parser = _parser()
    args, overrides = parser.parse_known_args(argv)
    try:
        arguments = _command_arguments(args, overrides)
    except ValueError as error:
        parser.error(str(error))
    _run_command(arguments)


if __name__ == "__main__":
    main()
