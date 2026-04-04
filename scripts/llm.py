#!/usr/bin/env python3
"""Interactive chat script for LLM models.

Usage
-----
# Chat with a model by name/path
python scripts/llm.py chat --model allenai/Olmo-3-1025-7B

# Chat with a model from a training run
python scripts/llm.py chat --run-name sft-olmo-3-1025-7b-nemotron_pt_v2-20260218-172238

# Interactive mode - choose from existing training runs (default)
python scripts/llm.py chat
python scripts/llm.py chat --interactive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the project root is on ``sys.path`` so that ``post_training`` is
# importable when running directly (``python scripts/chat.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from post_training.chat_templates.registry import get_chat_template
from post_training.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def find_available_runs(outputs_dir: Path) -> list[dict[str, str]]:
    """Find all available training runs in the outputs directory.

    Returns a list of dictionaries with keys:
    - run_name: name of the training run
    - run_dir: path to the run directory
    - has_inference_checkpoint: whether inference checkpoints exist
    - has_training_checkpoint: whether training checkpoints exist
    """
    runs = []
    if not outputs_dir.exists():
        return runs

    for item in sorted(outputs_dir.iterdir()):
        if not item.is_dir():
            continue

        run_name = item.name
        run_dir = item

        # Check for inference checkpoints
        inference_dir = run_dir / "inference_checkpoints"
        inference_checkpoints = []
        if inference_dir.exists():
            inference_checkpoints = sorted(
                inference_dir.glob("step-*"), key=lambda x: int(x.name.split("-")[1])
            )

        # Check for training checkpoints
        checkpoints_dir = run_dir / "checkpoints"
        training_checkpoints = []
        if checkpoints_dir.exists():
            training_checkpoints = sorted(
                checkpoints_dir.glob("checkpoint-*"),
                key=lambda x: int(x.name.split("-")[1]),
            )

        if inference_checkpoints or training_checkpoints:
            runs.append(
                {
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "has_inference_checkpoint": len(inference_checkpoints) > 0,
                    "has_training_checkpoint": len(training_checkpoints) > 0,
                    "latest_inference": str(inference_checkpoints[-1])
                    if inference_checkpoints
                    else None,
                    "latest_training": str(training_checkpoints[-1])
                    if training_checkpoints
                    else None,
                }
            )

    return runs


def load_model_from_run(run_name: str, outputs_dir: Path) -> tuple[str, str]:
    """Load model path and config from a training run.

    Returns:
        Tuple of (model_path, chat_template_name)
    """
    run_dir = outputs_dir / run_name
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")

    # Load config to get chat template
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    chat_template = config_dict.get("data", {}).get("chat_template", "default")

    # Prefer inference checkpoint, fall back to training checkpoint
    inference_dir = run_dir / "inference_checkpoints"
    if inference_dir.exists():
        inference_checkpoints = sorted(
            inference_dir.glob("step-*"), key=lambda x: int(x.name.split("-")[1])
        )
        if inference_checkpoints:
            model_path = str(inference_checkpoints[-1])
            logger.info("Using inference checkpoint: %s", model_path)
            return model_path, chat_template

    # Fall back to training checkpoint
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        training_checkpoints = sorted(
            checkpoints_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1]),
        )
        if training_checkpoints:
            model_path = str(training_checkpoints[-1])
            logger.info("Using training checkpoint: %s", model_path)
            return model_path, chat_template

    raise ValueError(f"No checkpoints found in run: {run_name}")


def load_model_and_tokenizer(
    model_path: str, chat_template: Optional[str] = None, device: str = "cuda"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer for inference."""
    logger.info("Loading model from: %s", model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply chat template if provided
    if chat_template:
        try:
            template_str = get_chat_template(chat_template)
            tokenizer.chat_template = template_str
            logger.info("Applied chat template: %s", chat_template)
        except Exception as e:
            logger.warning("Failed to load chat template '%s': %s", chat_template, e)
            logger.info("Using tokenizer's default chat template")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info("Model loaded successfully")
    return model, tokenizer


def chat_loop(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    """Run an interactive chat loop."""
    messages = []

    print("\n" + "=" * 80)
    print("Chat started. Type your message and press Enter.")
    print("Type 'quit', 'exit', or press Ctrl+C to end the conversation.")
    print("=" * 80 + "\n")

    try:
        while True:
            # Get user input
            try:
                user_input = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting chat...")
                break

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nExiting chat...")
                break

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Format conversation
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.error("Error applying chat template: %s", e)
                print(f"Error: Failed to format conversation: {e}")
                messages.pop()  # Remove the failed message
                continue

            # Tokenize
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            # Generate
            print("Assistant: ", end="", flush=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            print(response)

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response})
            print()

    except KeyboardInterrupt:
        print("\n\nExiting chat...")
    except Exception as e:
        logger.error("Error during chat: %s", e, exc_info=True)
        print(f"\nError: {e}")


def chat_command(args: argparse.Namespace) -> None:
    """Handle the chat command."""
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"

    # Determine model path and chat template
    model_path = None
    chat_template = None

    # Default to interactive mode if no option is provided
    if args.interactive or (not args.model and not args.run_name):
        # Interactive mode: show all runs and let user choose
        runs = find_available_runs(outputs_dir)

        if not runs:
            logger.error("No training runs found in %s", outputs_dir)
            sys.exit(1)

        print("\n" + "=" * 80)
        print("Available Training Runs")
        print("=" * 80)
        print()

        for i, run in enumerate(runs, 1):
            status = "✓" if run["has_inference_checkpoint"] else "⚠"
            print(f"{i:2d}. {status} {run['run_name']}")
            if run["latest_inference"]:
                print(f"    → inference: {Path(run['latest_inference']).name}")
            if run["latest_training"]:
                print(f"    → training: {Path(run['latest_training']).name}")
            print()

        print("=" * 80)
        print("Enter run number to chat with (or 'q' to quit):")
        selection = input("> ").strip()

        if selection.lower() in ("q", "quit", "exit"):
            print("Exiting...")
            return

        try:
            idx = int(selection) - 1
            if not (0 <= idx < len(runs)):
                logger.error("Invalid selection: %d", idx + 1)
                sys.exit(1)
            selected_run = runs[idx]
            model_path, chat_template = load_model_from_run(
                selected_run["run_name"], outputs_dir
            )
        except ValueError:
            logger.error("Invalid selection. Please enter a number.")
            sys.exit(1)

    elif args.run_name:
        # Non-interactive mode: use specified run name
        model_path, chat_template = load_model_from_run(args.run_name, outputs_dir)

    elif args.model:
        # Non-interactive mode: use specified model path
        model_path = args.model
        # Try to infer chat template from model name (optional)
        chat_template = args.chat_template

    if not model_path:
        logger.error("No model path specified")
        sys.exit(1)

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available, using CPU (may be slow)")

    model, tokenizer = load_model_and_tokenizer(model_path, chat_template, device)

    # Start chat loop
    chat_loop(model, tokenizer)


def main() -> None:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Interactive chat script for LLM models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Path to the outputs directory (default: outputs)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session",
        description="Start an interactive chat session. Interactive mode is the default. "
        "Use --model or --run-name for non-interactive mode.",
    )

    chat_group = chat_parser.add_mutually_exclusive_group(required=False)
    chat_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: show all runs and let user choose (default)",
    )
    chat_group.add_argument(
        "--model",
        type=str,
        help="Model name or path (e.g., 'allenai/Olmo-3-1025-7B' or '/path/to/model')",
    )
    chat_group.add_argument(
        "--run-name",
        type=str,
        help="Training run name (from outputs folder) to chat with",
    )

    chat_parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Chat template name (only used with --model, defaults to tokenizer's template)",
    )

    args = parser.parse_args()

    if args.command == "chat":
        chat_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
