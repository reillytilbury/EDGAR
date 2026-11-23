"""Run the unmodified upstream EDGAR search with OpenCode Go models in place of Gemini.

Upstream files stay byte-identical (checked against the pinned commit). Only module
attributes are replaced at runtime:
  * utils.call_llm_async -> GoClient.chat (plus the optional image-reader bridge)
  * prompt builders are called with llm_type='g', i.e. the Gemini prompts the paper used
  * hypothesis_engine.genai.Client -> stub, so no Google key is needed
  * hypothesis_engine.translate_to_jax / generate_new_* -> wrappers that tag each request
    with its role and record NumPy -> JAX translations for later rescoring
"""

import asyncio
import functools
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from edgar_study.go_client import GoClient, Ledger, role

IMAGE_READER_PROMPT = (
    "You are helping an AI scientist who cannot see images. The image shows tuning-curve fits "
    "for several neurons: scatter points are raw responses, the blue curve is the binned data "
    "mean, and the coloured curves are candidate models (red is the best model, green the "
    "second best). For each subplot, describe concisely where each model curve deviates from "
    "the binned mean: peak height, peak sharpness or cusp shape, width, asymmetry, baseline, "
    "second peak position and amplitude. Finish with the systematic weaknesses shared across subplots."
)


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def install(hypothesis_engine, utils, client, image_model=None, translations_path=None):
    """Swap upstream's LLM layer for `client`. Returns nothing; patches module attributes."""

    async def call_llm_async(prompt_text, client=None, model_name=None, temperature=1.0,
                             thinking_budget=1, img_bytes=None):
        if prompt_text is None:
            return None
        if img_bytes is not None and image_model and image_model != model_name.partition("#")[0]:
            token = role.set("image_reader")
            try:
                description = await _client.chat(image_model, IMAGE_READER_PROMPT, img_bytes=img_bytes)
            finally:
                role.reset(token)
            if description is None:
                return None
            prompt_text += ("\n\n[The image is not attached. A separate vision model described it as follows.]\n"
                            + description)
            img_bytes = None
        return await _client.chat(model_name, prompt_text, temperature=temperature, img_bytes=img_bytes)

    _client = client
    utils.call_llm_async = call_llm_async

    for name in ("create_program_prompt", "create_parameter_estimator_prompt", "create_linked_prompt"):
        original = getattr(utils, name)
        setattr(utils, name, functools.partial(_force_gemini_prompt, original))

    hypothesis_engine.genai.Client = lambda *args, **kwargs: None

    def tag(original, label):
        @functools.wraps(original)
        async def wrapper(*args, **kwargs):
            role.set(label)  # each gathered coroutine runs in its own task context
            return await original(*args, **kwargs)
        return wrapper

    original_translate = hypothesis_engine.translate_to_jax

    async def translate_to_jax(code_string, *args, **kwargs):
        role.set("translation")
        jax_code, func = await original_translate(code_string, *args, **kwargs)
        if translations_path and code_string is not None:
            with open(translations_path, "a") as stream:
                stream.write(json.dumps({"numpy_sha256": sha256(code_string), "numpy_code": code_string,
                                         "jax_code": jax_code, "compiled": func is not None}) + "\n")
        return jax_code, func

    hypothesis_engine.translate_to_jax = translate_to_jax
    hypothesis_engine.generate_new_neuron_model = tag(hypothesis_engine.generate_new_neuron_model, "proposal")
    hypothesis_engine.generate_new_parameter_estimator = tag(
        hypothesis_engine.generate_new_parameter_estimator, "estimator")


def _force_gemini_prompt(original, *args, **kwargs):
    kwargs["llm_type"] = "g"
    return original(*args, **kwargs)


def run(args, import_upstream):
    """Run hypothesis_engine.main with the configured models inside a fresh run directory."""
    repo = Path(__file__).resolve().parents[1]
    args.dataset = args.dataset.resolve()  # the search runs from inside the run directory
    upstream = import_upstream(args.upstream.resolve())
    hypothesis_engine, utils = upstream["hypothesis_engine"], upstream["utils"]

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = (args.results / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True)
    ledger = Ledger(args.results.resolve() / "spend.jsonl")
    client = GoClient(run_dir, ledger, session_id=f"edgar-{run_id}", allow_paid=args.allow_paid,
                      max_concurrency=args.concurrency, max_tokens=args.max_tokens,
                      reasoning_effort=args.reasoning_effort, temperature_scale=args.temperature_scale)
    install(hypothesis_engine, utils, client, image_model=args.image_model,
            translations_path=run_dir / "translations.jsonl")

    manifest = {
        "run_id": run_id, "repo_sha": git_sha(repo), "upstream_commit": upstream["commit"],
        "dataset": str(args.dataset), "iterations": args.iterations, "time_limit_min": args.time_limit,
        "models": {"translator": args.model, "proposer": args.model, "estimator": args.model,
                   "large_proposer": large_model(args), "image_reader": args.image_model},
        "use_image_feedback": not args.no_images, "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort, "temperature_scale": args.temperature_scale,
        "concurrency": args.concurrency,
        "spend_before_usd": ledger.total, "started": time.time(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run directory: {run_dir}")

    status = "completed"
    previous = Path.cwd()
    os.chdir(run_dir)  # upstream writes program_databases/ under the working directory
    try:
        asyncio.run(_main(hypothesis_engine, client, args))
    except Exception as error:  # record the failure, then re-raise after the manifest is updated
        status = f"failed: {type(error).__name__}: {error}"
        raise
    finally:
        os.chdir(previous)
        manifest.update(status=status, finished=time.time(), spend_after_usd=ledger.total)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


def large_model(args):
    """Upstream's every-third-generation thinking tier: the same model with more reasoning by default."""
    return args.large_model or f"{args.model}#{args.large_reasoning_effort}"


async def _main(hypothesis_engine, client, args):
    try:
        await hypothesis_engine.main(
            n_iterations=args.iterations, time_limit=args.time_limit,
            use_image_feedback=not args.no_images, use_large_every=3, param_penalty_weight=0.01,
            tiny_lm_name=args.model, little_lm_name=args.model,
            large_lm_name=large_model(args),
            data_path=str(args.dataset),
        )
    finally:
        await client.close()
