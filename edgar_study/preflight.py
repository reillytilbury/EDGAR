"""A few requests that show whether a model is usable before a run: availability, price, text, image."""

import asyncio
import io
import json
import time

from edgar_study.go_client import GoClient, Ledger, fetch_models, is_free, load_key


def _diagnostic_png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    theta = np.linspace(0, 2 * np.pi, 200)
    figure, axis = plt.subplots(figsize=(4, 3), dpi=80)
    axis.plot(theta, np.exp(-np.abs(theta - np.pi)), color="tab:blue", label="data")
    axis.plot(theta, np.exp(-0.5 * ((theta - np.pi) / 0.8) ** 2), color="tab:red", label="model")
    axis.legend()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return buffer.getvalue()


def preflight(args):
    models = fetch_models(load_key())
    report = {"models": {}}
    for name in filter(None, (args.model, args.image_model)):
        if name not in models:
            raise SystemExit(f"{name} is not in the OpenCode Go model list.")
        report["models"][name] = {"price_per_mtok": models[name]["cost"]["text"],
                                  "input_modalities": models[name]["modalities"]["input"]}

    run_dir = args.results / "preflight" / time.strftime("%Y%m%d-%H%M%S")
    go = GoClient(run_dir, Ledger(args.results / "spend.jsonl"), session_id="edgar-preflight",
                  free_models=[name for name, entry in models.items() if is_free(entry)],
                  allow_paid=args.allow_paid, max_tokens=args.max_tokens)

    async def checks():
        try:
            return await asyncio.gather(
                go.chat(args.model, "Reply with exactly: ok", temperature=0.5, effort=args.reasoning_effort),
                go.chat(args.image_model or args.model, "In one sentence: which curve has the sharper "
                        "peak, blue or red?", effort=args.reasoning_effort, img_bytes=_diagnostic_png()))
        finally:
            await go.close()

    text, image = asyncio.run(checks())
    report.update(text_reply=text, image_reply=image, expected_image_reply="blue",
                  spend_total_usd=go.ledger.total)
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if text is None or image is None:
        raise SystemExit(f"Preflight failed; see {run_dir / 'api_calls.jsonl'}.")
