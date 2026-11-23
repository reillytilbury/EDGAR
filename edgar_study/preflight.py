"""A few requests that show whether a model is usable before a run: availability, price, text, image."""

import asyncio
import io
import json
import time

import httpx

from edgar_study.go_client import BASE_URL, USER_AGENT, GoClient, Ledger, load_key


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
    key = load_key()
    listing = httpx.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {key}",
                                                       "User-Agent": USER_AGENT}, timeout=30).json()
    models = {entry["id"]: entry for entry in listing["data"]}
    run_dir = args.results / "preflight" / time.strftime("%Y%m%d-%H%M%S")
    report = {"models": {}}
    for name in filter(None, (args.model, args.image_model)):
        entry = models.get(name)
        if entry is None:
            raise SystemExit(f"{name} is not in the OpenCode Go model list.")
        report["models"][name] = {"price_per_mtok": entry["cost"]["text"],
                                  "input_modalities": entry["modalities"]["input"]}

    client = GoClient(run_dir, Ledger(args.results / "spend.jsonl"), session_id="edgar-preflight",
                      allow_paid=args.allow_paid, max_tokens=args.max_tokens,
                      reasoning_effort=args.reasoning_effort, temperature_scale=args.temperature_scale)

    async def checks():
        try:
            text = await client.chat(args.model, "Reply with exactly: ok", temperature=1.0)
            vision_model = args.image_model or args.model
            image = await client.chat(vision_model, "In one sentence: which curve has the sharper peak, "
                                      "blue or red?", img_bytes=_diagnostic_png())
            return text, image
        finally:
            await client.close()

    text, image = asyncio.run(checks())
    report.update(text_reply=text, image_reply=image, spend_total_usd=client.ledger.total)
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if text is None or image is None:
        raise SystemExit(f"Preflight failed; see {run_dir / 'api_calls.jsonl'}.")
