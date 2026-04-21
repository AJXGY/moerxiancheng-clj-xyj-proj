#!/usr/bin/env python3
import json
import os
from PIL import Image, ImageDraw, ImageFont


ROOT = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(ROOT, "charts")
W, H = 2200, 1400
DPI = (600, 600)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_latest_artifact():
    path = os.path.join(ROOT, "latest_artifact.txt")
    if not os.path.exists(path):
        raise FileNotFoundError("latest_artifact.txt not found, run benchmark_parallel_train_time.py first")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def canvas(title):
    img = Image.new("RGB", (W, H), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.text((80, 60), title, fill="#0f172a", font=font(54, True))
    return img, draw


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, dpi=DPI)


def draw_status(measurement_type):
    img, draw = canvas("5.2.14 Task Status")
    y = 190
    c_text = (
        "Completed real multi-config training iteration measurements"
        if measurement_type == "real_training_probe"
        else "Completed model-side sampling only (not real hardware runs)"
    )
    labels = {
        "A": "Environment and training script setup completed",
        "B": "PP=1/2 and MB=1/2/4 configs prepared",
        "C": c_text,
        "D": "Training time prediction completed",
        "E": "Error calculation and logging completed",
        "F": "Threshold check completed (<=20%)",
    }
    for label in "ABCDEF":
        fill = "#16a34a" if not (label == "C" and measurement_type != "real_training_probe") else "#d97706"
        draw.rounded_rectangle((90, y, 2060, y + 95), radius=24, fill=fill)
        draw.text((130, y + 24), label, fill="white", font=font(34, True))
        draw.text((230, y + 24), labels[label], fill="white", font=font(30))
        y += 120
    save(img, os.path.join(CHART_DIR, "status_af.png"))


def draw_error(model):
    img, draw = canvas("Training-Time Error by Parallel Config")
    configs = model["configs"]
    left, bottom, width, height = 180, 1180, 1720, 760
    draw.line((left, 250, left, bottom), fill="#94a3b8", width=5)
    draw.line((left, bottom, left + width, bottom), fill="#94a3b8", width=5)
    draw.line((left, bottom - height * 0.2, left + width, bottom - height * 0.2), fill="#ef4444", width=4)
    draw.text((left + width + 20, bottom - height * 0.2 - 18), "20%", fill="#ef4444", font=font(26, True))

    bar_w = 230
    gap = 90
    x = left + 70
    for cfg in configs:
        val = cfg["error_percent"]
        bh = int((val / 25.0) * height)
        draw.rounded_rectangle((x, bottom - bh, x + bar_w, bottom), radius=18, fill="#2563eb")
        draw.text((x + 45, bottom - bh - 50), f"{val:.2f}%", fill="#111827", font=font(24, True))
        draw.text((x + 8, bottom + 25), cfg["id"], fill="#334155", font=font(22))
        x += bar_w + gap

    save(img, os.path.join(CHART_DIR, "error_compare.png"))


def draw_runtime(model):
    img, draw = canvas("Measured vs Predicted Training Iteration Time")
    configs = model["configs"]
    left, top, row_h = 130, 200, 210
    max_ms = max(max(c["t_real_ms"], c["t_sim_ms"]) for c in configs)

    for idx, cfg in enumerate(configs):
        y = top + idx * row_h
        draw.text((left, y + 56), cfg["id"], fill="#0f172a", font=font(28, True))
        vals = [
            ("Measured", cfg["t_real_ms"], "#2563eb", y + 25),
            ("Predicted", cfg["t_sim_ms"], "#93c5fd", y + 82),
        ]
        for label, val, color, yy in vals:
            bw = int((val / max_ms) * 1180)
            draw.rounded_rectangle((460, yy, 460 + bw, yy + 30), radius=12, fill=color)
            draw.text((350, yy - 2), label, fill="#334155", font=font(24))
            draw.text((475 + bw + 10, yy - 2), f"{val:.3f} ms", fill="#111827", font=font(24))

    save(img, os.path.join(CHART_DIR, "runtime_compare.png"))


def draw_model(model):
    img, draw = canvas("Training-Time Predictor")
    draw.rounded_rectangle((160, 230, 1120, 620), radius=26, fill="#dbeafe", outline="#60a5fa", width=4)
    draw.text((210, 300), "Predictor Entry", fill="#1d4ed8", font=font(40, True))
    draw.text((210, 390), "train-infer-estimation-release-2026-04-11", fill="#0f172a", font=font(28))
    draw.text((210, 455), "torch_train_mvp.py", fill="#0f172a", font=font(34, True))

    draw.rounded_rectangle((1220, 230, 2020, 720), radius=26, fill="#dcfce7", outline="#4ade80", width=4)
    draw.text((1260, 300), "Prediction Formula", fill="#166534", font=font(36, True))
    draw.text((1260, 390), "T_sim = microbatch_num *", fill="#166534", font=font(30))
    draw.text((1260, 450), "independently profiled", fill="#166534", font=font(30))
    draw.text((1260, 510), "microbatch_time_ms", fill="#166534", font=font(30, True))
    draw.text((1260, 600), "Inputs: model / parallel config / hardware topology", fill="#166534", font=font(24))

    draw.text((180, 820), "PP: pipeline parallel size, MB: microbatch count", fill="#334155", font=font(28))
    save(img, os.path.join(CHART_DIR, "time_model.png"))


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "benchmark_results.json"))
    model = load_json(os.path.join(artifact, "time_model_results.json"))

    draw_status(bench["environment"]["mode"])
    draw_error(model)
    draw_runtime(model)
    draw_model(model)

    with open(os.path.join(ROOT, "5.2.14图表汇总.md"), "w", encoding="utf-8") as f:
        f.write(
            "# 5.2.14 图表汇总\n\n"
            "![状态](charts/status_af.png)\n"
            "![误差](charts/error_compare.png)\n"
            "![时间](charts/runtime_compare.png)\n"
            "![模型](charts/time_model.png)\n"
        )


if __name__ == "__main__":
    main()
