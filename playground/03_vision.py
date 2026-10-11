"""D1 练习 3：图片理解 —— 提前把 D4「文档摄取」的路摸清楚。

跑法：
    uv run python playground/03_vision.py                     # 用自带的 sample.png
    uv run python playground/03_vision.py 某张截图.png          # 用自己的图

这个脚本要回答三个问题：
  ① 图片是怎么变成 token 的？一张图大概多少 token？
  ② 同一个模型能做哪些"看图"的活：描述 / OCR / 结构化抽取？
  ③ 既然视觉模型能读文档，为什么 D4 的文档摄取不能全走视觉模型？（成本）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_core.llm import ChatResult, LLMClient, estimate_cost_cny

SAMPLE = Path(__file__).with_name("sample.png")


def generate_sample(path: Path) -> None:
    """生成一张模拟「采购订单」的图，用来做 OCR 与结构化抽取的测试。

    只有在你把 sample.png 删掉时才会触发。
    需要 Pillow：uv run --with pillow python playground/03_vision.py
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "缺少 Pillow，无法生成测试图。二选一：\n"
            "  a) uv run --with pillow python playground/03_vision.py\n"
            "  b) 自己放一张图：uv run python playground/03_vision.py 你的图.png"
        ) from exc

    width, height = 900, 560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    def font(size: int, bold: bool = False):  # type: ignore[no-untyped-def]
        for candidate in (
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
        ):
            if Path(candidate).exists():
                try:
                    return ImageFont.truetype(candidate, size, index=1 if bold else 0)
                except OSError:
                    continue
        return ImageFont.load_default()

    draw.text((40, 30), "采购订单 / Purchase Order", font=font(30, bold=True), fill="black")
    draw.line((40, 80, width - 40, 80), fill="black", width=2)

    rows = [
        ("订单编号 Order No.", "PO-2026-1012-0087"),
        ("供应商 Vendor", "杭州云栖智能科技有限公司"),
        ("下单日期 Date", "2026-10-12"),
        ("物料 Material", "铝合金外壳 AL-6061"),
        ("数量 Qty", "1,250 pcs"),
        ("单价 Unit Price", "CNY 38.50"),
        ("合计金额 Total", "CNY 48,125.00"),
        ("交货期 Delivery", "2026-11-20"),
    ]
    y = 110
    for label, value in rows:
        draw.text((50, y), label, font=font(21), fill="#333333")
        draw.text((380, y), value, font=font(21), fill="black")
        y += 44

    draw.line((40, y + 6, width - 40, y + 6), fill="#999999", width=1)
    draw.text((50, y + 20), "备注：需提供材质证明与 RoHS 报告。", font=font(19), fill="#555555")

    img.save(path)
    print(f"（已自动生成测试图 {path}）")


# 三个任务，难度递增：自由描述 -> 逐字 OCR -> 结构化抽取
#
# 每个任务都设了 max_tokens 上限 —— 这不是可选项，是必须项。
# 原因是我实测到的：同一个 prompt、同样 temperature=0.0，逐字 OCR 那一步
# 两次运行分别输出了 308 tok/9.4s 和 1393 tok/32s，成本差 3 倍。
# 也就是说「输出长度不可预测」→「成本不可预测」。
# 因此任何面向生产的调用都必须有输出上限，否则一次病态的长输出就能吃掉整个预算。
TASKS: list[tuple[str, str, int]] = [
    ("自由描述", "用两句话描述这张图片的内容。", 200),
    ("逐字 OCR", "提取图中所有文字，按「字段：值」的格式逐行输出，不要遗漏，不要解释。", 400),
    (
        "结构化抽取",
        "把图中信息抽取成 JSON，字段为 order_no / vendor / date / material / qty / "
        "unit_price / total / delivery / note。只输出 JSON，不要 markdown 代码块，"
        "不要任何解释文字。读不到的字段填 null。",
        500,
    ),
]


def report(label: str, r: ChatResult, model: str) -> None:
    cost = estimate_cost_cny(r.usage, model)
    print(f"\n{'─' * 72}")
    print(f"【{label}】")
    print(f"{'─' * 72}")
    print(r.text.strip())
    print(
        f"\n  ← 输入 {r.usage.prompt_tokens} tok / 输出 {r.usage.completion_tokens} tok"
        f" / {r.latency_ms:.0f} ms"
        f" / 成本 {cost:.6f} 元"
        if cost is not None
        else " / 成本 未知（未登记价格）"
    )


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
    if not image_path.exists():
        if image_path == SAMPLE:
            generate_sample(image_path)
        else:
            raise SystemExit(f"图片不存在：{image_path}")

    size_kb = image_path.stat().st_size / 1024
    client = LLMClient()
    model = client.settings.dashscope_vision_model

    print("=" * 72)
    print(f"图片: {image_path}  ({size_kb:.0f} KB)")
    print(f"视觉模型: {model}")
    print("=" * 72)

    results: dict[str, ChatResult] = {}
    for label, prompt, limit in TASKS:
        try:
            r = client.chat_with_image(prompt, image_path, temperature=0.0, max_tokens=limit)
        except Exception as exc:
            print(f"\n【{label}】失败：{type(exc).__name__}: {exc}")
            print("  → 检查 .env 的 DASHSCOPE_VISION_MODEL 是否是控制台里可用的模型名")
            continue
        results[label] = r
        report(label, r, model)
        if r.finish_reason == "length":
            print(
                f"  ⚠️ 输出被 max_tokens={limit} 截断了（finish_reason=length）。\n"
                "     生产环境里这种情况必须当成一次失败：要么调大上限，要么让模型输出更短。"
            )

    # ------------------------------------------------------------------
    # ③ 成本视角：这才是 D4 文档摄取要做的决策
    # ------------------------------------------------------------------
    if "自由描述" in results:
        per_image = results["自由描述"].usage.prompt_tokens
        print(f"\n{'=' * 72}")
        print("成本视角（D4 文档摄取要用这个数字做决策）")
        print(f"{'=' * 72}")
        print(f"  一张 {size_kb:.0f} KB 的图 ≈ {per_image} 输入 token（含提示词）")
        for pages in (10, 100, 1000):
            total = per_image * pages
            print(f"  {pages:>4} 页文档若全走视觉模型 ≈ {total:>9,} token")

        print(
            "\n  结论（写进 D4 的设计决策）：\n"
            "    · 文字版 PDF / HTML / Markdown → 直接抽文本，不要走视觉模型，几乎免费\n"
            "    · 只有扫描件、截图、照片文档才走视觉模型\n"
            "    · 走视觉模型时优先批量/降采样：把多页拼一张图，或用更小的图片尺寸\n"
            "    · 这个取舍必须写进 D5 的摄取策略，并在 trace 里按来源分别记成本"
        )

    # ------------------------------------------------------------------
    # 结构化抽取的可解析率（D2 的伏笔）
    # ------------------------------------------------------------------
    if "结构化抽取" in results:
        raw = results["结构化抽取"].text.strip()
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        print(f"\n{'=' * 72}")
        print("结构化抽取的可解析率（D2 要解决的问题）")
        print(f"{'=' * 72}")
        try:
            data = json.loads(cleaned)
            got = sum(1 for v in data.values() if v is not None)
            print(f"  直接 json.loads 成功，抽出 {got}/{len(data)} 个字段")
            print(f"  {json.dumps(data, ensure_ascii=False)[:200]}")
        except json.JSONDecodeError as exc:
            print(f"  json.loads 失败：{exc}")
            print(f"  原始输出前 200 字：{raw[:200]}")
            print(
                "\n  → 这正是 D2 结构化输出要解决的问题：\n"
                "    靠 prompt 说「只输出 JSON」是不可靠的。要用 response_format / "
                "json_schema 或 function calling，再加 Pydantic 校验 + 失败重试。"
            )


if __name__ == "__main__":
    main()
