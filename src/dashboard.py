"""Pretty training dashboard for the descent runs — rich-coloured goal banner + a per-model
scoreboard + an end-of-run verdict.

Design notes:
- **Cosmetic only.** Every public entry point is wrapped in try/except so a rendering failure
  can NEVER abort training (the diary + sacred number are what matter).
- **Render-on-update, not Live.** We re-print a fresh table after each model instead of using
  rich.Live — Live's cursor-movement escape codes render glitchy through Colab's subprocess pipe
  (duplicated/overdrawn lines). `force_terminal=True` makes rich emit ANSI colours, which Colab's
  cell output *does* render; it only chokes on the cursor tricks we avoid here.
- **Degrades gracefully** to plain unicode if `rich` is unavailable or output isn't a colour TTY.

The "goal" = the descent anchors below: carry-forward floor → v5 ensemble → winning-pool TARGET.
Keep these in sync with docs/roadmap_to_9.5.md.
"""
from __future__ import annotations

FLOOR = 14.08   # carry-forward (Δ=0) sacred RMSE — the bar we beat
V5 = 9.155      # v5_ensemble sacred (255 / stride-8) — current best reference
TARGET = 8.20   # winning-pool target (reset 2026-05-27 after first LB)
V5_RUNTIME_S = 90 * 60  # ~90 min reference for the ⚡ speed-up readout

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _C = Console(force_terminal=True)  # force ANSI colours through Colab's non-TTY pipe
    _RICH = True
except Exception:  # rich missing → plain-text fallback
    _RICH = False
    _C = None


def _pct(x: float) -> float:
    """Fraction of the floor→target distance covered (clamped 0..1)."""
    if FLOOR <= TARGET:
        return 0.0
    return max(0.0, min(1.0, (FLOOR - x) / (FLOOR - TARGET)))


def _bar(x: float, width: int = 22) -> str:
    n = int(round(_pct(x) * width))
    return "█" * n + "░" * (width - n)


def _style_for(delta_vs_v5: float) -> tuple[str, str]:
    """(rich-style, marker) by how a sacred compares to v5: ≤+0.05 good, ≤+0.4 ok, else bad."""
    if delta_vs_v5 <= 0.05:
        return "green", "✅"
    if delta_vs_v5 <= 0.40:
        return "yellow", "⚠"
    return "red", "✗"


def goal_banner(ver: str, run_desc: str, test_desc: str) -> None:
    """Print the descent 'goal' at run start: floor → v5 → target, where we are, what this run tests."""
    try:
        if _RICH:
            body = Text()
            body.append(f"floor {FLOOR:.2f}", style="dim")
            body.append("    ")
            body.append(f"v5 {V5:.3f}", style="cyan")
            body.append("    ")
            body.append(f"◆ TARGET {TARGET:.2f}\n", style="bold green")
            body.append(f"{FLOOR:.2f} ")
            body.append(_bar(V5), style="yellow")
            body.append(f" {TARGET:.2f}", style="dim")
            body.append(f"   {_pct(V5) * 100:.0f}% there (at v5)\n", style="dim")
            body.append("this run: ", style="dim")
            body.append(f"{run_desc}\n", style="bold")
            body.append("testing : ", style="dim")
            body.append(test_desc, style="italic")
            _C.print(Panel(body, title=f"[bold]ROGII · Descent to {TARGET:.1f}[/]  ·  {ver}",
                           border_style="green", box=box.ROUNDED))
        else:
            print(f"\n=== ROGII · Descent to {TARGET:.1f} · {ver} ===")
            print(f"floor {FLOOR:.2f} | v5 {V5:.3f} | TARGET {TARGET:.2f}  ({_pct(V5)*100:.0f}% there at v5)")
            print(f"this run: {run_desc}\ntesting : {test_desc}\n")
    except Exception:
        pass


def training(name: str, i: int, n: int) -> None:
    """A one-line 'now training model i/n' marker before each fit."""
    try:
        msg = f"▶ training {name}  ({i}/{n})…"
        _C.print(msg, style="bold blue") if _RICH else print(msg)
    except Exception:
        pass


def scoreboard(rows: list[dict]) -> None:
    """Re-print the per-model table. rows: [{name, oof, sacred}, ...] (newest run included)."""
    try:
        if _RICH:
            t = Table(box=box.SIMPLE_HEAVY, title="models so far", title_style="dim")
            t.add_column("model"); t.add_column("dev_oof", justify="right")
            t.add_column("sacred", justify="right"); t.add_column("vs v5", justify="right")
            t.add_column(f"→{TARGET:.1f}")
            for r in rows:
                d = r["sacred"] - V5
                style, mark = _style_for(d)
                t.add_row(r["name"], f"{r['oof']:.3f}", f"[{style}]{r['sacred']:.3f}[/]",
                          f"[{style}]{d:+.3f} {mark}[/]", _bar(r["sacred"], 10))
            _C.print(t)
        else:
            for r in rows:
                d = r["sacred"] - V5
                print(f"  {r['name']}: oof {r['oof']:.3f} | sacred {r['sacred']:.3f} ({d:+.3f} vs v5)")
    except Exception:
        pass


def verdict(ver: str, sacred: float, runtime_sec: float,
            simple_avg: float | None = None, parent: float = V5,
            parent_runtime_s: float = V5_RUNTIME_S) -> None:
    """End-of-run panel: the blend sacred, deltas vs v5 + target, runtime/speed-up, descent bar."""
    try:
        d_parent = sacred - parent
        d_target = sacred - TARGET
        if sacred <= TARGET:
            v_txt = "✅ beat target!"
        elif d_parent < -0.005:
            v_txt = f"📉 improved (−{-d_parent:.3f} vs v5)"
        elif abs(d_parent) <= 0.05:
            v_txt = "≈ matched v5"
        else:
            v_txt = f"⚠ above v5 (+{d_parent:.3f})"
        speed = f"  ⚡{parent_runtime_s / runtime_sec:.1f}x vs v5" if runtime_sec > 0 else ""
        if _RICH:
            body = Text()
            body.append(f"SACRED blend : {sacred:.3f}", style="bold")
            if simple_avg is not None:
                body.append(f"   (simple-avg {simple_avg:.3f})", style="dim")
            body.append("\n")
            body.append(f"vs v5 {parent:.3f} : {d_parent:+.3f}\n",
                        style="green" if d_parent <= 0.05 else "yellow")
            body.append(f"vs target {TARGET:.2f}: {d_target:+.3f}",
                        style="green" if d_target <= 0 else "cyan")
            body.append(f"   ({max(0.0, d_target):.2f} ft to go)\n", style="dim")
            body.append(f"runtime      : {runtime_sec / 60:.0f} min{speed}\n", style="dim")
            body.append(f"descent: {FLOOR:.2f} ")
            body.append(_bar(sacred), style="bold green")
            body.append(f" {TARGET:.2f}  ({_pct(sacred) * 100:.0f}% there)\n", style="dim")
            body.append("VERDICT: ", style="dim")
            body.append(v_txt, style="bold")
            _C.print(Panel(body, title=f"[bold]{ver} — RESULT[/]",
                           border_style="green" if d_parent <= 0.05 else "yellow", box=box.DOUBLE))
        else:
            extra = f" | simple-avg {simple_avg:.3f}" if simple_avg is not None else ""
            print(f"\n=== {ver} RESULT: SACRED {sacred:.3f}{extra} | vs v5 {d_parent:+.3f} | "
                  f"to target {max(0.0, d_target):.2f} ft | {runtime_sec/60:.0f} min{speed} | {v_txt} ===")
    except Exception:
        pass
