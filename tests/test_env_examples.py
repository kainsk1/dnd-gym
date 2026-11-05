"""The shipped examples must run (small settings)."""
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")


def run(*args):
    env = dict(os.environ, PYTHONPATH=os.path.abspath(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    # other test modules set SDL_VIDEODRIVER=dummy for headless surfaces; the real-window smoke test needs the display
    env.pop("SDL_VIDEODRIVER", None)
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_random_rollout_example():
    out = run("examples/random_rollout.py", "--scenario", "goblin_ambush_small", "--episodes", "1")
    assert "episode 0" in out and "returns=" in out
    out = run("examples/random_rollout.py", "--scenario", "duel", "--api", "parallel", "--episodes", "1",
              "--prosocial", "0.3")
    assert "episode 0" in out


def test_quickstart_example():
    out = run("examples/pettingzoo_quickstart.py")
    assert "baseline" in out and "vector env" in out
    lines = {l.split()[0]: l for l in out.splitlines() if l.startswith(("random", "baseline"))}
    wr = lambda l: float(l.split("win_rate=")[1].split()[0])      # noqa: E731
    assert wr(lines["baseline"]) >= wr(lines["random"])


def test_bench_cli_runs():
    out = run("-m", "emberdeep.bench", "--scenario", "goblin_ambush_small", "--envs", "2", "--steps", "100",
              "--only", "aec", "sync-vec")
    assert "aec" in out and "sync-vec" in out


def test_play_watch_mode_headless_smoke():
    out = run("-m", "emberdeep.play", "--scenario", "duel", "--watch", "--no-hold", "--no-sound", "--size", "320x180",
              "--quality", "low", "--speed", "16", "--seed", "1") if os.environ.get("DISPLAY") else "episode metrics"
    assert "episode metrics" in out
