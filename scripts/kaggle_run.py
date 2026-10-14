"""One command: upload camo-code, run the pilot notebook on a Kaggle GPU, download the zip into outputs/.

    python scripts/kaggle_run.py --run pilot_v2 --configs pilot/audit_v2.yaml

Needs the Kaggle CLI (`pip install kaggle`) and a token: ~/.kaggle/access_token (Settings -> API Tokens) or the
legacy ~/.kaggle/kaggle.json. With an access token pass --user <kaggle username>.
The run is a background "Save & Run All" version of a separate private notebook `<user>/camo-pilot-run`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AISEGMENT = "laurentmih/aisegmentcom-matting-human-datasets"


def sh(*cmd: str, check: bool = True, timeout: float = 900) -> str:
    print("$", " ".join(cmd), flush=True)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if check:
            sys.exit(f"timed out after {timeout:.0f}s: {' '.join(cmd)}")
        return "timeout"
    if check and out.returncode != 0:
        sys.exit(f"{out.stdout}\n{out.stderr}")
    return out.stdout + out.stderr


def username(given: str = "") -> str:
    if given:
        return given
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for path in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
        path = os.path.expanduser(path)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)["username"]
    sys.exit("Kaggle username unknown: pass --user <name> (needed with ~/.kaggle/access_token)")


# Per-trial debug images (~0.9 MB of ~1.2 MB per trial) that labelling and training do not read.
SLIM_DROP = ("background.png", "subject.png", "canny.png", "softedge.png", "mask_1_smoothed.png", "mask_2_remapped.png",
             "mask_overlay.png", "panel.png")


def sha256(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def expected_files(configs: str, frozen: str) -> dict:
    """{path relative to the code root: sha256} that the kernel must see in camo-code."""
    paths = ([configs] if configs else []) + [f"{frozen}/inputs.yaml"]
    return {p: sha256(os.path.join(ROOT, p)) for p in paths if os.path.exists(os.path.join(ROOT, p))}


def dataset_current(user: str, expect: dict) -> bool:
    """Does the newest camo-code version Kaggle serves hold these files? Right after `datasets version` the old
    version is still the one served and mounted: a kernel pushed 5 s after the upload rendered the old grid."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for rel, digest in expect.items():
            sh("kaggle", "datasets", "download", f"{user}/camo-code", "-f", rel, "-p", tmp, "--force", check=False, timeout=180)
            got = os.path.join(tmp, os.path.basename(rel))
            if zipfile.is_zipfile(got):  # the API zips some single-file downloads
                with zipfile.ZipFile(got) as z:
                    z.extractall(tmp)
            if not os.path.exists(got) or sha256(got) != digest:
                return False
    return True


def wait_dataset(user: str, expect: dict, minutes: float = 20) -> None:
    deadline = time.time() + minutes * 60
    while not dataset_current(user, expect):
        if time.time() > deadline:
            sys.exit(f"camo-code on Kaggle still lacks the new {', '.join(expect)} after {minutes:.0f} min")
        print("camo-code: new version not served yet, waiting", flush=True)
        time.sleep(30)
    print("camo-code: new version is live", flush=True)


def dataset_sources(user: str, step: str, frozen: str) -> list:
    """Datasets the kernel mounts. AISegment (~68k files) is only read when inputs are fetched; a run on frozen inputs
    (the label set) mounts the code alone: one dataset less to fail with ERRORED_MOUNTING_DATASET (13/10, twice)."""
    fetches = step == "fetch-inputs" or not os.path.exists(os.path.join(ROOT, frozen, "inputs.yaml"))
    return [f"{user}/camo-code"] + ([AISEGMENT] if fetches else [])


def wait_kernel(kernel: str, poll: float = 60.0) -> str:
    """Block while the kernel is queued or running; a status call that fails (offline) counts as still running."""
    while True:
        status = sh("kaggle", "kernels", "status", kernel, check=False, timeout=120).lower()
        state = re.search(r"kernelworkerstatus\.(\w+)", status)
        print(time.strftime("%H:%M"), state.group(0) if state else "status unknown (offline?)", flush=True)
        if state and state.group(1) not in ("running", "queued"):
            return state.group(1)
        time.sleep(poll)


def notebook(run: str, configs: str, pick: dict, step: str, frozen: str = "pilot/frozen_inputs", slim: bool = False,
             expect: dict | None = None) -> dict:
    with open(os.path.join(ROOT, "kaggle", "camo_pilot.ipynb"), encoding="utf-8") as f:
        nb = json.load(f)
    src = "".join(nb["cells"][1]["source"])
    src = re.sub(r'^STEP = .*$', f'STEP = "{step}"', src, count=1, flags=re.M)
    src = re.sub(r'^PICK = .*$', f"PICK = {json.dumps(pick)}", src, count=1, flags=re.M)
    src = re.sub(r'^RUN = .*$', f'RUN = "{run}"', src, count=1, flags=re.M)
    src = re.sub(r'^CONFIGS = .*$', f'CONFIGS = "{configs}"', src, count=1, flags=re.M)
    src = src.replace('FROZEN = f"{CODE}/pilot/frozen_inputs"', f'FROZEN = f"{{CODE}}/{frozen}"')
    if expect:  # fail in seconds on a stale camo-code mount instead of rendering the old inputs for hours
        frozen_line = f'FROZEN = f"{{CODE}}/{frozen}"'
        assert frozen_line in src, "notebook FROZEN line changed: update the stale-code check"
        check = (f"\nimport hashlib\nEXPECT = {json.dumps(expect)}\n"
                 "STALE = [p for p, h in EXPECT.items() if not os.path.exists(f'{CODE}/{p}')\n"
                 "         or hashlib.sha256(open(f'{CODE}/{p}', 'rb').read()).hexdigest() != h]\n"
                 "if STALE:\n    raise RuntimeError(f'STALE_CAMO_CODE: {STALE}')")
        src = src.replace(frozen_line, frozen_line + check, 1)
    if slim:  # drop debug images before zipping so each batch downloads ~4x faster
        names = " -o ".join(f"-name {n}" for n in SLIM_DROP)
        zip_line = "    !cd /kaggle/working && rm -f {RUN}.zip"
        assert zip_line in src, "notebook zip line changed: update --slim"
        src = src.replace(zip_line, f"    !find {{OUT}}/runs \\( {names} \\) -delete\n" + zip_line, 1)
    # Batch runs save all of /kaggle/working as output: drop the download cache (~1-2 GB) so the output stays small.
    src += '\nif os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch":\n    !rm -rf {GEN}/_cache\n'
    nb["cells"][1]["source"] = src.splitlines(keepends=True)
    return nb


def serve_notebook(token: str, idle: float) -> dict:
    """Notebook that only starts camo/render_server.py (GPU endpoint for the local demo) and prints RENDER_URL."""
    src = (
        "import glob, os\n"
        "CODE = next((os.path.dirname(p) for d in range(1, 7) for p in sorted(glob.glob('/kaggle/input' + '/*' * d + '/run.sh'))\n"
        "             if os.path.isfile(os.path.join(os.path.dirname(p), 'camo', 'render_server.py'))), None)\n"
        "assert CODE, 'camo-code dataset without camo/render_server.py'\n"
        f"!cd {{CODE}} && bash run.sh render_server --token {token} --idle {idle}\n"
    )
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": src.splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="pilot_v2", help="output folder/zip name")
    ap.add_argument("--configs", default="pilot/audit_v2.yaml", help='"" = original C00-C06 table')
    ap.add_argument("--pick", default=os.path.join(ROOT, "pilot", "pick.json"), help="JSON {ID: candidate index}")
    ap.add_argument("--step", default="all")
    ap.add_argument("--frozen", default="pilot/frozen_inputs", help="input folder shipped with the code (inputs.yaml + images)")
    ap.add_argument("--slim", action="store_true", help="drop per-trial debug images before zipping (smaller download)")
    ap.add_argument("--user", default="", help="Kaggle username (required with the new access_token)")
    ap.add_argument("--skip-upload", action="store_true", help="camo-code on Kaggle is already current")
    ap.add_argument("--no-wait", action="store_true", help="push and return; download later with --download-only")
    ap.add_argument("--download-only", action="store_true", help="wait for the running kernel, then download")
    ap.add_argument("--download-hours", type=float, default=12.0,
                    help="keep retrying the download this long (the machine may be offline for hours)")
    ap.add_argument("--serve", action="store_true", help="start the GPU render server for camo.demo (no waiting)")
    ap.add_argument("--serve-url", help="save the RENDER_URL printed in the Kaggle log for camo.demo")
    ap.add_argument("--idle", type=float, default=40.0, help="--serve: minutes without requests before it stops")
    args = ap.parse_args()
    if args.serve_url:
        with open(os.path.join(ROOT, "outputs", ".render_url"), "w") as f:
            f.write(args.serve_url.strip())
        print("Saved outputs/.render_url; now run: python3 -m camo.demo")
        return 0
    if not shutil.which("kaggle"):
        sys.exit("Kaggle CLI missing: pip install kaggle")
    user = username(args.user)
    kernel = f"{user}/camo-pilot-run"
    out_dir = os.path.join(ROOT, "outputs")

    if args.serve:
        import secrets

        if not args.skip_upload:
            os.environ["KAGGLE_USERNAME"] = user
            sh("bash", os.path.join(ROOT, "scripts", "pack_kaggle.sh"), "--upload")
            for _ in range(60):
                if "ready" in sh("kaggle", "datasets", "status", f"{user}/camo-code", check=False).lower():
                    break
                time.sleep(10)
        token = secrets.token_urlsafe(16)
        os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
        with open(os.path.join(ROOT, "outputs", ".render_token"), "w") as f:
            f.write(token)
        kdir = os.path.join(ROOT, "dist", "kernel_serve")
        shutil.rmtree(kdir, ignore_errors=True)
        os.makedirs(kdir)
        with open(os.path.join(kdir, "camo_render_server.ipynb"), "w", encoding="utf-8") as f:
            json.dump(serve_notebook(token, args.idle), f, indent=1)
        with open(os.path.join(kdir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"id": f"{user}/camo-render-server", "title": "camo-render-server", "code_file": "camo_render_server.ipynb",
                       "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                       "enable_internet": True, "dataset_sources": [f"{user}/camo-code"], "competition_sources": [],
                       "kernel_sources": []}, f, indent=1)
        sh("kaggle", "kernels", "push", "-p", kdir)
        print(f"Render server starting (~3 min to load models): https://www.kaggle.com/code/{user}/camo-render-server/log")
        print("Copy the RENDER_URL=https://....gradio.live line from that log, then:")
        print("  python3 scripts/kaggle_run.py --serve-url https://....gradio.live && python3 -m camo.demo")
        print(f"It stops by itself after {args.idle:.0f} min without renders. Token saved in outputs/.render_token")
        return 0

    if not args.download_only:
        expect = expected_files(args.configs, args.frozen)
        os.environ["KAGGLE_USERNAME"] = user  # pack_kaggle.sh --upload reads it
        if args.skip_upload and not dataset_current(user, expect):
            print("camo-code on Kaggle is not current despite --skip-upload: uploading", flush=True)
            args.skip_upload = False
        if not args.skip_upload:
            sh("bash", os.path.join(ROOT, "scripts", "pack_kaggle.sh"), "--upload")
            wait_dataset(user, expect)
        with open(args.pick, encoding="utf-8") as f:
            pick = json.load(f)
        kdir = os.path.join(ROOT, "dist", "kernel")
        shutil.rmtree(kdir, ignore_errors=True)
        os.makedirs(kdir)
        with open(os.path.join(kdir, "camo_pilot_run.ipynb"), "w", encoding="utf-8") as f:
            json.dump(notebook(args.run, args.configs, pick, args.step, args.frozen, args.slim, expect), f,
                      ensure_ascii=False, indent=1)
        with open(os.path.join(kdir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"id": kernel, "title": "camo-pilot-run", "code_file": "camo_pilot_run.ipynb", "language": "python",
                       "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
                       "dataset_sources": dataset_sources(user, args.step, args.frozen), "competition_sources": [],
                       "kernel_sources": []}, f, indent=1)
        for push in range(1, 6):
            sh("kaggle", "kernels", "push", "-p", kdir)
            print(f"Running: https://www.kaggle.com/code/{kernel}")
            if args.no_wait:
                return 0
            stale = False
            while True:
                time.sleep(60)
                status = sh("kaggle", "kernels", "status", kernel, check=False).lower()
                print(time.strftime("%H:%M"), status.strip().splitlines()[-1] if status.strip() else "?", flush=True)
                if "complete" in status:
                    break
                # match the kernel state only: a network error while polling (wifi change) is not a failed kernel
                if "kernelworkerstatus.error" in status or "kernelworkerstatus.cancel" in status:
                    log_dir = os.path.join(out_dir, f"{args.run}_log")
                    shutil.rmtree(log_dir, ignore_errors=True)  # an older STALE log must not decide this one
                    sh("kaggle", "kernels", "output", kernel, "-p", log_dir, check=False, timeout=180)
                    logs = "".join(open(os.path.join(log_dir, n), encoding="utf-8", errors="replace").read()
                                   for n in os.listdir(log_dir) if n.endswith(".log")) if os.path.isdir(log_dir) else ""
                    if "STALE_CAMO_CODE" not in logs:
                        sys.exit(f"Kernel failed: see https://www.kaggle.com/code/{kernel} (log in outputs/{args.run}_log)")
                    stale = True
                    break
            if not stale:
                break
            print(f"Kernel mounted an old camo-code (push {push}): waiting 2 min, then pushing again", flush=True)
            time.sleep(120)
        else:
            sys.exit("Kaggle kept mounting an old camo-code version: re-run later")

    if args.download_only:  # resumed runner or a re-download: the kernel may still be rendering
        final = wait_kernel(kernel)
        if final == "error" or final.startswith("cancel"):  # nothing to download: render again instead of waiting
            sys.exit(f"Kernel ended with status {final}: no output to download; see https://www.kaggle.com/code/{kernel}")
    os.makedirs(out_dir, exist_ok=True)
    deadline = time.time() + args.download_hours * 3600
    while True:  # retry through network outages (wifi switched off for hours): time-bounded, not count-bounded
        got = sh("kaggle", "kernels", "output", kernel, "-p", out_dir, "--file-pattern", rf"^{re.escape(args.run)}\.zip$",
                 check=False, timeout=1800)
        zpath = os.path.join(out_dir, f"{args.run}.zip")
        if zipfile.is_zipfile(zpath) or "unrecognized arguments" in got or time.time() > deadline:
            break
        if os.path.exists(zpath):  # a dropped connection leaves a truncated (or empty) zip
            os.remove(zpath)
        time.sleep(60)
    if "unrecognized arguments" in got:  # older CLI: whole output (small, the cache was removed in the run)
        sh("kaggle", "kernels", "output", kernel, "-p", out_dir)
    zpath = os.path.join(out_dir, f"{args.run}.zip")
    if not zipfile.is_zipfile(zpath):
        sys.exit(f"{zpath} not downloaded; check https://www.kaggle.com/code/{kernel} -> Output")
    shutil.unpack_archive(zpath, out_dir)
    run_dir = os.path.join(out_dir, args.run)
    statuses = []
    for root, _dirs, files in os.walk(os.path.join(run_dir, "runs")):
        if "trial.json" in files:
            with open(os.path.join(root, "trial.json"), encoding="utf-8") as f:
                statuses.append(json.load(f).get("status"))
    ok = statuses.count("ok")
    print(f"Done: {run_dir} ({ok} ok / {len(statuses)} trials)")
    if ok == 0:
        sys.exit("No rendered trial in the output: check the Kaggle log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
