#!/usr/bin/env python3
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "datasets" / "sintel" / "training"
    out = root.parent.parent
    scenes = sorted(p.name for p in (root / "flow").iterdir() if p.is_dir())
    all_host = []
    for render in ("clean", "final"):
        lines = []
        for scene in scenes:
            for flo in sorted((root / "flow" / scene).glob("frame_*.flo")):
                number = int(flo.stem.split("_")[1])
                image0 = root / render / scene / f"frame_{number:04d}.png"
                image1 = root / render / scene / f"frame_{number + 1:04d}.png"
                if image0.exists() and image1.exists():
                    lines.append(
                        f"{image0.resolve()} {image1.resolve()} {flo.resolve()} {scene}\n"
                    )
        (out / f"sintel_{render}_host_manifest.txt").write_text("".join(lines))
        board_lines = [
            line.replace(str(root.resolve()), "/tmp/sintel_eval/training")
            for line in lines
        ]
        (out / f"sintel_{render}_board_manifest.txt").write_text("".join(board_lines))
        all_host.extend(lines)
    (out / "sintel_all_host_manifest.txt").write_text("".join(all_host))
    print(f"scenes={len(scenes)}")
    print(f"clean_pairs={sum(1 for _ in (out / 'sintel_clean_host_manifest.txt').open())}")
    print(f"final_pairs={sum(1 for _ in (out / 'sintel_final_host_manifest.txt').open())}")


if __name__ == "__main__":
    main()
