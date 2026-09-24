#!/usr/bin/env python3
from pathlib import Path
import shutil


def main():
    base = Path(__file__).resolve().parent
    datasets = base.parent / "datasets"
    data_root = datasets / "sintel" / "training"
    subset_root = datasets / "sintel_board_subset" / "training"
    manifest = datasets / "sintel_clean_host_manifest.txt"
    board_manifest = datasets / "sintel_clean_board_subset_manifest.txt"
    max_pairs_per_scene = 5
    selected = []
    counts = {}
    for line in manifest.read_text().splitlines():
        image0, image1, flow, scene = line.split()
        if counts.get(scene, 0) >= max_pairs_per_scene:
            continue
        selected.append((Path(image0), Path(image1), Path(flow), scene))
        counts[scene] = counts.get(scene, 0) + 1

    board_lines = []
    for image0, image1, flow, scene in selected:
        for source in (image0, image1, flow):
            relative = source.relative_to(data_root)
            target = subset_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
        b0 = "/tmp/sintel_eval/training/" + str(image0.relative_to(data_root))
        b1 = "/tmp/sintel_eval/training/" + str(image1.relative_to(data_root))
        bf = "/tmp/sintel_eval/training/" + str(flow.relative_to(data_root))
        board_lines.append(f"{b0} {b1} {bf} {scene}\n")
    board_manifest.write_text("".join(board_lines))
    print(f"pairs={len(selected)} scenes={len(counts)}")
    print(f"subset={subset_root}")


if __name__ == "__main__":
    main()
