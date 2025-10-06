#!/usr/bin/env python3
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict


def read_text(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return f.read().splitlines()


def write_text(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_instance_inline(line: str) -> Dict[str, int]:
    """Parse inline instance mapping from a line like:
      "  instance: {C: 768, M: 768, P: 197}"
    Returns dict with int values when possible.
    """
    mapping: Dict[str, int] = {}
    if "instance:" not in line:
        return mapping
    # Get text after colon
    after = line.split(":", 1)[1].strip()
    if not (after.startswith("{") and after.endswith("}")):
        return mapping
    inner = after[1:-1]
    for part in inner.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            mapping[k] = int(v)
        except ValueError:
            # keep raw if not int
            try:
                mapping[k] = float(v)
            except ValueError:
                mapping[k] = v
    return mapping


def parse_hist_array(line: str) -> List[float]:
    """Parse a histogram array from a line like:
      "    Inputs:  [0.001,0.002,...]"
    Returns list of floats (empty if not parsable).
    """
    m = re.search(r"\[(.*)\]", line)
    if not m:
        return []
    inside = m.group(1).strip()
    if not inside:
        return []
    parts = inside.split(",")
    values: List[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # handle scientific notation like 7.51e-05
        try:
            values.append(float(p))
        except ValueError:
            # if something odd, fallback to 0.0
            values.append(0.0)
    return values


def read_layer(file_path: Path) -> Tuple[Dict[str, int], Dict[str, List[float]], List[str]]:
    """Read a VT YAML layer. Return (instance_map, histograms_map, original_lines).
    histograms_map keys: 'Inputs', 'Weights', 'Outputs'
    """
    lines = read_text(file_path)
    instance: Dict[str, int] = {}
    hists: Dict[str, List[float]] = {}

    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("instance:"):
            instance = parse_instance_inline(ln)
        elif stripped.startswith("Inputs:"):
            hists["Inputs"] = parse_hist_array(ln)
        elif stripped.startswith("Weights:"):
            hists["Weights"] = parse_hist_array(ln)
        elif stripped.startswith("Outputs:"):
            hists["Outputs"] = parse_hist_array(ln)

    return instance, hists, lines


def average_histograms(hist_list: List[List[float]]) -> List[float]:
    if not hist_list:
        return []
    length = len(hist_list[0])
    if any(len(h) != length for h in hist_list):
        # fallback: truncate to min length
        min_len = min(len(h) for h in hist_list)
        hist_list = [h[:min_len] for h in hist_list]
        length = min_len
    summed = [0.0] * length
    for h in hist_list:
        for i, v in enumerate(h):
            summed[i] += v
    n = float(len(hist_list))
    return [v / n for v in summed]


def format_hist(values: List[float]) -> str:
    # six significant digits formatting similar to other generators
    return "[" + ",".join(f"{v:.6g}" for v in values) + "]"


def make_fused_qkv_layer(instance_q: Dict[str, int],
                         h_q: Dict[str, List[float]],
                         h_k: Dict[str, List[float]],
                         h_v: Dict[str, List[float]]) -> List[str]:
    """Create YAML lines for the fused QKV linear layer."""
    # Determine instance: C and P from Q (assumed equal across Q/K/V). M = 3 * M_q
    c_val = int(instance_q.get("C", 768))
    p_val = int(instance_q.get("P", 197))
    m_q = int(instance_q.get("M", 768))
    m_val = 3 * m_q

    inputs_avg = average_histograms([h_q.get("Inputs", []), h_k.get("Inputs", []), h_v.get("Inputs", [])])
    weights_avg = average_histograms([h_q.get("Weights", []), h_k.get("Weights", []), h_v.get("Weights", [])])
    outputs_avg = average_histograms([h_q.get("Outputs", []), h_k.get("Outputs", []), h_v.get("Outputs", [])])

    out_lines: List[str] = []
    out_lines.append("{{include_text('../problem_base.yaml')}}")
    out_lines.append("problem:")
    out_lines.append("  <<<: *problem_base")
    out_lines.append(f"  instance: {{C: {c_val}, M: {m_val}, P: {p_val}}}")
    out_lines.append("")
    out_lines.append("  name: Linear")
    out_lines.append("  dnn_name: vision_transformer")
    out_lines.append("  notes: Fused QKV (3x Linear 768→768)")
    out_lines.append("  # These histograms symmetric and zero-centered (the centermost bin is the")
    out_lines.append("  # probability of zero). Histograms are normalized to sum to 1.0 and they have")
    out_lines.append("  # 2^N-1 bins for some integer N. Higher N yields higher-fidelity histograms,")
    out_lines.append("  # but also increases runtime & the size of YAML files. Encoding functions will")
    out_lines.append("  # upsample or downsample histograms depending on the bitwidth of the")
    out_lines.append("  # corresponding operands.")
    out_lines.append("  histograms:")
    out_lines.append(f"    Inputs:  {format_hist(inputs_avg)}")
    out_lines.append(f"    Weights: {format_hist(weights_avg)}")
    out_lines.append(f"    Outputs: {format_hist(outputs_avg)}")
    return out_lines


def is_numeric_yaml(p: Path) -> bool:
    return p.suffix == ".yaml" and p.stem.isdigit()


def main() -> None:
    here = Path(__file__).resolve().parent
    dnn_root = (here / ".." / "assets" / "DNN_models").resolve()
    src_dir = dnn_root / "vision_transformer"
    dst_dir = dnn_root / "vision_transformer_qkv_fusion"

    if not src_dir.exists():
        raise SystemExit(f"Source directory not found: {src_dir}")

    # Collect numeric layer files
    src_layers = sorted([p for p in src_dir.iterdir() if is_numeric_yaml(p)], key=lambda p: int(p.stem))
    if not src_layers:
        raise SystemExit("No numeric YAML layers found in source directory.")

    # Prepare destination
    if dst_dir.exists():
        # clear existing numeric yaml files but keep anything else
        for p in dst_dir.iterdir():
            if is_numeric_yaml(p):
                p.unlink()
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy over any helper files we want to mirror (optional: skip). We keep directory clean with only layers.

    # Determine formatting width for indices: after fusion we expect fewer files (< 100), so width=2
    # We'll compute dynamically at the end, but we need to know count; instead, collect target layers first.
    # Strategy: build a list of (filename_lines) then write with correct zero padding.
    fused_layers_texts: List[List[str]] = []

    # Copy layer 00 as-is
    first_layer = src_dir / "00.yaml"
    if not first_layer.exists():
        raise SystemExit("Expected first layer '00.yaml' not found.")
    fused_layers_texts.append(read_text(first_layer))

    # Identify final classification head (last numeric file)
    last_numeric = int(src_layers[-1].stem)

    # For ViT-B/16 we have 12 encoder blocks from 01..96 (inclusive), each of size 8
    # Fuse files [b, b+1, b+2] for b in {1,9,17,...,89}
    block_starts = list(range(1, last_numeric))
    # constrain to observed 8-file pattern
    block_starts = [b for b in block_starts if (b - 1) % 8 == 0 and b + 7 <= last_numeric - 1]

    # Map index->path for quick access
    idx_to_path = {int(p.stem): p for p in src_layers}

    for b in block_starts:
        # Q, K, V
        q_path = idx_to_path[b]
        k_path = idx_to_path[b + 1]
        v_path = idx_to_path[b + 2]

        q_inst, q_hist, _ = read_layer(q_path)
        _, k_hist, _ = read_layer(k_path)
        _, v_hist, _ = read_layer(v_path)

        fused_lines = make_fused_qkv_layer(q_inst, q_hist, k_hist, v_hist)
        fused_layers_texts.append(fused_lines)

        # Copy the remaining 5 files of the block as-is: b+3 .. b+7
        for i in range(b + 3, b + 8):
            fused_layers_texts.append(read_text(idx_to_path[i]))

    # Append the final classification head (last file)
    if last_numeric in idx_to_path:
        fused_layers_texts.append(read_text(idx_to_path[last_numeric]))

    # Now write files with updated sequential indices starting at 0
    total = len(fused_layers_texts)
    width = len(str(total - 1)) if total > 0 else 1
    for i, lines in enumerate(fused_layers_texts):
        fname = f"{i:0{width}d}.yaml"
        write_text(dst_dir / fname, lines)

    print(f"Created fused model directory: {dst_dir}")
    print(f"Original layers: {len(src_layers)} | Fused layers: {total}")


if __name__ == "__main__":
    main()


