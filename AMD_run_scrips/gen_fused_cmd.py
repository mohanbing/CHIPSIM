#!/usr/bin/env python3
import csv, argparse, math, sys, os, re

def parse_num_with_suffix(s: str) -> int:
    """
    Parse numbers like: 1000, 1_000, 12.5k, 3M, 1.2G, scientific 1e9.
    Returns integer (rounded toward +inf for non-integers).
    """
    s = s.strip().replace("_", "")
    if not s:
        raise ValueError("empty number")
    # Scientific notation path first
    try:
        val = float(s)
        return int(math.ceil(val))
    except ValueError:
        pass
    m = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*([kKmMgG])', s)
    if m:
        base = float(m.group(1))
        suf  = m.group(2).lower()
        mult = 1
        if suf == 'k': mult = 1_000
        elif suf == 'm': mult = 1_000_000
        elif suf == 'g': mult = 1_000_000_000
        return int(math.ceil(base * mult))
    # plain integer fallback
    return int(s)

def pick_col(row, names):
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    raise KeyError(f"Missing required column among {names}")

def read_layers(csv_path: str):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        # Peek first non-empty, non-comment line to decide header
        first_line = None
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            first_line = line
            break
        if first_line is None:
            return []
        # Determine header by checking if it contains alpha
        has_alpha = any(c.isalpha() for c in first_line)
        f.seek(0)
        if has_alpha:
            reader = csv.DictReader(
                (ln for ln in f if not ln.strip().startswith("#") and ln.strip())
            )
            for r in reader:
                rows.append(r)
        else:
            # No header: assume order = read_mb, macs, write_mb
            reader = csv.reader(
                (ln for ln in f if not ln.strip().startswith("#") and ln.strip())
            )
            for r in reader:
                if len(r) < 3: 
                    raise ValueError("CSV row must have at least 3 columns: read_mb, macs, write_mb")
                rows.append({"read_mb": r[0], "macs": r[1], "write_mb": r[2]})
    return rows

def build_command(csv_path: str, exe: str, tR: int, tC: int, tW: int,
                  pinR: str, pinC: str, pinW: str, rr: int, rw: int,
                  nt_writes: bool):
    rows = read_layers(csv_path)
    if not rows:
        raise SystemExit("CSV is empty after filtering comments/blank lines.")

    read_mib_list = []
    write_mib_list = []
    iters_list = []

    # Column name aliases
    read_aliases  = ["read_mb", "read_mib", "read", "Sr", "read_MB"]
    macs_aliases  = ["macs", "MACs", "total_macs", "ops"]
    write_aliases = ["write_mb", "write_mib", "write", "Sw", "write_MB"]

    for row in rows:
        # Normalize keys for DictReader (strip spaces)
        row = {k.strip(): v.strip() for k, v in row.items()}

        r_mb_str = pick_col(row, read_aliases)
        w_mb_str = pick_col(row, write_aliases)
        macs_str = pick_col(row, macs_aliases)

        r_mb = round(parse_num_with_suffix(r_mb_str)/1000000)
        w_mb = round(parse_num_with_suffix(w_mb_str)/1000000)
        macs = parse_num_with_suffix(macs_str)

        if r_mb < 0 or w_mb < 0 or macs < 0:
            raise ValueError("Negative values are not allowed in read/write/ MACs")

        # MACs -> iterations (256 MACs / iter)
        iters = int(math.ceil(macs / 256.0))

        read_mib_list.append(str(r_mb))
        write_mib_list.append(str(w_mb))
        iters_list.append(str(iters))

    L = len(read_mib_list)

    # Repeat rr and rw L times as comma-separated lists
    rr_list = ",".join([str(rr)] * L)
    rw_list = ",".join([str(rw)] * L)

    parts = [exe, "-L", str(L),
             "-Sr", ",".join(read_mib_list),
             "-rr", rr_list,
             "-ci", ",".join(iters_list),
             "-Sw", ",".join(write_mib_list),
             "-rw", rw_list]

    # Threads per kernel (using user's preferred -tR/-tC/-tW flags)
    if tR is not None: parts += ["-tR", str(tR)]
    if tC is not None: parts += ["-tC", str(tC)]
    if tW is not None: parts += ["-tW", str(tW)]

    # Optional pinning lists (pass-through)
    if pinR: parts += ["--pinR", pinR]
    if pinC: parts += ["--pinC", pinC]
    if pinW: parts += ["--pinW", pinW]

    if nt_writes:
        parts.append("--nt-writes")

    return " ".join(parts)

def main():
    p = argparse.ArgumentParser(description="Generate fused_bench_rcw_layers_omp command from CSV")
    p.add_argument("--csv", required=True, help="CSV file with columns: read_mb, macs, write_mb (header optional)")
    p.add_argument("--exe", default="./fused_bench_rcw_layers_omp", help="Path to benchmark executable")
    p.add_argument("-tR", type=int, default=4, help="Threads for READ kernel")
    p.add_argument("-tC", type=int, default=8, help="Threads for COMPUTE kernel")
    p.add_argument("-tW", type=int, default=2, help="Threads for WRITE kernel")
    p.add_argument("--pinR", default='0,1,2,3', help="Comma-separated core list for READ phase (pass-through)")
    p.add_argument("--pinC", default='0,1,2,3,4,5,6,7', help="Comma-separated core list for COMPUTE phase (pass-through)")
    p.add_argument("--pinW", default='0,1', help="Comma-separated core list for WRITE phase (pass-through)")
    p.add_argument("--rr", type=int, default=1, help="Read repetitions per layer (repeated across L)")
    p.add_argument("--rw", type=int, default=1, help="Write repetitions per layer (repeated across L)")
    p.add_argument("--nt-writes", default=True, help="Add --nt-writes flag")
    args = p.parse_args()

    try:
        cmd = build_command(args.csv, args.exe, args.tR, args.tC, args.tW,
                            args.pinR, args.pinC, args.pinW,
                            args.rr, args.rw, args.nt_writes)
        print(cmd)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
