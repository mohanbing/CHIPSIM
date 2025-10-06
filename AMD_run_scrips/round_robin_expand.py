#!/usr/bin/env python3
"""
Round-robin expand flits so there's one flit per line.

Usage:
  python3 round_robin_expand.py input.txt [output.txt]

Input format (detected automatically):
  clk i_ni i_router o_ni o_router vnet flits
  (header line optional)

Behavior:
  - Groups lines by clk.
  - For each group, repeatedly emit one-flit lines in round-robin order
    among the src-dst entries until all flits for that clk are exhausted.
  - The emitted flit lines keep all columns the same except the flits column
    is set to 1.
"""
import sys
from collections import deque

def parse_line(line):
    toks = line.strip().split()
    if not toks:
        return None
    # detect header (non-integer first token)
    try:
        clk = int(toks[0])
    except ValueError:
        return ("header", line.rstrip("\n"))
    # Expect at least 7 tokens: clk, 4 src/dst, vnet(ignore), flits
    if len(toks) < 7:
        raise ValueError(f"Unexpected line format: {line!r}")
    flits = int(toks[6])
    return (clk, toks, flits)

def process_group(group, out_f):
    """group: list of (toks, flits). Write expanded lines to out_f."""
    if not group:
        return
    dq = deque()
    remaining = []
    for i, (toks, fl) in enumerate(group):
        if fl > 0:
            dq.append(i)
            remaining.append(fl)
        else:
            remaining.append(0)
    # round-robin
    while dq:
        i = dq.popleft()
        toks = list(group[i][0])  # copy tokens
        toks[6] = "1"             # set flits to 1
        out_f.write(" ".join(toks) + "\n")
        remaining[i] -= 1
        if remaining[i] > 0:
            dq.append(i)

def main():
    if len(sys.argv) < 2:
        print("Usage: round_robin_expand.py input.txt [output.txt]", file=sys.stderr)
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) >= 3 else infile + ".expanded"
    with open(infile, "r") as inf, open(outfile, "w") as outf:
        current_clk = None
        group = []  # list of (toks, flits)
        for raw in inf:
            p = parse_line(raw)
            if p is None:
                continue
            if p[0] == "header":
                outf.write(p[1] + "\n")
                continue
            clk, toks, flits = p
            if current_clk is None:
                current_clk = clk
            if clk != current_clk:
                # flush previous group
                process_group(group, outf)
                group = []
                current_clk = clk
            group.append((toks, flits))
        # flush last group
        process_group(group, outf)
    print(f"Wrote expanded file: {outfile}", file=sys.stderr)

if __name__ == "__main__":
    main()