#!/usr/bin/env python3
"""
Collatz Affine-Cover Searcher
==============================

For each positive integer n, the Collatz conjecture is equivalent to the
following purely integer statement:

    There exist  r >= 0, l >= 0, and a NON-INCREASING integer sequence
    l >= s[0] >= s[1] >= ... >= s[r-1] >= 0  such that

        2^l - 3^r * n  =  sum_{i=0}^{r-1}  3^i * 2^{s[i]}          (*)

This comes from the affine-group embedding of the Collatz semigroup into
Aff(Z[1/6]): every Collatz word w of type (r,l) acts as the affine map
f_w(x) = (3^r / 2^l) * x + b_w, and f_w(n) = 1 iff b_w = 1 - 3^r*n/2^l.
The intercept b_w is a sum of r terms  3^{r_after_j} / 2^{l_before_j}  over
the R-positions in w; multiplying through by 2^l gives condition (*).
The s-sequence records the number of L's before each R in the word (reversed).

This program searches for valid (r, l, s) triples, finding solutions that
are typically far shorter than the actual Collatz word (e.g. r=9 instead
of r=43 for n=97).

Note: finding a solution does NOT require following the Collatz trajectory;
it only requires the affine map f_w(n) = 1 to hold over Z[1/6], which can
be verified exactly in O(r) integer arithmetic.

Algorithm
---------
For each n, try r = 1, 2, 3, ... in turn.  For each r:

  1.  Compute  Q = 3^r * n  and  P_min = (3^r - 1) / 2  (the minimum
      possible value of the RHS, achieved when all s[i] = 0).

  2.  The smallest viable l satisfies  2^l >= Q + P_min.

  3.  For l = l_min, l_min+1, ... up to a window:
      Set T = 2^l - Q.  If T < P_min, skip.
      Call decompose(T, r, l) to find a non-increasing s.

  4.  decompose(T, r, max_s) recurses: choose s[0] (the largest value,
      coefficient 3^0 = 1) then recurse on  (T - 2^{s[0]}) / 3  with r-1
      terms bounded by s[0].  Parity pruning (s[0] must satisfy
      2^{s[0]} ≡ T mod 3) and a lower-bound cut (remaining T must be at
      least P_min[r-1]) keep the search fast.

Usage
-----
    python3 collatz_cover_search.py [--start N] [--end N]
                                    [--single N]
                                    [--max-r R]
                                    [--l-window W]
                                    [--cutoff C]
                                    [--verify]
                                    [--quiet]

    --start / --end    Range of n to search (default 1..200)
    --single N         Search for a single n; overrides --start/--end
    --max-r R          Maximum r to try per n (default 150)
    --l-window W       Search l in [l_min, l_min + W*r] (default 3)
    --cutoff C         Max backtrack calls per (r,l) attempt (default 10000)
    --verify           After finding a solution, verify it algebraically
    --quiet            Print only failures / final summary

Examples
--------
    python3 collatz_cover_search.py --start 1 --end 100
    python3 collatz_cover_search.py --single 837799 --verify
    python3 collatz_cover_search.py --start 1 --end 1000 --quiet
"""

import argparse
import sys
import time


# ---------------------------------------------------------------------------
# Precomputed constants
# ---------------------------------------------------------------------------

_MAX_PRECOMPUTE = 300

# P_MIN[r] = (3^r - 1) // 2  = sum_{i=0}^{r-1} 3^i  (minimum RHS when all s=0)
P_MIN = [(3**r - 1) // 2 for r in range(_MAX_PRECOMPUTE)]


# ---------------------------------------------------------------------------
# Core decomposition routine
# ---------------------------------------------------------------------------

def decompose(T, r, max_s, counter, cutoff):
    """
    Find a non-increasing sequence  max_s >= s[0] >= ... >= s[r-1] >= 0
    such that  T = sum_{i=0}^{r-1} 3^i * 2^{s[i]}.

    Parameters
    ----------
    T       : int  -- target value
    r       : int  -- number of terms
    max_s   : int  -- upper bound on s[0]
    counter : list -- single-element list used as a mutable call counter
    cutoff  : int  -- abort and return False when counter[0] exceeds this

    Returns (True, s_list)  or  (False, []).
    """
    counter[0] += 1
    if counter[0] > cutoff:
        return False, []

    if r == 0:
        return (T == 0), []

    if T < P_MIN[r]:
        return False, []   # cannot fill r terms even at minimum

    t_mod3 = T % 3
    if t_mod3 == 0:
        return False, []   # no power of 2 is divisible by 3

    # s[0] must satisfy  2^{s[0]} ≡ T  (mod 3).
    # Since 2^k mod 3 = 1 when k is even, 2 when k is odd:
    parity = 0 if t_mod3 == 1 else 1

    # Upper bound: leave enough room for the remaining r-1 terms.
    room = T - P_MIN[r - 1]
    if room <= 0:
        return False, []

    s0_max = min(max_s, room.bit_length())
    if s0_max % 2 != parity:
        s0_max -= 1   # align to correct parity

    for s0 in range(s0_max, -1, -2):
        contrib = 1 << s0
        if contrib > room:
            continue
        rem = T - contrib
        if rem % 3 != 0:
            continue   # safety (should always hold given parity choice)
        ok, sub = decompose(rem // 3, r - 1, s0, counter, cutoff)
        if ok:
            return True, [s0] + sub

    return False, []


# ---------------------------------------------------------------------------
# Collatz reference solution (for comparison)
# ---------------------------------------------------------------------------

def collatz_solution(n):
    """
    Return (r, l, s) extracted from the actual Collatz trajectory of n.
    s[i] = l_before_{r-1-i}  where l_before_j is the count of L-steps
    strictly before the j-th R-step.
    """
    word, x = [], n
    while x != 1:
        if x % 2 == 0:
            word.append('L')
            x //= 2
        else:
            word.append('R')
            x = 3 * x + 1
    r = word.count('R')
    l = word.count('L')
    lb, lc = [], 0
    for c in word:
        if c == 'R':
            lb.append(lc)
        else:
            lc += 1
    return r, l, list(reversed(lb))


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def find_solution(n, max_r=150, l_window=3, cutoff=10000):
    """
    Find the smallest r for which a valid (r, l, s) triple exists.

    Returns (r, l, s)  or  (None, None, None) if nothing found within limits.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")

    # n=1 and n=2 feed directly into the trivial word
    if n == 1:
        return 0, 0, []   # empty word, 2^0 - 3^0*1 = 0
    if n == 2:
        return 0, 1, []   # 2^1 - 1*2 = 0

    for r in range(1, max_r + 1):
        if r >= _MAX_PRECOMPUTE:
            # Extend if needed
            while len(P_MIN) <= r:
                P_MIN.append((3 ** len(P_MIN) - 1) // 2)

        pow3r = 3 ** r
        Q = pow3r * n
        P_min_r = P_MIN[r]
        total_min = Q + P_min_r

        # Smallest l with 2^l >= total_min
        bl = total_min.bit_length()
        l_min = bl - 1 if (1 << (bl - 1)) >= total_min else bl
        while (1 << l_min) < total_min:
            l_min += 1

        window = max(l_window * r, 6)

        for l in range(l_min, l_min + window):
            T = (1 << l) - Q
            if T < P_min_r:
                continue

            ctr = [0]
            ok, s = decompose(T, r, l, ctr, cutoff)
            if ok:
                return r, l, s

    return None, None, None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_solution(n, r, l, s):
    """
    Verify that (r, l, s) is a valid solution for n.
    Checks:
      1. s is non-increasing and 0 <= s[i] <= l.
      2. sum_{i} 3^i * 2^{s[i]}  ==  2^l - 3^r * n.
      3. f_w(n) = 1  (affine map check).
    Returns (True, '') or (False, error_message).
    """
    if len(s) != r:
        return False, f"len(s)={len(s)} != r={r}"

    for i in range(len(s) - 1):
        if s[i] < s[i + 1]:
            return False, f"s not non-increasing at i={i}: s[{i}]={s[i]} < s[{i+1}]={s[i+1]}"

    if s and (s[0] > l or s[-1] < 0):
        return False, f"s out of range [0, {l}]: s[0]={s[0]}, s[-1]={s[-1]}"

    rhs = sum(3 ** i * (1 << s[i]) for i in range(r))
    lhs = (1 << l) - (3 ** r) * n
    if rhs != lhs:
        return False, f"RHS={rhs} != LHS={lhs} (diff={rhs - lhs})"

    # Affine map check: a*n + b == 1 over rationals
    from fractions import Fraction
    a = Fraction(3 ** r, 1 << l)
    b = Fraction(rhs, 1 << l)
    result = a * n + b
    if result != 1:
        return False, f"f_w(n) = {result} != 1"

    return True, ''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--start',   type=int, default=1,
                   help='First n to search (default 1)')
    p.add_argument('--end',     type=int, default=200,
                   help='Last n to search inclusive (default 200)')
    p.add_argument('--single',  type=int, default=None,
                   help='Search for a single n (overrides --start/--end)')
    p.add_argument('--max-r',   type=int, default=150,
                   help='Max r per n (default 150)')
    p.add_argument('--l-window',type=int, default=3,
                   help='l search window multiplier (default 3)')
    p.add_argument('--cutoff',  type=int, default=10000,
                   help='Max backtrack calls per (r,l) attempt (default 10000)')
    p.add_argument('--verify',  action='store_true',
                   help='Verify every solution found')
    p.add_argument('--quiet',   action='store_true',
                   help='Only print failures and final summary')
    return p.parse_args()


def format_s(s, max_show=6):
    if len(s) <= max_show:
        return str(s)
    return str(s[:max_show])[:-1] + f', ... ({len(s)} terms)]'


def main():
    args = parse_args()

    if args.single is not None:
        targets = [args.single]
    else:
        targets = range(args.start, args.end + 1)

    found    = 0
    missing  = []
    t_start  = time.time()

    print(f"Collatz affine-cover search")
    print(f"  n range : {targets[0]}..{targets[-1]}" if len(targets) > 1
          else f"  n       : {args.single}")
    print(f"  max_r   : {args.max_r}")
    print(f"  l_window: {args.l_window}")
    print(f"  cutoff  : {args.cutoff}")
    print(f"  verify  : {args.verify}")
    print()
    print(f"{'n':>10}  {'r':>5}  {'l':>5}  {'s (first 6 terms)':30}  {'Collatz r':>10}  {'notes'}")
    print("-" * 90)

    for n in targets:
        t0 = time.time()

        r, l, s = find_solution(
            n,
            max_r    = args.max_r,
            l_window = args.l_window,
            cutoff   = args.cutoff,
        )
        dt = time.time() - t0

        r_c, l_c, _ = collatz_solution(n)

        if r is None:
            missing.append(n)
            note = f"NOT FOUND  [{dt:.3f}s]"
            print(f"{n:>10}  {'?':>5}  {'?':>5}  {'':30}  {r_c:>10}  {note}")
            continue

        found += 1
        notes = []
        if r < r_c:
            notes.append(f"shorter by {r_c - r} R-steps")
        if dt > 1.0:
            notes.append(f"{dt:.2f}s")

        if args.verify:
            ok, msg = verify_solution(n, r, l, s)
            if not ok:
                notes.append(f"VERIFY FAILED: {msg}")
            else:
                notes.append("verified ✓")

        note_str = "  ".join(notes)

        if not args.quiet or r is None or notes:
            print(f"{n:>10}  {r:>5}  {l:>5}  {format_s(s):<30}  {r_c:>10}  {note_str}")

    elapsed = time.time() - t_start

    print()
    print("=" * 90)
    print(f"Summary: {found}/{len(list(targets))} found in {elapsed:.2f}s")
    if missing:
        print(f"Not found: {missing}")
    else:
        print("All n in range covered. ✓")


if __name__ == '__main__':
    main()
