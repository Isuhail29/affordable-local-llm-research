"""E046 v2: genuinely hard problems, answers still brute-forced.

v1 saturated at 100% single-sample accuracy. These target the known slip modes of
mid-size models: strict-inequality case analysis, derangements, non-attacking
placements, and searches with subtle conditions. All answers computed here.
"""
import json
from itertools import combinations, permutations, product

P = []


def add(q, ans):
    P.append({"id": len(P) + 1, "q": q, "answer": str(ans)})


# 1. ordered triples with strict inequality
add("How many triples of positive integers (a, b, c) satisfy a + b + c = 15 and a < b < c?",
    sum(1 for a in range(1, 16) for b in range(1, 16) for c in range(1, 16)
        if a + b + c == 15 and a < b < c))

# 2. derangements of 6
def derange(n):
    return sum(1 for p in permutations(range(n)) if all(p[i] != i for i in range(n)))
add("In how many ways can the six letters A, B, C, D, E, F be arranged so that no letter "
    "stays in its original position?", derange(6))

# 3. non-attacking rooks
def rooks(n, k):
    c = 0
    for rows in combinations(range(n), k):
        for cols in permutations(range(n), k):
            c += 1
    return c
add("In how many ways can 3 rooks be placed on a 4x4 chessboard so that no two rooks "
    "share a row or a column?", rooks(4, 3))

# 4. 5-digit palindromes divisible by 11
add("How many 5-digit palindromes are divisible by 11?",
    sum(1 for n in range(10000, 100000) if str(n) == str(n)[::-1] and n % 11 == 0))

# 5. three consecutive integers each divisible by a square > 1
def has_sq_factor(n):
    d = 2
    while d * d <= n:
        if n % (d * d) == 0:
            return True
        d += 1
    return False
add("What is the smallest integer n greater than 1 such that each of n, n+1, and n+2 is "
    "divisible by a perfect square greater than 1?",
    next(n for n in range(2, 100000)
         if has_sq_factor(n) and has_sq_factor(n + 1) and has_sq_factor(n + 2)))

# 6. subsets with no two consecutive
def no_consec(n):
    c = 0
    for r in range(n + 1):
        for s in combinations(range(1, n + 1), r):
            if all(s[i + 1] - s[i] > 1 for i in range(len(s) - 1)):
                c += 1
    return c
add("How many subsets of the set {1, 2, 3, ..., 10} contain no two consecutive integers? "
    "(Include the empty set.)", no_consec(10))

# 7. divisible by none of 2, 3, 5
add("How many integers from 1 to 1000 inclusive are divisible by none of 2, 3, and 5?",
    sum(1 for n in range(1, 1001) if n % 2 and n % 3 and n % 5))

# 8. strictly decreasing digits
add("How many 4-digit positive integers have digits that are strictly decreasing from left to right?",
    sum(1 for n in range(1000, 10000)
        if all(int(str(n)[i]) > int(str(n)[i + 1]) for i in range(3))))

# 9. divisors of a square
def ndiv(n):
    c, d = 0, 1
    while d * d <= n:
        if n % d == 0:
            c += 2 if d * d != n else 1
        d += 1
    return c
add("How many positive divisors does 2024^2 have?", ndiv(2024 ** 2))

# 10. taxicab
def taxicab():
    seen = {}
    for a in range(1, 40):
        for b in range(a, 40):
            s = a ** 3 + b ** 3
            seen.setdefault(s, []).append((a, b))
    return min(k for k, v in seen.items() if len(v) >= 2)
add("What is the smallest positive integer that can be written as the sum of two positive "
    "cubes in two different ways?", taxicab())

# 11. Euler polynomial failures
def is_prime(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True
add("For how many integers n from 1 to 100 inclusive is n^2 + n + 41 NOT a prime number?",
    sum(1 for n in range(1, 101) if not is_prime(n * n + n + 41)))

# 12. trailing zeros in base 12
def vp(n, p):
    t, q = 0, p
    while q <= n:
        t += n // q
        q *= p
    return t
add("How many trailing zeros does 1000! have when written in base 12?",
    min(vp(1000, 2) // 2, vp(1000, 3)))

if __name__ == "__main__":
    with open("datasets/e046-hard.jsonl", "w", encoding="utf-8") as f:
        for p in P:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(P)} hard problems with brute-forced answers")
    for p in P:
        print(f"  q{p['id']:>2}: {p['answer']}")
