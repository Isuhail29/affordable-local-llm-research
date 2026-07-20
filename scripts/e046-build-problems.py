"""E046 problem builder: every ground truth is computed by brute force here, never by hand.

E042 shipped two wrong answers written by hand; E045 was invalidated by a parser that
guessed. This closes the first hole: the answer key is machine-derived by construction.
Problems are chosen to have SHORT correct solutions but high slip rates (the band where
self-consistency can actually operate), avoiding enumeration-heavy items that would blow
the token budget and reintroduce truncation.
"""
import json
from itertools import product
from math import gcd

P = []


def add(q, ans):
    P.append({"id": len(P) + 1, "q": q, "answer": str(ans)})


# 1. three-way CRT
x = next(n for n in range(1, 10000) if n % 5 == 2 and n % 7 == 3 and n % 9 == 4)
add("Find the smallest positive integer x such that x leaves remainder 2 when divided by 5, "
    "remainder 3 when divided by 7, and remainder 4 when divided by 9.", x)

# 2. Legendre: smallest n with 3^10 | n!
def v3(n):
    t, p = 0, 3
    while p <= n:
        t += n // p
        p *= 3
    return t
add("What is the smallest positive integer n such that n! (n factorial) is divisible by 3^10?",
    next(n for n in range(1, 1000) if v3(n) >= 10))

# 3. smallest integer with exactly 12 divisors
def ndiv(n):
    return sum(1 for d in range(1, n + 1) if n % d == 0)
add("What is the smallest positive integer that has exactly 12 divisors?",
    next(n for n in range(1, 100000) if ndiv(n) == 12))

# 4. largest prime factor of 2^20 - 1
def largest_pf(n):
    f, d = n, 2
    best = 1
    while d * d <= f:
        while f % d == 0:
            best = d
            f //= d
        d += 1
    return max(best, f)
add("What is the largest prime factor of 2^20 - 1?", largest_pf(2**20 - 1))

# 5. ordered pairs of positive integers with 3x + 5y = 100
add("How many ordered pairs (x, y) of positive integers satisfy 3x + 5y = 100?",
    sum(1 for xx in range(1, 100) for yy in range(1, 100) if 3 * xx + 5 * yy == 100))

# 6. 100th positive integer that is not a perfect square
sq = {i * i for i in range(1, 200)}
nons = [n for n in range(1, 500) if n not in sq]
add("What is the 100th smallest positive integer that is NOT a perfect square?", nons[99])

# 7. three dice, sum divisible by 4
add("Three fair six-sided dice are rolled. Of the 216 equally likely outcomes, in how many "
    "is the sum divisible by 4?",
    sum(1 for a, b, c in product(range(1, 7), repeat=3) if (a + b + c) % 4 == 0))

# 8. 3-digit numbers, distinct digits, divisible by 11
add("How many 3-digit positive integers are divisible by 11 and have three distinct digits?",
    sum(1 for n in range(100, 1000) if n % 11 == 0 and len(set(str(n))) == 3))

# 9. sum of 3-digit palindromes divisible by 7
add("What is the sum of all 3-digit palindromes that are divisible by 7?",
    sum(n for n in range(100, 1000) if str(n) == str(n)[::-1] and n % 7 == 0))

# 10. integers 1..1000 divisible by 3 or 7 but not both
add("How many integers from 1 to 1000 inclusive are divisible by 3 or by 7, but not by both?",
    sum(1 for n in range(1, 1001) if (n % 3 == 0) != (n % 7 == 0)))

# 11. recurrence
a = 2
for _ in range(7):
    a = 3 * a - 1
add("A sequence is defined by a(1) = 2 and a(n+1) = 3*a(n) - 1. What is a(8)?", a)

# 12. coprime count below 60 (Euler phi)
add("How many positive integers less than 60 are coprime to 60 (share no common factor with 60 other than 1)?",
    sum(1 for n in range(1, 60) if gcd(n, 60) == 1))

# 13. two-digit numbers equal to 7x sum of digits? count
add("How many two-digit positive integers are exactly 7 times the sum of their digits?",
    sum(1 for n in range(10, 100) if n == 7 * sum(int(d) for d in str(n))))

# 14. trailing zeros of 100!
def v5(n):
    t, p = 0, 5
    while p <= n:
        t += n // p
        p *= 5
    return t
add("How many trailing zeros does 100! (100 factorial) have?", v5(100))

# 15. paths on grid avoiding a point
from math import comb
total = comb(10, 5)
through = comb(4, 2) * comb(6, 3)
add("A path on a lattice goes from the point (0,0) to the point (5,5), moving only one unit "
    "right or one unit up at each step. How many such paths do NOT pass through the point (2,2)?",
    total - through)

# 16. digit sum divisible by 5, 1..300
add("How many integers from 1 to 300 inclusive have a digit sum that is divisible by 5?",
    sum(1 for n in range(1, 301) if sum(int(d) for d in str(n)) % 5 == 0))

# 17. smallest n where n^2 ends in 89
add("What is the smallest positive integer n such that n^2 ends with the digits 89?",
    next(n for n in range(1, 100000) if str(n * n).endswith("89")))

# 18. count of 4-digit numbers with all distinct digits and divisible by 25
add("How many 4-digit positive integers are divisible by 25 and have four distinct digits?",
    sum(1 for n in range(1000, 10000) if n % 25 == 0 and len(set(str(n))) == 4))

if __name__ == "__main__":
    with open("datasets/e046-reasoning.jsonl", "w", encoding="utf-8") as f:
        for p in P:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(P)} problems with brute-forced answers")
    for p in P:
        print(f"  q{p['id']:>2}: {p['answer']}")
