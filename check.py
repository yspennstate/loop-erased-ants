"""Exact checks for the six-vertex ant-process counterexample.

Graph: vertices 0..5, nest 0, food 5, edges 01,12,25,03,34,45,13.
A. the path probabilities and their sum D, the drift, the fourteen coefficients of D*LV,
   the error constant 434717/9000000 and the trapping bound 3943/7143 at S = 400008.
B. the law of the reinforced path at three weight vectors, three ways: spanning trees,
   the forward (chronological) erasure by Lawler's product formula, and the backward rule
   (erasure of the time-reversed walk) by an exact absorbing chain on
   (position, first-entrance parents); plus a pathwise test that the erasure of the
   reversed walk follows first-entrance parents.
"""
import itertools
import random
import sympy as sp

R = sp.Rational
u, v, z, t, d, r = sp.symbols("u v z t delta r")

# ---------------- A ----------------
A, H, C0, J = u, v / 2, 1 - u, (1 - v) / 2
Z = A * H * (C0 + J) + C0 * J * (A + H) + z * (A + C0) * (H + J)
pU = A * H * (C0 + J + z) / Z
pL = C0 * J * (A + H + z) / Z
pC = A * z * J / Z
pB = C0 * z * H / Z
assert sp.simplify(pU + pL + pC + pB - 1) == 0
D = 2 * u * (1 - u) + v * (1 - v) + 2 * z
assert sp.simplify(Z - D / 4) == 0
fu = pU + pC - u          # edge 01 lies on U and C
fv = pU + pB - v          # edge 12 lies on U and B
fz = pC + pB - z          # edge 13 lies on C and B
assert sp.simplify(fu + 2 * u * (1 - u) * (u - v) / D) == 0
assert sp.simplify(fv - v * (1 - v) * (u - v) / D) == 0
assert sp.simplify(fz - z * (2 * u**2 - 4 * u * v + v**2 + v - 2 * z) / D) == 0
# central point
sub = {u: R(1, 2), v: R(1, 2), z: R(1, 8)}
print("central path probabilities", [sp.nsimplify(p.subs(sub)) for p in (pU, pL, pC, pB)])

q = (u + 2 * v) / 3
delta = u - v
rr = z - q * (1 - q) / 2
V = delta**2 + rr**2
F = (fu, fv, fz)
LV = sum(sp.diff(V, s) * f for s, f in zip((u, v, z), F))
back = {u: R(1, 2) + t + 2 * d / 3, v: R(1, 2) + t - d / 3}
back[z] = r + (R(1, 2) + t) * (R(1, 2) - t) / 2
DLV = sp.expand(sp.cancel((D * LV).subs(back)))
poly = sp.Poly(DLV, t, d, r)
claimed = {
    (0, 2, 0): R(-3, 2), (0, 0, 2): R(-1, 2), (0, 4, 0): 2, (1, 3, 1): R(4, 9),
    (1, 3, 0): 4, (0, 2, 2): R(34, 9), (2, 2, 1): R(7, 9), (0, 2, 1): R(17, 36),
    (2, 2, 0): 6, (1, 1, 2): R(4, 3), (3, 1, 1): R(-2, 3), (1, 1, 1): R(1, 6),
    (0, 0, 3): -4, (2, 0, 2): 2,
}
got = {m: c for m, c in zip(poly.monoms(), poly.coeffs())}
print("D*LV monomials:", len(got), "match (5):", got == claimed)
if got != claimed:
    print("  got", got)
eta = R(1, 100)
err = sum(abs(c) * eta ** (a + b + e - 2) for (a, b, e), c in got.items()
          if (a, b, e) not in ((0, 2, 0), (0, 0, 2)))
assert all(b + e >= 2 for (a, b, e) in got)
print("error constant", err, "claimed 434717/9000000:", err == R(434717, 9000000))
S = 400008
p11 = 1 - R(16) / (S * eta**2) - R(192) / (S * eta)
print("(10) at S=400008:", p11, "== 3943/7143:", p11 == R(3943, 7143))
# Lq coefficient
Lq = sum(sp.diff(q, s) * f for s, f in zip((u, v, z), F))
assert sp.simplify(Lq - 2 * delta**2 * (u + v - 1) / (3 * D)) == 0

# ---------------- B ----------------
E = [(0, 1), (1, 2), (2, 5), (0, 3), (3, 4), (4, 5), (1, 3)]
paths = [(0, 1, 2, 5), (0, 3, 4, 5), (0, 1, 3, 4, 5), (0, 3, 1, 2, 5)]


def tree_law(w):
    out = {p: 0 for p in paths}
    tot = 0
    for T in itertools.combinations(range(7), 5):
        par = list(range(6))

        def f(x):
            while par[x] != x:
                x = par[x]
            return x
        ok = True
        for i in T:
            a, b = E[i]
            ra, rb = f(a), f(b)
            if ra == rb:
                ok = False
                break
            par[ra] = rb
        if not ok:
            continue
        wt = 1
        for i in T:
            wt *= w[i]
        adj = {x: [] for x in range(6)}
        for i in T:
            a, b = E[i]
            adj[a].append(b)
            adj[b].append(a)
        prev = {0: None}
        st = [0]
        while st:
            x = st.pop()
            for y in adj[x]:
                if y not in prev:
                    prev[y] = x
                    st.append(y)
        pth = [5]
        while pth[-1] != 0:
            pth.append(prev[pth[-1]])
        out[tuple(reversed(pth))] += wt
        tot += wt
    return {p: out[p] / tot for p in paths}


def chain(w):
    c = {}
    for i, (a, b) in enumerate(E):
        c[(a, b)] = c[(b, a)] = w[i]
    deg = {x: sum(val for (a, b), val in c.items() if a == x) for x in range(6)}
    return lambda x, y: c.get((x, y), 0) / deg[x]


def green(p, dom, x):
    dom = sorted(dom)
    M = sp.Matrix(len(dom), len(dom), lambda i, j: (1 if i == j else 0) - p(dom[i], dom[j]))
    G = M.inv()
    k = dom.index(x)
    return G[k, k]


def lawler(p, eta_path, stop):
    # P(chronological erasure = eta) for the walk from eta[0] stopped on hitting `stop`
    pr = 1
    for i in range(len(eta_path) - 1):
        dom = set(range(6)) - stop - set(eta_path[:i])
        pr *= green(p, dom, eta_path[i]) * p(eta_path[i], eta_path[i + 1])
    return pr


def loop_erase(path):
    # chronological loop erasure: sigma_0 = path[0]; sigma_{i+1} follows the LAST visit to sigma_i
    out = [path[0]]
    while out[-1] != path[-1]:
        j = max(k for k, x in enumerate(path) if x == out[-1])
        out.append(path[j + 1])
    return tuple(out)


def backward_path_from_parents(par):
    # path read from F back to N through first-entrance parents, then written from N to F
    pth = [5]
    while pth[-1] != 0:
        pth.append(par[pth[-1]])
    return tuple(reversed(pth))


def backward_law(w):
    # exact law of the backward rule: absorbing chain on (position, first-entrance parents)
    p = chain(w)
    adj = {x: [y for y in range(6) if p(x, y) != 0] for x in range(6)}
    s0 = (0, ((0, None),))
    idx, states, todo = {s0: 0}, [s0], [s0]
    while todo:
        x, par = todo.pop()
        d = dict(par)
        for y in adj[x]:
            if y == 5 or y in d:
                continue
            nd = dict(d)
            nd[y] = x
            st = (y, tuple(sorted(nd.items(), key=lambda t: t[0])))
            if st not in idx:
                idx[st] = len(states)
                states.append(st)
                todo.append(st)
        for y in adj[x]:
            if y in d and y != 5:
                st = (y, par)
                if st not in idx:
                    idx[st] = len(states)
                    states.append(st)
                    todo.append(st)
    n = len(states)
    A = [[R(0)] * n for _ in range(n)]
    rhs = [{pp: R(0) for pp in paths} for _ in range(n)]
    for i, (x, par) in enumerate(states):
        d = dict(par)
        A[i][i] += 1
        for y in adj[x]:
            q = R(p(x, y))
            if y == 5:
                nd = dict(d)
                nd[5] = x
                rhs[i][backward_path_from_parents(nd)] += q
            elif y in d:
                A[i][idx[(y, par)]] -= q
            else:
                nd = dict(d)
                nd[y] = x
                A[i][idx[(y, tuple(sorted(nd.items(), key=lambda t: t[0])))]] -= q
    # Gauss-Jordan in exact arithmetic, all path columns at once
    cols = [[rhs[i][pp] for pp in paths] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        cols[c], cols[piv] = cols[piv], cols[c]
        inv = 1 / A[c][c]
        A[c] = [a * inv for a in A[c]]
        cols[c] = [a * inv for a in cols[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
                cols[r] = [a - f * b for a, b in zip(cols[r], cols[c])]
    return {pp: sp.nsimplify(cols[0][k]) for k, pp in enumerate(paths)}, n


# pathwise identity: the erasure of the reversed walk follows first-entrance parents
rng = random.Random(20261119)
for _ in range(20000):
    walk, par, x = [0], {0: None}, 0
    while x != 5:
        nbrs = [b if a == x else a for (a, b) in E if x in (a, b)]
        y = rng.choice(nbrs)
        if y not in par:
            par[y] = x
        walk.append(y)
        x = y
    lit = tuple(reversed(loop_erase(list(reversed(walk)))))
    assert lit == backward_path_from_parents(par), (walk, lit)
print("pathwise: erasure of the reversed walk = first-entrance path on 20000 random walks")

ok = True
for w in ([R(3), R(1), R(2), R(5), R(7), R(4), R(2)],
          [R(1, 3), R(9), R(9), R(2), R(1, 2), R(1, 2), R(11, 5)],
          [R(1)] * 7):
    p = chain(w)
    tl = tree_law(w)
    fw = {pp: lawler(p, pp, {5}) for pp in paths}
    bw, nstates = backward_law(w)
    print("w=", [str(x) for x in w])
    print("  trees      ", [str(tl[pp]) for pp in paths])
    print("  forward    ", [str(sp.nsimplify(fw[pp])) for pp in paths])
    print("  backward   ", [str(bw[pp]) for pp in paths], "(", nstates, "chain states )")
    a = all(sp.simplify(tl[pp] - fw[pp]) == 0 for pp in paths)
    b = all(sp.simplify(tl[pp] - bw[pp]) == 0 for pp in paths)
    print("  trees==forward:", a, " trees==backward:", b)
    ok = ok and a and b
print("PART B", "PASS" if ok else "FAIL")
