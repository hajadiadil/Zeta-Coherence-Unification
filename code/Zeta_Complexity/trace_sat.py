import random, csv, math, sys

def rand_3sat(n_vars=200, n_clauses=900, seed=0):
    rng = random.Random(seed)
    F = []
    for _ in range(n_clauses):
        lits = set()
        while len(lits) < 3:
            v = rng.randrange(1, n_vars+1)
            sgn = rng.choice([1,-1])
            lits.add(sgn*v)
        F.append(tuple(lits))
    return F, n_vars

def eval_unsat(F, assign):
    # nombre de clauses non satisfaites
    unsat = 0
    for c in F:
        ok = False
        for lit in c:
            v = abs(lit)
            val = assign[v]
            if (lit > 0 and val) or (lit < 0 and not val):
                ok = True; break
        if not ok: unsat += 1
    return unsat

def walksat_trace(F, n_vars, max_steps=20000, p_random=0.5, seed=0):
    rng = random.Random(seed)
    # assignation initiale
    assign = {v: rng.choice([False, True]) for v in range(1, n_vars+1)}
    trace = []
    for step in range(max_steps):
        U = []  # indices des clauses non satisfaites
        for i,c in enumerate(F):
            ok = False
            for lit in c:
                v = abs(lit)
                val = assign[v]
                if (lit > 0 and val) or (lit < 0 and not val):
                    ok = True; break
            if not ok: U.append(i)

        trace.append(len(U))  # <-- métrique: nb de clauses non-satisfaites

        if not U:  # satisfiable & trouvé
            break

        # WalkSAT: clause non satisfaite aléatoire
        ci = rng.choice(U)
        clause = F[ci]

        if rng.random() < p_random:
            # flip aléatoire
            v = abs(rng.choice(clause))
            assign[v] = not assign[v]
        else:
            # flip qui minimise les non-satisfaites
            best_v, best_score = None, math.inf
            for lit in clause:
                v = abs(lit)
                assign[v] = not assign[v]
                score = eval_unsat(F, assign)
                assign[v] = not assign[v]
                if score < best_score:
                    best_score, best_v = score, v
            assign[best_v] = not assign[best_v]
    return trace

if __name__ == "__main__":
    # instance aléatoire "dure" (ajuste n_clauses/n_vars vers le ratio ~4.2 pour 3-SAT)
    F, n = rand_3sat(n_vars=200, n_clauses=840, seed=1)
    trace = walksat_trace(F, n, max_steps=20000, p_random=0.5, seed=1)

    # export CSV une colonne 'value'
    with open("trace_sat.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["value"])
        for v in trace: w.writerow([v])

    print(f"Trace écrit: trace_sat.csv (longueur={len(trace)})")
