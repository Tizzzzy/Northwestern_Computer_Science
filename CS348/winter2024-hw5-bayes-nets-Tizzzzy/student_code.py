def enumerate_all(vars, evidence):
    if not vars:
        return 1.0
    first, rest = vars[0], vars[1:]
    if first.name in evidence:
        print(first.name)
        return first.probability(evidence[first.name], evidence) * enumerate_all(rest, evidence)
    else:
        sum = 0
        for val in [True, False]:
            new_evidence = evidence.copy()
            new_evidence[first.name] = val
            sum += first.probability(val, new_evidence) * enumerate_all(rest, new_evidence)
        return sum

def normalize(answer, total):
    return answer / total

def ask(var, value, evidence, bn):
    vars = bn.variables[:]
    # print(var.name for var in vars)
    extended_evidence = evidence.copy()
    # print(extended_evidence)
    
    extended_evidence[var] = value
    true_prob = enumerate_all(vars, extended_evidence)
    
    evidence_without_var = evidence.copy()
    if var in evidence_without_var:
        del evidence_without_var[var]
    normalization_constant = enumerate_all(vars, evidence_without_var.copy())
    
    return normalize(true_prob, normalization_constant)
