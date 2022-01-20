

def _extend_inst(inst1, n1, inst2, n2):

    inst1.extend(inst2)
    for i in range(n1, n1 + n2):

        if inst1[i][1] is not None:
            inst1[i][1] += n1

        if inst1[i][2] is not None:
            inst1[i][2] += n1

    return inst1


def _compute(inst, n, x):

    opp = inst[n][0]
    val = inst[n][3]

    if opp is None:
        return val(x)
    elif inst[n][1] is None:
        a = val
        b = _compute(inst, inst[n][2], x)
    elif inst[n][2] is None:
        a = _compute(inst, inst[n][1], x)
        b = val
    else:
        a = _compute(inst, inst[n][1], x)
        b = _compute(inst, inst[n][2], x)

    return opp(a, b)
