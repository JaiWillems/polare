

def _extend_inst(inst1, n1, inst2, n2):

    for i in range(n2):

        opp, a, b, val = inst2[i][0], inst2[i][1], inst2[i][2], inst2[i][3]

        if a is not None:
            a += n1
        
        if b is not None:
            b += n1
        
        inst1.append([opp, a, b, val])

    return inst1


def _compute(inst, n, x):

    opp, val = inst[n][0], inst[n][3]

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

    temp = opp(a) if b is None else opp(a, b)

    return temp
