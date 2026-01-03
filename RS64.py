from typing import List, Tuple

ori_prim = 0x43
pn = 2 ** 6


def gf_add(x: int, y: int) -> int: 
    return x ^ y

def gf_sub(x: int, y: int) -> int: 
    return x ^ y

def cl_mul(x: int, y: int) -> int:
    z = 0
    i = 0
    while (y >> i) > 0:
        if y & (1 << i):
            z ^= x << i
        i += 1
    return z

def cl_div(dividend: int, divisor: int) -> int:
    d1 = dividend.bit_length()
    d2 = divisor.bit_length()  
    if d1 < d2:
        return dividend  
    for i in range(d1 - d2, -1, -1):
        if dividend & (1 << (i + d2 - 1)):
            dividend ^= divisor << i
    return dividend

def gf_mul(x: int, y: int, prim: int = 0) -> int: 
    z = cl_mul(x, y)
    if prim > 0:
        z = cl_div(z, prim)
    return z


def init_table(prim: int = 0x13) -> Tuple[List[int], List[int]]:
    table_exp = [0] * 2 * pn
    table_log = [0] * pn
    x = 1 
    for i in range(pn - 1):
        table_exp[i] = x
        table_log[x] = i
        x = gf_mul(x, 2, prim)
    for i in range((pn - 1), 2 * pn):
        table_exp[i] = table_exp[i - (pn - 1)]
    return table_exp, table_log

table_exp, table_log = init_table(ori_prim)
def gf_mul_lut(x: int, y: int) -> int: 
    if x == 0 or y == 0:
        return 0
    return table_exp[table_log[x] + table_log[y]]


def gf_div(x: int, y: int) -> int:
    if y == 0:
        raise ZeroDivisionError()
    if x == 0:
        return 0
    return table_exp[(table_log[x] + (pn - 1) - table_log[y]) % (pn - 1)] 


def gf_pow(x: int, power: int) -> int:
    return table_exp[(table_log[x] * power) % (pn - 1)]


def gf_inv(x: int) -> int:
    return table_exp[(pn - 1) - table_log[x]]  



def gf_poly_scale(p: List[int], x: int) -> List[int]:  
    r = [0] * len(p)
    for i in range(len(p)):
        r[i] = gf_mul_lut(p[i], x)
    return r

def gf_poly_add(p: List[int], q: List[int]) -> List[int]: 
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i + len(r) - len(p)] = p[i]
    for i in range(len(q)):
        r[i + len(r) - len(q)] ^= q[i]
    return r


def gf_poly_mul(p: List[int], q: List[int]) -> List[int]: 
    r = [0] * (len(p) + len(q) - 1)
    for j in range(len(q)):
        for i in range(len(p)):
            r[i+j] ^= gf_mul_lut(p[i], q[j])
    return r

def gf_poly_eval(p: List[int], x: int) -> int:  
    y = p[0]
    for i in range(1, len(p)):
        y = gf_mul_lut(y, x) ^ p[i]
    return y



def rs_generator_poly(nsym: int) -> List[int]: 
    g = [1]
    for i in range(nsym):
        g = gf_poly_mul(g, [1, gf_pow(2, i)])
    return g

def gf_poly_div(dividend: List[int], divisor: List[int]) -> Tuple[List[int], List[int]]: 
    msg_out = list(dividend)
    for i in range(len(dividend) - len(divisor) + 1):
        coef = msg_out[i]
        if coef != 0:
            for j in range(1, len(divisor)):
                if divisor[j] != 0:
                    msg_out[i + j] ^= gf_mul_lut(divisor[j], coef)
    sep = -(len(divisor)-1)
    return msg_out[:sep], msg_out[sep:]


def rs_encode_msg(msg_in: List[int], nsym: int) -> List[int]:  
    gen = rs_generator_poly(nsym)
    _, remainder = gf_poly_div(msg_in + [0] * nsym, gen)
    msg_out = msg_in + remainder
    return msg_out

def rs_calc_syndromes(msg: List[int], nsym: int) -> List[int]:
    synd = [0] * nsym
    for i in range(nsym):
        synd[i] = gf_poly_eval(msg, gf_pow(2, i))
    return [0] + synd



def rs_find_errata_locator(e_pos: List[int]) -> List[int]:
    e_loc = [1]
    for i in e_pos:
        e_loc = gf_poly_mul(e_loc, gf_poly_add([1], [gf_pow(2, i), 0]))
    return e_loc


def rs_find_error_evaluator(synd: List[int], err_loc: List[int], nsym: int) -> List[int]:
    _, remainder = gf_poly_div(gf_poly_mul(synd, err_loc), [1] + [0] * (nsym + 1))
    return remainder
    

def rs_correct_errata(msg_in: List[int], synd: List[int], err_pos: List[int]) -> List[int]:
    coef_pos = [len(msg_in) - 1 - p for p in err_pos]
    err_loc = rs_find_errata_locator(coef_pos)
    err_eval = rs_find_error_evaluator(synd[::-1], err_loc, len(err_loc) - 1)[::-1]
    x = []
    for i in range(len(coef_pos)):
        l = (pn - 1) - coef_pos[i]
        x.append(gf_pow(2, -l))
    e = [0] * len(msg_in)
    xlength = len(x)
    for i, xi in enumerate(x):
        xi_inv = gf_inv(xi)
        err_loc_prime_tmp = []
        for j in range(xlength):
            if j != i:
                err_loc_prime_tmp.append(gf_sub(1, gf_mul_lut(xi_inv, x[j])))
        err_loc_prime = 1
        for coef in err_loc_prime_tmp:
            err_loc_prime = gf_mul_lut(err_loc_prime, coef)
        y = gf_poly_eval(err_eval[::-1], xi_inv)
        y = gf_mul_lut(gf_pow(xi, 1), y)
        if err_loc_prime == 0:
            return
        magnitude = gf_div(y, err_loc_prime)
        e[err_pos[i]] = magnitude
    msg_in = gf_poly_add(msg_in, e)
    return msg_in



def rs_find_error_locator(synd: List[int], nsym: int) -> List[int]:
    err_loc = [1]
    old_loc = [1]
    synd_shift = 0
    if len(synd) > nsym:
        synd_shift = len(synd) - nsym
    for i in range(nsym):
        k = i + synd_shift
        delta = synd[k]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul_lut(err_loc[-(j+1)], synd[k-j])
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = gf_poly_scale(old_loc, delta)
                old_loc = gf_poly_scale(err_loc, gf_inv(delta))
                err_loc = new_loc
            err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta))
    while len(err_loc) and err_loc[0] == 0:
        del err_loc[0]
    errs = len(err_loc) - 1
    if errs * 2 > nsym:
        return
    return err_loc

def rs_find_errors(err_loc: List[int], nmsg: int) -> List[int]:
    errs = len(err_loc) - 1
    err_pos = []
    for i in range(nmsg):
        if gf_poly_eval(err_loc, gf_pow(2, i)) == 0:
            err_pos.append(nmsg - 1 - i)
    if len(err_pos) != errs:
        return
    return err_pos

def rs_forney_syndrome(synd: List[int], nmsg: int, pos: List[int]) -> List[int]:
    erase_pos_rev = [nmsg - 1 - p for p in pos]
    fsynd = list(synd[1:])
    for i in range(len(pos)):
        x = gf_pow(2, erase_pos_rev[i])
        for j in range(len(fsynd) - 1):
            fsynd[j] = gf_mul(fsynd[j], x) ^ fsynd[j + 1]
    return fsynd


def rs_correct_msg(msg_in: List[int], nsym: int) -> List[int]:
    if len(msg_in) > (pn - 1):
        return
    msg_out = list(msg_in)
    synd = rs_calc_syndromes(msg_out, nsym)
    if max(synd) == 0:
        return msg_out
    fsynd = synd[1:]
    err_loc = rs_find_error_locator(fsynd, nsym)
    if err_loc == None:
        return
    err_pos = rs_find_errors(err_loc[::-1], len(msg_out))
    if err_pos is None:
        return
    msg_out = rs_correct_errata(msg_out, synd, err_pos)
    return msg_out









































