import RS64

  
def code_lib(k, n, client_num): 
    code_lib = [] 
    for i in range(client_num):
        ms = [i // 10, i % 10]
        if k == n:
            rs = ms
        else:
            rs = RS64.rs_encode_msg(ms ,n - k)
        code_lib.append(rs)
    return code_lib







