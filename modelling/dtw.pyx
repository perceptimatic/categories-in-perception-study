# Copyright 2022 CoML team, based on code by Thomas Schatz and Gabriel Synnaeve

import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool
ctypedef np.float32_t CTYPE_t # cost type
ctypedef np.intp_t IND_t # array index type
CTYPE = np.float32 # cost type

cpdef _dtw(IND_t N, IND_t M, CTYPE_t[:,:] dist_array, bool normalized):
    cdef IND_t i, j
    cdef CTYPE_t[:,:] cost = np.empty((N, M), dtype=CTYPE)
    cdef CTYPE_t final_cost, c_diag, c_left, c_up
    # initialization
    cost[0,0] = dist_array[0,0]
    for i in range(1,N):
        cost[i,0] = dist_array[i,0] + cost[i-1,0]
    for j in range(1,M):
        cost[0,j] = dist_array[0,j] + cost[0,j-1]
    # the dynamic programming loop
    for i in range(1,N):
        for j in range(1,M):
            cost[i,j] = dist_array[i,j] + min(cost[i-1,j], cost[i-1,j-1], cost[i,j-1])

    final_cost = cost[N-1, M-1]
    if normalized:
        path_len = 1
        i = N-1
        j = M-1
        while i > 0 and j > 0:
            c_up = cost[i - 1, j]
            c_left = cost[i, j-1]
            c_diag = cost[i-1, j-1]
            if c_diag <= c_left and c_diag <= c_up:
                i -= 1
                j -= 1
            elif c_left <= c_up:
                j -= 1
            else:
                i -= 1
            path_len += 1
        if i == 0:
            path_len += j
        if j == 0:
            path_len += i
        final_cost /= path_len
    return final_cost