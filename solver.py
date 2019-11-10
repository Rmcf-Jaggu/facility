from collections import namedtuple
from matplotlib import pyplot as plt
import cvxopt as op
import numpy as np
import random

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'init_price', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def solve_it(input_data):

    lines = input_data.split('\n')
    parts = lines[0].split()
    global fact_cnt, cust_cnt
    fact_cnt = int(parts[0])
    cust_cnt = int(parts[1])

    fact = []
    locate_fact = np.zeros((fact_cnt, 2))
    for i in range(1, fact_cnt+1):
        parts = lines[i].split()
        fact.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))
        locate_fact[i-1] = [float(parts[2]), float(parts[3])]

    cust = []
    locate_cust = np.zeros((cust_cnt, 2))
    for i in range(fact_cnt+1, fact_cnt+1+cust_cnt):
        parts = lines[i].split()
        cust.append(Customer(i-1-fact_cnt, int(parts[0]), Point(float(parts[1]), float(parts[2]))))
        locate_cust[i-fact_cnt-1] = [float(parts[1]), float(parts[2])]

    dist = solve_it_dist_matrx(locate_fact, locate_cust)

    if cust_cnt <= 100:
        A_urban_link, b_urban_link, A_temp, b_temp, c = solve_it_tab_cons(fact, cust, dist)
        global obj, solution
        solution = [-1] * cust_cnt
        obj = float('inf')
        solve_it_bnb(fact, cust, dist, A_urban_link, b_urban_link, A_temp, b_temp, c)

    else:
        not_that_imp_temp = np.argsort([facility.init_price for facility in fact])[::-1][:fact_cnt//10]
        init_obj_func, init_soln = solve_it_with_greed(fact, cust, dist, not_that_imp_temp)

        obj, solution = solve_it_taboo(fact, cust, dist, init_obj_func, init_soln)
        for i in range(4):
            if i % 2 == 0:
                obj, solution = solve_it_taboo(fact, cust, dist, obj, solution)
            else:
                populate_fact = list(range(fact_cnt))
                not_that_imp_temp = random.sample(populate_fact, fact_cnt//3)
                obj_random, soln_random = solve_it_with_greed(fact, cust, dist, not_that_imp_temp)
                obj_current, soln_current = solve_it_taboo(fact, cust, dist, obj_random, soln_random)
                if obj_current < obj:
                    obj, solution = obj_current, soln_current

    visualize(fact, cust, solution)

    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    return output_data


def solve_it_dist_matrx(locate_fact, locate_cust):

    dist = np.zeros((fact_cnt, cust_cnt))

    for i in range(cust_cnt):
        cur_customer = np.array([locate_cust[i]])
        dist[:, i] = np.sqrt(np.square(cur_customer[:, 0] - locate_fact[:, 0]) + \
                                  np.square(cur_customer[:, 1] - locate_fact[:, 1]))

    return dist

def solve_it_bnb(fact, cust, dist, A_urban_link, b_urban_link, A_temp, b_temp, c):

    global obj, solution
    soln_LP_relax = op.solvers.lp(c, A_urban_link, b_urban_link, A_temp, b_temp)

    if soln_LP_relax['primal objective'] > obj:
        return

    x = np.array(soln_LP_relax['x']).reshape(-1)
    var_fact_rand = np.argsort(np.abs(x - 0.5))[0]

    if x[var_fact_rand] < 0.0001 or abs(x[var_fact_rand] - 1) < 0.0001:
        soln_current = [-1] * cust_cnt
        for i in range(fact_cnt):
            for j in range(cust_cnt):
                if abs(x[fact_cnt+cust_cnt*i+j] - 1) < 0.0001:
                    soln_current[j] = i
        obj_current = obj_calc(fact, cust, dist, soln_current)
        if obj_current < obj:
            obj, solution = obj_current, soln_current
        return

    new_A_temp = np.zeros((1, fact_cnt+fact_cnt*cust_cnt))
    new_A_temp[0, var_fact_rand] = 1
    new_A_temp = np.concatenate((np.array(A_temp), new_A_temp), axis=0)
    new_A_temp = op.matrix(new_A_temp)
    
    new_b_temp = np.concatenate((np.array(b_temp).T[0], [0]))
    new_b_temp = op.matrix(new_b_temp)
    solve_it_bnb(fact, cust, dist, A_urban_link, b_urban_link, new_A_temp, new_b_temp, c)
    
    new_b_temp = np.concatenate((np.array(b_temp).T[0], [1]))
    new_b_temp = op.matrix(new_b_temp)
    solve_it_bnb(fact, cust, dist, A_urban_link, b_urban_link, new_A_temp, new_b_temp, c)


def obj_calc(fact, cust, dist, solution):

    used = [0] * fact_cnt
    for facility_index in solution:
        used[facility_index] = 1

    obj = sum([facility.init_price * used[facility.index] for facility in fact])
    for c in range(cust_cnt):
        f = solution[c]
        obj += dist[f, c]

    return obj

def solve_it_tab_cons(fact, cust, dist):

    A_urban_link = np.zeros((fact_cnt*cust_cnt+fact_cnt, fact_cnt+fact_cnt*cust_cnt))
    b_urban_link = np.zeros(fact_cnt*cust_cnt+fact_cnt)
    A_temp = np.zeros((cust_cnt, fact_cnt+fact_cnt*cust_cnt))
    b_temp = np.zeros(cust_cnt)
    c = np.zeros(fact_cnt+fact_cnt*cust_cnt)

    c[:fact_cnt] = [facility.init_price for facility in fact]
    c[fact_cnt:] = dist.reshape(-1)
    c = op.matrix(c)

    demand = [customer.demand for customer in cust]
    for i in range(fact_cnt):
        A_urban_link[cust_cnt*i:cust_cnt*(i+1), i] = [-1] * cust_cnt
        A_urban_link[fact_cnt*cust_cnt+i, fact_cnt+cust_cnt*i:fact_cnt+cust_cnt*(i+1)] = demand
    A_urban_link[:fact_cnt*cust_cnt, fact_cnt:] = np.eye(fact_cnt*cust_cnt)
    A_urban_link = np.concatenate((A_urban_link, np.eye(fact_cnt+fact_cnt*cust_cnt), -np.eye(fact_cnt+fact_cnt*cust_cnt)), axis=0)
    A_urban_link = op.matrix(A_urban_link)
    
    capacity = [facility.capacity for facility in fact]
    b_urban_link[fact_cnt*cust_cnt:] = capacity
    b_urban_link = np.concatenate((b_urban_link, np.ones(fact_cnt+fact_cnt*cust_cnt), np.zeros(fact_cnt+fact_cnt*cust_cnt)))
    b_urban_link = op.matrix(b_urban_link)

    for i in range(cust_cnt):
        temp = np.zeros((fact_cnt, cust_cnt))
        temp[:, i] += 1
        temp = temp.reshape(-1)
        A_temp[i, fact_cnt:] = temp
    A_temp = op.matrix(A_temp)

    b_temp += 1
    b_temp = op.matrix(b_temp)

    return A_urban_link, b_urban_link, A_temp, b_temp, c


def solve_it_with_greed(fact, cust, dist, not_that_imp_temp):

    solution = [-1] * cust_cnt
    rem_capacity = [facility.capacity for facility in fact]

    for c in range(cust_cnt):
        sorted_fact = np.argsort(dist[:, c])

        i = 0
        f = sorted_fact[i]

        while f in not_that_imp_temp or cust[c].demand > rem_capacity[f]:
            i += 1
            
            if i == fact_cnt:
                return float('inf'), solution
            f = sorted_fact[i]


        rem_capacity[f] -= cust[c].demand
        solution[c] = f

    obj = obj_calc(fact, cust, dist, solution)
    return obj, solution


def solve_it_taboo(fact, cust, dist, obj, solution):

    M = 100
    N = 80 
    L = 60 
    min_obj, min_solution = obj, solution
    obj_current, soln_current = obj, solution

    sol_list_in_tab = [set(solution)] * L
    obj_temp_pool = [obj_current]
    find = False
    for i in range(M):

        near_neigh = soln_just_got_neigh(soln_current, fact, cust, dist, N)
        n = 0
        while set(near_neigh[n][1]) in sol_list_in_tab:
            if min_obj - near_neigh[n][0] > 1:
                break
            n += 1

            if n == L:
                break

        if n == L:
            sol_list_in_tab = soln_updt_tab(sol_list_in_tab, [])
        else:
            obj_current, soln_current = near_neigh[n][0], near_neigh[n][1]
            sol_list_in_tab = soln_updt_tab(sol_list_in_tab, soln_current)
            
            if obj_current < min_obj:
                min_obj, min_solution = obj_current, soln_current
                find = True
        obj_temp_pool.append(obj_current)

    if find:
        plt.ion()
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.plot(range(M+1), obj_temp_pool)
        plt.pause(5)
        plt.close()

    return min_obj, min_solution



def soln_just_got_neigh(solution, fact, cust, dist, N):

    used = set(solution)
    unused = set(range(fact_cnt)) - used
    plus_no = min(int(N * 0.1), len(unused))
    subs_no = min(int(N * 0.4), len(used))
    switch_pt_1 = int(N * 0.3) 
    switch_pt_2 = N - plus_no - subs_no - switch_pt_1 

    near_neigh = []

    add_list = random.sample(unused, plus_no)
    for add in add_list:
        not_that_imp_temp = unused - set([add])
        obj_current, soln_current = solve_it_with_greed(fact, cust, dist, not_that_imp_temp)
        near_neigh.append([obj_current, soln_current])

    delete_list = random.sample(used, subs_no)
    for delete in delete_list:
        not_that_imp_temp = unused | set([delete])
        obj_current, soln_current = solve_it_with_greed(fact, cust, dist, not_that_imp_temp)
        near_neigh.append([obj_current, soln_current])

    i = 0
    switch_hash = []
    while i < switch_pt_1:
        switch = (random.choice(list(used)), random.choice(list(unused)))
        if switch in switch_hash:
            continue
        else:
            switch_hash.append(switch)
            
        demand = 0
        for c in range(cust_cnt):
            if solution[c] == switch[0]:
                demand += cust[c].demand

        if demand > fact[switch[1]].capacity:
            continue
        soln_current = [switch[1] if f == switch[0] else f for f in solution]
        obj_current = obj_calc(fact, cust, dist, soln_current)
        near_neigh.append([obj_current, soln_current])
        i += 1

    i = 0
    switch_hash = []
    while i < switch_pt_2:
        switch = random.sample(used, 2)
        if switch in switch_hash:
            continue
        else:
            switch_hash.append(switch)

        demand_0, demand_1 = 0, 0
        for c in range(cust_cnt):
            f = solution[c]
            if f == switch[0]:
                demand_1 += cust[c].demand
            if f == switch[1]:
                demand_0 += cust[c].demand

        if demand_1 > fact[switch[1]].capacity or demand_0 > fact[switch[0]].capacity:
            continue
        soln_current = [-1] * cust_cnt
        for c in range(cust_cnt):
            f = solution[c]
            if f == switch[0]:
                soln_current[c] = switch[1]
            elif f == switch[1]:
                soln_current[c] = switch[0]
            else:
                soln_current[c] = f
        obj_current = obj_calc(fact, cust, dist, soln_current)
        near_neigh.append([obj_current, soln_current])
        i += 1

    near_neigh.sort(key=lambda neighbor: neighbor[0])
    return near_neigh


def soln_updt_tab(sol_list_in_tab, soln_current):
    return sol_list_in_tab[1:] +[set(soln_current)]


def visualize(fact, cust, solution):

    plt.ion()
    plt.figure(figsize=(12, 8))
    color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'y']


    for i in range(cust_cnt):

        customer = cust[i]
        facility = fact[solution[i]]

        x = [customer.location[0], facility.location[0]]
        y = [customer.location[1], facility.location[1]]
        plt.plot(x, y, c=color_list[solution[i]%7], ls="-", lw=0.2, marker='.', ms=2)

    for i in range(fact_cnt):

        facility = fact[i]
        x, y = facility.location[0], facility.location[1]
        plt.scatter(x, y, c=color_list[i%7], marker='p', s=20)

    plt.pause(15)
    plt.close()


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')