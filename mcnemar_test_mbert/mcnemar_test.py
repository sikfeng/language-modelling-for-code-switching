import math
from scipy.stats.distributions import chi2
from statsmodels.stats.contingency_tables import mcnemar

def fact(n):
    if n == 0:
        return 1
    return n*fact(n-1)

def asymptotic(table):
    b = max(table[0][1], table[1][0])
    c = min(table[0][1], table[1][0])
    n = table[0][1] + table[1][0]

    chi_square = (b-c)**2/(b+c)
    return chi_square, chi2.sf(chi_square, 1)

def mid_p(table):
    b = max(table[0][1], table[1][0])
    c = min(table[0][1], table[1][0])
    n = table[0][1] + table[1][0]

    stat = 0
    for i in range(b, n + 1):
        stat += 2*(fact(n)/(fact(n-i)*fact(i)))*0.5**n

    return stat

# the values in `tables` are obtained from `evaluate_models.py`
tables = [[[11, 6],
           [12, 971]],
          [[24, 12],
           [13, 951]],
          [[34, 15],
           [15, 936]],
          [[69, 33],
           [40, 2858]],
          [[5, 10],
           [11, 974]],
          [[23, 13],
           [17, 947]],
          [[27, 19],
           [12, 942]],
          [[55, 42],
           [40, 2863]],
          [[12, 7],
           [11, 970]],
          [[15, 6],
           [12, 947]],
          [[39, 18],
           [19, 933]],
          [[57, 31],
           [42, 2850]],
          [[6, 7],
           [10, 977]],
          [[22, 7],
           [18, 953]],
          [[23, 24],
           [16, 937]],
          [[51, 38],
           [44, 2867]]]


for table in tables:
    result = mcnemar(table, exact=False, correction=False)
    stat = result.statistic
    p = result.pvalue
    print(f'chi-squared={stat:.4}, p-value={p:.4}')
