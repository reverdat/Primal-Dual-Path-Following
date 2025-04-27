#### ARBITRARY LINEAR PROBLEM ###

set VARS ordered;
set CONSTR ordered;

param c{VARS};
param A{CONSTR, VARS} default 0;
param b{CONSTR};

var x{VARS} >= 0;

minimize objective: sum{j in VARS} c[j]*x[j];

subject to constraints{i in CONSTR}:
    sum{j in VARS} A[i,j]*x[j] = b[i];