qp.k_max = 100;
qp.mu0 = 0.0000000000000000e+00;
qp.mu_tol = 1.0000000000000000e-10;
qp.N = 2;
qp.nx = 2, 2, 2, ;
qp.nu = 1, 1, ;
qp.nb = 3, 3, 2, ;
qp.ng = 0, 0, 0, ;
qp.warm_start = 0;
qp.A{1} = 1.0000000000000000e+00, 1.0000000000000000e+00, ;
0.0000000000000000e+00, 1.0000000000000000e+00, ;
qp.B{1} = 5.0000000000000000e-01, ;
1.0000000000000000e+00, ;
qp.b{1} = 0.0000000000000000e+00, 0.0000000000000000e+00, ;
qp.Q{1} = 6.6000000000000000e+01, 7.8000000000000000e+01, ;
7.8000000000000000e+01, 9.3000000000000000e+01, ;
qp.S{1} = 9.0000000000000000e+01, 1.0800000000000000e+02, ;
qp.R{1} = 1.2600000000000000e+02, ;
qp.q{1} = 0.0000000000000000e+00, 0.0000000000000000e+00, ;
qp.r{1} = 0.0000000000000000e+00, ;
qp.lb{1} = -1.0000000000000000e+00, 1.0000000000000000e+00, 0.0000000000000000e+00, ;
qp.ub{1} = 1.0000000000000000e+00, 1.0000000000000000e+00, 0.0000000000000000e+00, ;
qp.C{1} = qp.D{1} = qp.lg{1} = ;
qp.ug{1} = ;
qp.x{1} = -nan, -nan, ;
qp.u{1} = -nan, ;
qp.A{2} = 1.0000000000000000e+00, 1.0000000000000000e+00, ;
0.0000000000000000e+00, 1.0000000000000000e+00, ;
qp.B{2} = 5.0000000000000000e-01, ;
1.0000000000000000e+00, ;
qp.b{2} = 0.0000000000000000e+00, 0.0000000000000000e+00, ;
qp.Q{2} = 6.6000000000000000e+01, 7.8000000000000000e+01, ;
7.8000000000000000e+01, 9.3000000000000000e+01, ;
qp.S{2} = 9.0000000000000000e+01, 1.0800000000000000e+02, ;
qp.R{2} = 1.2600000000000000e+02, ;
qp.q{2} = 0.0000000000000000e+00, 0.0000000000000000e+00, ;
qp.r{2} = 0.0000000000000000e+00, ;
qp.lb{2} = -1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, ;
qp.ub{2} = 1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000000000000e+00, ;
qp.C{2} = qp.D{2} = qp.lg{2} = ;
qp.ug{2} = ;
qp.x{2} = -nan, -nan, ;
qp.u{2} = -nan, ;
qp.Q{3} = 1.0000000000000000e+01, 1.4000000000000000e+01, ;
1.4000000000000000e+01, 2.0000000000000000e+01, ;
qp.q{3} = 0.0000000000000000e+00, 0.0000000000000000e+00, ;
qp.lb{3} = -1.0000000000000000e+00, -1.0000000000000000e+00, ;
qp.ub{3} = nan, nan, ;
qp.C{3} = qp.lg{3} = ;
qp.ug{3} = ;
qp.x{3} = -nan, -nan, ;