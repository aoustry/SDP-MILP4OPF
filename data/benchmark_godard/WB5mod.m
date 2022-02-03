function mpc = case5mod
%% 5 bus case in MATPOWER format
%  W. A. Bukhsh, Feb 2013 
%%   References:
%   [1] W. A. Bukhsh, Andreas Grothey, Ken McKinnon, Paul trodden, "Local Solutions of Optimal Power Flow Problem"
%       submitted to IEEE Transactions on Power Systems, 2013
%   [2] W. A. Bukhsh, Andreas Grothey, Ken McKinnon, Paul trodden, "Local Solutions of Optimal Power Flow Problem"
%       Technical Report ERGO, 2011

%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	3	0	0	0	0	1	0.95	0       345	1	1.1	0.90;
	2	1	130	20	0	0	1	0.95   -16  	345	1	1.1	0.90;
	3	1	130	20	0	0	1	0.95   -20  	345	1	1.1	0.90;
	4	1	65  10	0	0	1	0.95   -135 	345	1	1.1	0.90;
	5	2	0	0	0	0	1	0.95   -140 	345	1	1.1	0.90;	

    ];
%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
	1	500	50	160	    -160	1	100	1	500 	0	0	0	0	0	0	0	0	0	0	0	0;
	5	0	0	160 	-160	1	100	1	250  	0	0	0	0	0	0	0	0	0	0	0	0;
  
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	2	0.04	0.09	   0.0     2500	2500	2500	0	0	1	-360	360;
	1	3	0.05	0.10       0.0     2500	2500	2500	0	0	1	-360	360;
	2	4	0.55   	0.90	   0.45    2500	2500	2500	0	0	1	-360	360;
	3	5	0.55    0.90	   0.45    2500	2500	2500	0	0	1	-360	360;
	4	5	0.06    0.1        0.0     2500	2500	2500	0	0	1	-360	360;
	2	3	0.07 	0.09       0.0     2500	2500	2500	0	0	1	-360	360;
	
];

%%-----  OPF Data  -----%%
%% area data
%	area	refbus
mpc.areas = [
	1	5;
];

%% generator cost data
%	1	startup	shutdown	n	x1	y1	...	xn	yn
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	2000	0	3	0	6.00 	0;
	2	2000	0	3	0	2.00	0;
	
];
