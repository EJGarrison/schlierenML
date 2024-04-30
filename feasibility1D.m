%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MLP Applied to 1D Normal Shock   %
%                                   %
%     Authored By Evan Garrison     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This code requires the use of the aerospace and deep learning toolboxes.

clc; clearvars; close all;

%% Get normal shock data
Minf = 1.1:0.001:5;             % Freestream mach number
gamma = 1.4;                    % Ratio of specific heats
rhoinf = 1.225;                 % [kg/m^3] Density
Tinf = 300;                     % [K] Temperature
R = 287;                        % [J/kg-K] Specific gas constant (air)
pinf = rhoinf*R*Tinf;           % [Pa] Pressure
ainf = sqrt(gamma*R*Tinf);      % [m/s] Speed of sound
uinf = ainf*Minf;

% "inf" means before shock, "2" means after shock, "ratio" means 2/inf.
[Minf,T2ratio,pratio,rhoratio,M2,p0ratio,RPratio] = flownormalshock(gamma,Minf);

T2 = T2ratio*Tinf;
p2 = pratio*pinf;
rho2 = rhoratio*rhoinf;
a2 = sqrt(gamma*R*T2);
u2 = M2.*a2;

%% Train MLP
data = struct("rho",rho2,"p",p2,"u",u2);

mlpnet = feedforwardnet(20);
[mlpnet,tr] = train(mlpnet,data.rho, data.p);

%% Plot Abs and Error Results

figure(1)
scatter(M2, abs(mlpnet(data.rho)-data.p))
title("Absolute MLP Error Behind Normal Shock", Interpreter="latex",FontSize=20)
xlabel("$M_2$", Interpreter="latex", FontSize=15)
ylabel("$|p_{predicted} - p_{true}|$ [Pa]", Interpreter="latex",FontSize=15)

figure(2)
scatter(M2, abs(mlpnet(data.rho)-data.p)./data.p)
title("Relative MLP Error Behind Normal Shock", Interpreter="latex",FontSize=20)
xlabel("$M_2$", Interpreter="latex", FontSize=15)
ylabel("Relative Error (% of Actual Value)", Interpreter="latex",FontSize=15)