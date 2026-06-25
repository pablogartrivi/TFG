%   S_MinimizationCNPRotatingWP.m
% Minimization of the CN Rotating WavePlate polarimeter
% El polarizador inicial se supone a lo largo del eje X
clear
clc
close all

warning off all
format compact
format long

Sini = [1,1,0,0]';

% Desfase lámina
philamina = pi/2;

%Genera esfera de Poincare
fact = 0.96; % radio ligeramente inferior a 1
[X, Y, Z] = sphere;
X = fact*X;
Y = fact*Y;
Z = fact*Z;

figure (2)
surf(X,Y,Z)
hold on
axis equal
xlabel('S1');
ylabel('S2');
zlabel('S3');
colormap('Bone');    
hold on


% Dibujo sobre la esfera del "8" de los estados de polarización generados
% al rotar la lámina

% Estados de polarización generados al rotar la lámina
Nrot = 100; % Número de rotaciones
SoPs = zeros(Nrot,4);

rot = linspace(0,pi,Nrot); % Rotaciones
for k=1:Nrot
    M = Mueller('waveplate', rot(k), philamina);
    SoPs(k,:) = M * Sini;
end

hold on
plot3(SoPs(:,2), SoPs(:,3), SoPs(:,4),'bo', 'MarkerFaceColor','b');
hold off

%% Minimización CN

% Numero de iteraciones
%===================
%===================
nitera = 400;
%===================
%===================

% Número de rotaciones de la lámina
Nrot = 6;

% Rotaciones iniciales
% ============================    
%     % Todos a cero
%     x0 = zeros(Nrot,1);
% 
    % aleatorios
    %=======================================
    x0 = pi*rand(Nrot,1);
    
    % "Equiespaciados"
    %=======================================
%     x0 = linspace(0.1, pi-0.1, Nrot);

    % Valor inicial del CN
    % valor inicial del Condition Number
CNB = RotatingWPPolCN(x0);
XB  = x0;  
    

% Minimización del CN a partir del punto inicial
%===============================================
% opciones para el programa de optimizacion
TolFun = 1.e-5; %1.e-15;
TolX   = 1.e-3; %1e-13;
options = optimset('MaxIter', 300, 'Display', 'iter', 'MaxFunEvals', 30000,'Algorithm','active-set','TolFun',TolFun,'TolX',TolX); 

% Las variables a optimizar en este caso son las rotaciones
% Intervalos donde se deja variar las rotaciones
% En este caso no my grande 
lb = 0; % Lower bounds
ub = pi; % Upper bounds
x = fmincon(@RotatingWPPolCN, x0, [], [], [], [], lb,ub, [], options);
x

[CN, TSoP] = RotatingWPPolSops(x, philamina);

if CN < CNB
    CNB = CN;
    XB   = x;
end

%Esto se dibuja en la figura 2
hold on
plot3(TSoP(:,2), TSoP(:,3), TSoP(:,4),'go', 'MarkerFaceColor','g');
hold off

% opciones para el programa de optimizacion
% Por si se quieren cambiar
options = optimset('MaxIter', 300, 'Display', 'off', 'MaxFunEvals', 30000, 'Algorithm','active-set','TolFun',TolFun); 

for n=1:nitera  % Bucle paralelo
    x0 = pi*rand(Nrot,1);
    x = fmincon(@RotatingWPPolCN, x0, [], [], [], [], lb,ub, [], options);

    [CN, TSoP] = RotatingWPPolSops(x, philamina);

    if CN < CNB
        CNB = CN;
        XB   = x;
    end
    disp(n)
    fprintf('CN=%e   CNB=%e',CN, CNB)
             
end

  
% Dibuja los estados de polarizacion en la figura 2
figure (2)
hold on
plot3(TSoP(:,2), TSoP(:,3), TSoP(:,4),'ro', 'MarkerFaceColor','r');
hold off

% Rotaciones en gados ordenadas
rotacion = sort(XB*180/pi)