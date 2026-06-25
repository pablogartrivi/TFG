function CN = RotatingWPPolCN(rot)

% philamina = pi/2;
philamina = pi/2;

Sini = [1,1,0,0]';

% Numero de rotaciones
Nrot = length(rot);

A = zeros(Nrot,4);
for k=1:Nrot
    M = Mueller('waveplate', rot(k), philamina);
    A(k,:) = M * Sini;
end

% Condition Number
CN  = cond(A);

% CN  = (CN - sqrt(3))*1.e6;