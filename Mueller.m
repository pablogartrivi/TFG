function M = Mueller(Type, Rotation, Phase)
% To generate the Mueller matrix of an element turned an angle
%   Rotation (radians), if the element is a waveplate which retarder
%   is Phase (radians).
%   
%   VARIABLES:  Type: 'polarizer', 'waveplate', 'rotated'
%               Rotation: angle to turn the element
%               Phase: retarder of the waveplate
%   
%   USE:        M = Mueller(Type, Rotation, Phase)

M = zeros(4,4);
switch lower(Type)
    case {'polarizer', 'polarizador'}
        M = zeros(4,4);
        M(1:2, 1:2) = 0.5;
        M = TurnMatrixM(M, Rotation);
      
    case {'waveplate', 'retardador'}
        M = eye(4);
        M(3,3) = cos(Phase);
        M(4,4) = cos(Phase);
        M(3,4) = sin(Phase);
        M(4,3) = -sin(Phase); 
        M = TurnMatrixM(M, Rotation);
        
    case 'rotated'
        M = eye(4);
        M(2,2) = cos(2*Rotation);
        M(3,3) = cos(2*Rotation);
        M(2,3) = sin(2*Rotation);
        M(3,2) = -sin(2*Rotation);            
end
        
 