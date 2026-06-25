function TM = TurnMatrixM(M, Rotation)
% Turn the Mueller matrix of an element M, an angle Rotation
%
%   VARIABLES:  M: Mueller Matrix of the element that we want to rotate
%               Rotation: Rotation angle
%   
%   USE:        TM = TurnMatrixM(M, Rotation)

TM = Mueller('rotated', -Rotation, 0.0) * M * Mueller('rotated', Rotation, 0.0);