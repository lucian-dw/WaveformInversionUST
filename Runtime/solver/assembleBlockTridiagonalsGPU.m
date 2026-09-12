function [Dd, Dl, Du, Ld, Ll, Lu, Ud, Ul, Uu] = ...
    assembleBlockTridiagonalsGPU(A, B, C, k, b, d, e, h, g, Nx, Ny)
%ASSEMBLEBLOCKTRIDIAGONALSGPU Form the nine block diagonals on the GPU.
%   The fixed nine-point stencil is mapped directly to the diagonal,
%   lower-block and upper-block arrays consumed by decompBlockLU.  This
%   avoids constructing host block arrays from the global sparse matrix.

%   Arithmetic is intentionally performed in double precision before the
%   final single cast so that this path follows the existing CPU assembly
%   as closely as possible.

narginchk(11, 11);
validateattributes(b, {'single', 'double'}, ...
    {'scalar', 'real', 'finite'}, mfilename, 'b');
validateattributes(d, {'single', 'double'}, ...
    {'scalar', 'real', 'finite'}, mfilename, 'd');
validateattributes(e, {'single', 'double'}, ...
    {'scalar', 'real', 'finite'}, mfilename, 'e');
validateattributes(h, {'single', 'double'}, ...
    {'scalar', 'real', 'finite', 'positive'}, mfilename, 'h');
validateattributes(g, {'single', 'double'}, ...
    {'scalar', 'real', 'finite', 'positive'}, mfilename, 'g');
validateattributes(Nx, {'single', 'double'}, ...
    {'scalar', 'real', 'finite', 'integer', '>=', 3}, mfilename, 'Nx');
validateattributes(Ny, {'single', 'double'}, ...
    {'scalar', 'real', 'finite', 'integer', '>=', 3}, mfilename, 'Ny');

% Downstream arithmetic is intentionally double regardless of the stencil
% helper's scalar output class.
b = double(b);
d = double(d);
e = double(e);
h = double(h);
g = double(g);
Nx = double(Nx);
Ny = double(Ny);

Ag = gpuArray(double(A));
Bg = gpuArray(double(B));
Cg = gpuArray(double(C));
kg = gpuArray(double(k));
k2 = kg .* kg;
h2 = h * h;
g2 = g * g;

Dd = complex(zeros(Ny, Nx, 'double', 'gpuArray'));
Dl = complex(zeros(Ny-1, Nx, 'double', 'gpuArray'));
Du = complex(zeros(Ny-1, Nx, 'double', 'gpuArray'));
Ld = complex(zeros(Ny, Nx-1, 'double', 'gpuArray'));
Ll = complex(zeros(Ny-1, Nx-1, 'double', 'gpuArray'));
Lu = complex(zeros(Ny-1, Nx-1, 'double', 'gpuArray'));
Ud = complex(zeros(Ny, Nx-1, 'double', 'gpuArray'));
Ul = complex(zeros(Ny-1, Nx-1, 'double', 'gpuArray'));
Uu = complex(zeros(Ny-1, Nx-1, 'double', 'gpuArray'));

% Dirichlet boundary rows have a unit diagonal and no off-diagonal terms.
Dd([1 Ny], :) = 1;
Dd(2:Ny-1, [1 Nx]) = 1;

y = 2:Ny-1;
x = 2:Nx-1;
Dd(y, x) = (1-d-e) .* Cg(y, x) .* k2(y, x) - ...
    b .* (Ag(y, x) + Ag(y, x-1) + ...
    Bg(y, x)./g2 + Bg(y-1, x)./g2) ./ h2;

% Within-block lower and upper diagonals.
Dl(1:Ny-2, x) = (b .* Bg(1:Ny-2, x)./g2 - ...
    ((1-b)/2) .* (Ag(1:Ny-2, x) + Ag(1:Ny-2, x-1))) ./ h2 + ...
    (d/4) .* Cg(1:Ny-2, x) .* k2(1:Ny-2, x);
Du(y, x) = (b .* Bg(y, x)./g2 - ...
    ((1-b)/2) .* (Ag(y+1, x) + Ag(y+1, x-1))) ./ h2 + ...
    (d/4) .* Cg(y+1, x) .* k2(y+1, x);

% Lower block: rows in x+1, columns in x.
xl = 1:Nx-2;
Ld(y, xl) = (b .* Ag(y, xl) - ...
    ((1-b)/2) .* (Bg(y, xl)./g2 + Bg(y-1, xl)./g2)) ./ h2 + ...
    (d/4) .* Cg(y, xl) .* k2(y, xl);
Ll(1:Ny-2, xl) = ((1-b)/2) .* ...
    (Ag(1:Ny-2, xl) + Bg(1:Ny-2, xl)./g2) ./ h2 + ...
    (e/4) .* Cg(1:Ny-2, xl) .* k2(1:Ny-2, xl);
Lu(y, xl) = ((1-b)/2) .* ...
    (Ag(y+1, xl) + Bg(y, xl)./g2) ./ h2 + ...
    (e/4) .* Cg(y+1, xl) .* k2(y+1, xl);

% Upper block: rows in x, columns in x+1.
xu = 2:Nx-1;
Ud(y, xu) = (b .* Ag(y, xu) - ...
    ((1-b)/2) .* (Bg(y, xu+1)./g2 + Bg(y-1, xu+1)./g2)) ./ h2 + ...
    (d/4) .* Cg(y, xu+1) .* k2(y, xu+1);
Ul(1:Ny-2, xu) = ((1-b)/2) .* ...
    (Ag(1:Ny-2, xu) + Bg(1:Ny-2, xu+1)./g2) ./ h2 + ...
    (e/4) .* Cg(1:Ny-2, xu+1) .* k2(1:Ny-2, xu+1);
Uu(y, xu) = ((1-b)/2) .* ...
    (Ag(y+1, xu) + Bg(y, xu+1)./g2) ./ h2 + ...
    (e/4) .* Cg(y+1, xu+1) .* k2(y+1, xu+1);

Dd = single(Dd); Dl = single(Dl); Du = single(Du);
Ld = single(Ld); Ll = single(Ll); Lu = single(Lu);
Ud = single(Ud); Ul = single(Ul); Uu = single(Uu);
end
