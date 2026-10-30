classdef HelmholtzSolver
    % Nine-point Helmholtz/PML solve; arrays [row=y,col=x], positions m,
    % sound speed m/s, frequency Hz. Source RHS: [ny,nx,n_source].
    % CPU owns a sparse double matrix; GPU owns complex-single block inverses.
    % options require backend and fixed [min,max] stencil_bounds;
    % optional dispersion is empty for continuum.
    % Created by Rehman Ali; maintained discrete mass derivative included.

    properties (SetAccess = private)
        backend
    end

    properties
        cpu_factor
        assembly_seconds
        factor_seconds
        % GPU block factors, retained until the last solve/curvature consumer.
        Ld % Ny x (Nx-1) gpuArray of main diagonals on lower block matrices
        Ll % (Ny-1) x (Nx-1) gpuArray of lower diagonals on lower block matrices
        Lu % (Ny-1) x (Nx-1) gpuArray of upper diagonals on lower block matrices
        Ud % Ny x (Nx-1) gpuArray of main diagonals on upper block matrices
        Ul % (Ny-1) x (Nx-1) gpuArray of lower diagonals on upper block matrices
        Uu % (Ny-1) x (Nx-1) gpuArray of upper diagonals on upper block matrices
        invT % Ny x Ny x Nx gpuArray of Schur Complement Inverses
        virtualSourceMultiplier
        massD
        massE
    end

    methods

        function obj = HelmholtzSolver(x, y, ...
                                       vel, atten, f, signConvention, a0, L_PML, options)
            assert(nargin == 9 && isstruct(options) && ...
                   all(isfield(options, {'backend', 'stencil_bounds'})), ...
                   'WUST:InvalidInput', 'Explicit backend and fixed stencil_bounds are required');
            bounds = options.stencil_bounds;
            assert(isnumeric(bounds) && isreal(bounds) && numel(bounds) == 2 && ...
                   all(isfinite(bounds)) && bounds(1) > 0 && bounds(2) > bounds(1), ...
                   'WUST:InvalidInput', 'stencil_bounds must be a positive ordered [min,max] pair');
            if ~isfield(options, 'dispersion')
                options.dispersion = [];
            end
            assert(ismember(string(options.backend), ["cpu", "gpu"]), ...
                   'WUST:InvalidInput', 'backend must be cpu or gpu');
            assert(isequal(size(vel), [numel(y), numel(x)]) && min(size(vel)) >= 3 && ...
                   isequal(size(atten), size(vel)) && all(isfinite(vel(:)) & vel(:) > 0), ...
                   'WUST:InvalidInput', 'Expected a positive finite [ny,nx] medium');
            assert(all(diff(x) > 0) && all(diff(y) > 0) && ...
                   max(abs(diff(x) - mean(diff(x)))) < mean(diff(x)) * 1e-7 && ...
                   max(abs(diff(y) - mean(diff(y)))) < mean(diff(y)) * 1e-7, ...
                   'WUST:InvalidInput', 'Uniform increasing grids required');
            assemblyTimer = tic;
            obj.backend = char(options.backend);
            % Extract Grid Information
            h = mean(diff(x)); % Grid Spacing in X [length]
            gh = mean(diff(y));
            g = gh / h; % Grid Spacing in Y [length]
            xmin = min(x);
            xmax = max(x); % [length]
            ymin = min(y);
            ymax = max(y); % [length]
            Nx = numel(x);
            Ny = numel(y); % Grid Points in X and Y

            % Calculate Complex Sound Speed
            SI = atten / (2 * pi);
            velocity = 1 ./ (1 ./ vel + 1i * SI * sign(signConvention));

            % Calculate Complex Wavenumber
            k = (2 * pi * f ./ velocity); % Wavenumber [1/m]

            % Generate Functions for PML
            xe = linspace(xmin, xmax, 2 * (Nx - 1) + 1);
            ye = linspace(ymin, ymax, 2 * (Ny - 1) + 1);
            [Xe, Ye] = meshgrid(xe, ye);
            xctr = (xmin + xmax) / 2;
            xspan = (xmax - xmin) / 2;
            yctr = (ymin + ymax) / 2;
            yspan = (ymax - ymin) / 2;
            sx = 2 * pi * a0 * f * ((max(abs(Xe - xctr) - xspan + L_PML, 0) / L_PML).^2);
            sy = 2 * pi * a0 * f * ((max(abs(Ye - yctr) - yspan + L_PML, 0) / L_PML).^2);
            ex = 1 + 1i * sx * sign(signConvention) / (2 * pi * f);
            ey = 1 + 1i * sy * sign(signConvention) / (2 * pi * f);
            A = ey ./ ex;
            A = A(1:2:end, 2:2:end);
            B = ex ./ ey;
            B = B(2:2:end, 1:2:end);
            C = ex .* ey;
            C = C(1:2:end, 1:2:end);

            % Linear Indexing Into Sparse Array
            lin_idx = @(x_idx, y_idx) y_idx + Ny * (x_idx - 1);
            val_idx = 1;

            % Optimal Stencil Parameters
            [b, d, e] = stencilOptParams(double(bounds(1)), double(bounds(2)), f, h, g);
            % Fork: optional frozen k-Wave-aware dispersion; no inferred dt.
            obj.massD = d;
            obj.massE = e;
            derivativeSlowness = 8 * pi^2 * f^2 ./ velocity;
            if ~isempty(options.dispersion)
                spec = options.dispersion;
                assert(all(atten(:) == 0) && abs(g - 1) < 1e-10, ...
                       'LDR9 requires zero attenuation and square grid cells');
                [q, dq] = kwaveLdr9WavenumberSquared(vel, f, h, b, d, e, ...
                                                     spec.time_step_s, spec.reference_speed_mps);
                k = sqrt(q);
                derivativeSlowness = dq;
            end

            [Dd, Dl, Du, Ld, Ll, Lu, Ud, Ul, Uu] = ...
                ninePointCoefficients(A, B, C, k, b, d, e, h, g, Nx, Ny);
            if strcmp(obj.backend, 'cpu')
                rows = zeros(9 * Nx * Ny, 1);
                cols = rows;
                vals = complex(rows);
                for x_idx = 1:Nx
                    for y_idx = 1:Ny
                        row = lin_idx(x_idx, y_idx);
                        if x_idx == 1 || x_idx == Nx || y_idx == 1 || y_idx == Ny
                            columns = row;
                            coefficients = 1;
                        else
                            columns = [row, row - 1, row + 1, row - Ny, row - Ny - 1, row - Ny + 1, row + Ny, row + Ny - 1, row + Ny + 1];
                            coefficients = [Dd(y_idx, x_idx), Dl(y_idx - 1, x_idx), Du(y_idx, x_idx), ...
                                            Ld(y_idx, x_idx - 1), Ll(y_idx - 1, x_idx - 1), Lu(y_idx, x_idx - 1), ...
                                            Ud(y_idx, x_idx), Ul(y_idx - 1, x_idx), Uu(y_idx, x_idx)];
                        end
                        ids = val_idx:val_idx + numel(columns) - 1;
                        rows(ids) = row;
                        cols(ids) = columns;
                        vals(ids) = coefficients;
                        val_idx = val_idx + numel(columns);
                    end
                end
                used = 1:val_idx - 1;
                helmholtz = sparse(rows(used), cols(used), vals(used), Nx * Ny, Nx * Ny);
                obj.virtualSourceMultiplier = C .* derivativeSlowness;
                obj.assembly_seconds = toc(assemblyTimer);
                factorTimer = tic;
                obj.cpu_factor = decomposition(helmholtz, 'lu');
                obj.factor_seconds = toc(factorTimer);
            end
            if strcmp(obj.backend, 'gpu')
                obj.virtualSourceMultiplier = ...
                    gpuArray(complex(single(C .* derivativeSlowness)));
            end

            % Compute Block LU Factorization on GPU
            if strcmp(obj.backend, 'gpu') % Check if GPU can be used to solve linear system
                obj.Ld = gpuArray(single(Ld));
                obj.Ll = gpuArray(single(Ll));
                obj.Lu = gpuArray(single(Lu));
                obj.Ud = gpuArray(single(Ud));
                obj.Ul = gpuArray(single(Ul));
                obj.Uu = gpuArray(single(Uu));
                Dd = gpuArray(single(Dd));
                Dl = gpuArray(single(Dl));
                Du = gpuArray(single(Du));
                wait(gpuDevice);
                obj.assembly_seconds = toc(assemblyTimer);
                factorTimer = tic;
                clear Ld Ll Lu Ud Ul Uu;
                % Two-Step Solution
                % B) Calculate the Schur Complements and Their Inverses
                obj.invT = decompBlockLU(obj.Ld, obj.Ll, obj.Lu, ...
                                         Dd, Dl, Du, obj.Ud, obj.Ul, obj.Uu);
                wait(gpuDevice);
                obj.factor_seconds = toc(factorTimer);
            end
        end

        function [wvfield, virtSrcs] = solve(obj, src, adjoint)
            % Solve H u = src or H' u = src; virtual sources are dH/ds factors.
            % Number of Grid Points and Sources
            [Ny, Nx, Nsrcs] = size(src);

            % Solve Helmholtz Equation Depending on Availability of GPU
            if strcmp(obj.backend, 'gpu')
                % 2) Solve Helmholtz Equation Using Block LU Decomposition
                src = gpuArray(complex(single(src)));
                wvfield = applyBlockLU(src, obj.Ld, obj.Ll, obj.Lu, ...
                                       obj.Ud, obj.Ul, obj.Uu, obj.invT, adjoint);
            else
                % Reuse the same LU and its conjugate transpose for every RHS batch.
                src = double(src); % MATLAB sparse backslash requires double RHS.
                if adjoint
                    sol = obj.cpu_factor' \ reshape(src, [Nx * Ny, Nsrcs]);
                else
                    sol = obj.cpu_factor \ reshape(src, [Nx * Ny, Nsrcs]);
                end
                wvfield = reshape(sol, size(src));
            end

            % Compute [dH/ds u] where H is Helmholtz matrix and u is the wavefield
            if nargout > 1
                virtSrcs = obj.virtualSourceMultiplier .* wvfield;
            end

            % GPU solves remain resident. Callers gather only final diagnostics.

        end

        function out = massApply(obj, v, adjoint)
            % Exact derivative of the optimized nine-point mass stencil.
            % Legacy virtual-source multiplication alone drops off-diagonals.
            d = obj.massD;
            e = obj.massE;
            kernel = [e / 4, d / 4, e / 4; d / 4, 1 - d - e, d / 4; e / 4, d / 4, e / 4];
            if adjoint
                v([1, end], :, :) = 0;
                v(:, [1, end], :) = 0;
            end
            out = convn(v, kernel, 'same');
            if ~adjoint
                out([1, end], :, :) = 0;
                out(:, [1, end], :) = 0;
            end
        end

    end
end
