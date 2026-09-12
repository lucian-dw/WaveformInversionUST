classdef HelmholtzSolver
    % HELMHOLTZSOLVER Solver for Helmholtz Equation with PML
    %
    % Constructor: 
    %   obj = HelmholtzSolver(x, y, vel, atten, f, signConvention, a0, L_PML)
    % 
    % Properties:
    %   x, y = 1 x Nx and 1 x Ny arrays of x and y positions, respectively
    %   vel = Ny x Nx array of wave velocities [length/time]
    %   atten = Ny x Nx array of attenuations [Neper/(length/time)]
    %   f = Wave Frequency [1/time]
    %   signConvention = -1 for exp(-ikr), +1 for exp(+ikr)
    %   a0 = PML strength parameter from Chen/Cheng/Feng/Wu 2013 Paper
    %   L_PML = Length [length] of PML
    %   Computed During Constructor:
    %       HelmholtzEqn = sparse array of dimension Ny*Nx x Ny*Nx
    %       if wfiUseGPU:
    %           Ld = Ny x (Nx-1) gpuArray of main diagonals on lower block matrices
    %           Ll = (Ny-1) x (Nx-1) gpuArray of lower diagonals on lower block matrices
    %           Lu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on lower block matrices
    %           Ud = Ny x (Nx-1) gpuArray of main diagonals on upper block matrices
    %           Ul = (Ny-1) x (Nx-1) gpuArray of lower diagonals on upper block matrices
    %           Uu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on upper block matrices
    %           invT = Ny x Ny x Nx gpuArray of Schur Complement Inverses
    %
    % Methods:
    %   Constructor:
    %       obj = HelmholtzSolver(x, y, vel, atten, f, signConvention, a0, L_PML)
    %           x, y = 1 x Nx and 1 x Ny arrays of x and y positions, respectively
    %           vel = Ny x Nx array of wave velocities [length/time]
    %           atten = Ny x Nx array of attenuations [Neper/(length/time)]
    %           f = Wave Frequency [1/time]
    %           signConvention = -1 for exp(-ikr), +1 for exp(+ikr)
    %           a0 = PML strength parameter from Chen/Cheng/Feng/Wu 2013 Paper
    %           L_PML = Length [length] of PML
    %           Computed During Constructor:
    %               obj.HelmholtzEqn = sparse array of dimension Ny*Nx x Ny*Nx
    %               if wfiUseGPU:
    %                   obj.Ld = Ny x (Nx-1) gpuArray of main diagonals on lower block matrices
    %                   obj.Ll = (Ny-1) x (Nx-1) gpuArray of lower diagonals on lower block matrices
    %                   obj.Lu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on lower block matrices
    %                   obj.Ud = Ny x (Nx-1) gpuArray of main diagonals on upper block matrices
    %                   obj.Ul = (Ny-1) x (Nx-1) gpuArray of lower diagonals on upper block matrices
    %                   obj.Uu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on upper block matrices
    %                   obj.invT = Ny x Ny x Nx gpuArray of Schur Complement Inverses
    %   
    %   Solver Method:
    %       [wvfield, virtSrcs] = obj.solve(src, adjoint)
    %           obj = HelmholtzSolver object containing fields:
    %               x, y = 1 x Nx and 1 x Ny arrays of x and y positions, respectively
    %               vel = Ny x Nx array of wave velocities [length/time]
    %               atten = Ny x Nx array of attenuations [Neper/(length/time)]
    %               f = Wave Frequency [1/time]
    %               signConvention = -1 for exp(-ikr), +1 for exp(+ikr)
    %               a0 = PML strength parameter from Chen/Cheng/Feng/Wu 2013 Paper
    %               L_PML = Length [length] of PML
    %               HelmholtzEqn = sparse array of dimension Ny*Nx x Ny*Nx
    %               if wfiUseGPU:
    %                   Ld = Ny x (Nx-1) gpuArray of main diagonals on lower block matrices
    %                   Ll = (Ny-1) x (Nx-1) gpuArray of lower diagonals on lower block matrices
    %                   Lu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on lower block matrices
    %                   Ud = Ny x (Nx-1) gpuArray of main diagonals on upper block matrices
    %                   Ul = (Ny-1) x (Nx-1) gpuArray of lower diagonals on upper block matrices
    %                   Uu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on upper block matrices
    %                   invT = Ny x Ny x Nx gpuArray of Schur Complement Inverses
    %           src = Ny x Nx x K array of K source images (Ny x Nx)
    %           adjoint = true if solving adjoint Helmholtz; 
    %                     false if solving normal Helmholtz
    %           wvfield = Ny x Nx x K array of K solved wavefields (Ny x Nx) vs space
    %           virtSrcs = Ny x Nx x K array of K solved virtual sources (Ny x Nx) vs space
    %       
    %   Created by Rehman Ali, University of Rochester Medical Center

    properties
        % 1) Inputs to Constructor
        x % 1xN array of x grid positions
        y % 1xM array of y grid positions
        vel % MxN array of wave velocities [length/time]
        atten % MxN array of attenuations [Neper/(length/time)]
        f % Wave Frequency [1/time]
        signConvention % -1 for exp(-ikr), +1 for exp(+ikr)
        a0 % PML strength parameter from Chen/Cheng/Feng/Wu 2013 Paper
        L_PML % Length [length] of PML
        % 2) Computed During Constructor
        HelmholtzEqn % Sparse System of Equations for Helmholtz Equation
        PML % PML factors over identity term of stencil
        V % Complex Velocity [m/s] for Virtual Source Calculation
        % 3) Computed During Constructor if wfiUseGPU
        Ld % Ny x (Nx-1) gpuArray of main diagonals on lower block matrices
        Ll % (Ny-1) x (Nx-1) gpuArray of lower diagonals on lower block matrices
        Lu % (Ny-1) x (Nx-1) gpuArray of upper diagonals on lower block matrices
        Ud % Ny x (Nx-1) gpuArray of main diagonals on upper block matrices
        Ul % (Ny-1) x (Nx-1) gpuArray of lower diagonals on upper block matrices
        Uu % (Ny-1) x (Nx-1) gpuArray of upper diagonals on upper block matrices
        invT % Ny x Ny x Nx gpuArray of Schur Complement Inverses
        luFactors % Ny x Ny x Nx gpuArray of Schur Complement LU factors
        pivots % (Ny-2) x Nx int32 gpuArray of LU pivots
        factorBackend = 'inverse'
        virtualSourceMultiplier
        derivativeSlowness
        massD
        massE
    end

    methods
        function obj = HelmholtzSolver(x, y, ...
            vel, atten, f, signConvention, a0, L_PML, varargin)
            %HELMHOLTZSOLVER Construct instance of HelmholtzSolver class
            % Constructor:
            %   obj = HelmholtzSolver(x, y, vel, atten, f, signConvention, a0, L_PML)
            % Inputs to Constructor:
            %   x, y = 1 x Nx and 1 x Ny arrays of x and y positions, respectively
            %   vel = Ny x Nx array of wave velocities [length/time]
            %   atten = Ny x Nx array of attenuations [Neper/(length/time)]
            %   f = Wave Frequency [1/time]
            %   signConvention = -1 for exp(-ikr), +1 for exp(+ikr)
            %   a0 = PML strength parameter from Chen/Cheng/Feng/Wu 2013 Paper
            %   L_PML = Length [length] of PML
            % Optional Inputs:
            %   stencilVelBounds = [] to use [min(vel(:)), max(vel(:))];
            %                      scalar to use a single reference velocity;
            %                      [vmin, vmax] to override the stencil range
            % Output of Constructor:
            %   obj = HelmholtzSolver object containing fields:
            %       x, y = 1 x Nx and 1 x Ny arrays of x and y positions, respectively
            %       vel = Ny x Nx array of wave velocities [length/time]
            %       atten = Ny x Nx array of attenuations [Neper/(length/time)]
            %       f = Wave Frequency [1/time]
            %       signConvention = -1 for exp(-ikr), +1 for exp(+ikr)
            %       a0 = PML strength parameter from Chen/Cheng/Feng/Wu 2013 Paper
            %       L_PML = Length [length] of PML
            %       HelmholtzEqn = sparse array of dimension Ny*Nx x Ny*Nx
            %       if wfiUseGPU:
            %           Ld = Ny x (Nx-1) gpuArray of main diagonals on lower block matrices
            %           Ll = (Ny-1) x (Nx-1) gpuArray of lower diagonals on lower block matrices
            %           Lu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on lower block matrices
            %           Ud = Ny x (Nx-1) gpuArray of main diagonals on upper block matrices
            %           Ul = (Ny-1) x (Nx-1) gpuArray of lower diagonals on upper block matrices
            %           Uu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on upper block matrices
            %           invT = Ny x Ny x Nx gpuArray of Schur Complement Inverses
            %

            % First Set All Input Values to Constructor
            obj.x = x; obj.y = y;
            obj.vel = vel; obj.atten = atten;
            obj.f = f; obj.signConvention = signConvention;
            obj.a0 = a0; obj.L_PML = L_PML;
            if numel(varargin) >= 2 && ~isempty(varargin{2})
                obj.factorBackend = lower(char(string(varargin{2})));
            end
            if ~ismember(obj.factorBackend, {'inverse', 'factors'})
                error('factorBackend must be inverse or factors.');
            end
            
            % Extract Grid Information
            h = mean(diff(x)); % Grid Spacing in X [length]
            gh = mean(diff(y)); g = gh/h; % Grid Spacing in Y [length]
            xmin = min(x); xmax = max(x); % [length]
            ymin = min(y); ymax = max(y); % [length]
            Nx = numel(x); Ny = numel(y); % Grid Points in X and Y
            
            % Calculate Complex Sound Speed
            SI = atten/(2*pi);
            obj.V = 1./(1./vel + 1i*SI*sign(signConvention));
                
            % Calculate Complex Wavenumber
            k = (2*pi*f./obj.V); % Wavenumber [1/m]
            
            % Generate Functions for PML
            xe = linspace(xmin, xmax, 2*(Nx-1)+1);
            ye = linspace(ymin, ymax, 2*(Ny-1)+1);
            [Xe, Ye] = meshgrid(xe,ye);
            xctr = (xmin+xmax)/2; xspan = (xmax-xmin)/2;
            yctr = (ymin+ymax)/2; yspan = (ymax-ymin)/2;
            sx = 2*pi*a0*f*((max(abs(Xe-xctr)-xspan+L_PML,0)/L_PML).^2);
            sy = 2*pi*a0*f*((max(abs(Ye-yctr)-yspan+L_PML,0)/L_PML).^2);
            ex = 1+1i*sx*sign(signConvention)/(2*pi*f); 
            ey = 1+1i*sy*sign(signConvention)/(2*pi*f);
            A = ey./ex; A = A(1:2:end,2:2:end);
            B = ex./ey; B = B(2:2:end,1:2:end);
            C = ex.*ey; C = C(1:2:end,1:2:end);
            
            % Linear Indexing Into Sparse Array
            lin_idx = @(x_idx, y_idx) y_idx+Ny*(x_idx-1); val_idx = 1;
            
            % Optional override for the stencil optimization range.
            if (~isempty(varargin) && ~isempty(varargin{1}))
                stencilVelBounds = varargin{1};
                if isscalar(stencilVelBounds)
                    vStencilMin = double(stencilVelBounds);
                    vStencilMax = double(stencilVelBounds);
                elseif (numel(stencilVelBounds) == 2)
                    stencilVelBounds = double(stencilVelBounds(:));
                    vStencilMin = stencilVelBounds(1);
                    vStencilMax = stencilVelBounds(2);
                else
                    error('stencilVelBounds must be empty, scalar, or 2-element vector.');
                end
            else
                vStencilMin = min(vel(:));
                vStencilMax = max(vel(:));
            end

            % Optimal Stencil Parameters
            [b, d, e] = stencilOptParams(vStencilMin, vStencilMax, f, h, g);
            % Fork: optional frozen k-Wave-aware dispersion; no inferred dt.
            obj.massD=d; obj.massE=e;
            obj.derivativeSlowness=8*pi^2*f^2./obj.V;
            if numel(varargin)>=3 && ~isempty(varargin{3})
                spec=varargin{3};
                assert(all(atten(:)==0) && abs(g-1)<1e-10, ...
                    'LDR9 requires zero attenuation and square grid cells');
                [q,dq]=kwaveLdr9WavenumberSquared(vel,f,h,b,d,e, ...
                    spec.time_step_s,spec.reference_speed_mps,spec.model_reference_speed_mps);
                k=sqrt(q);
                obj.derivativeSlowness=dq.*(2*spec.model_reference_speed_mps^2./vel);
            end
            
            % Structures to Form Sparse Matrices
            rows = zeros(9*(Nx-2)*(Ny-2)+(Nx*Ny-(Nx-2)*(Ny-2)),1);
            cols = zeros(9*(Nx-2)*(Ny-2)+(Nx*Ny-(Nx-2)*(Ny-2)),1);
            vals = zeros(9*(Nx-2)*(Ny-2)+(Nx*Ny-(Nx-2)*(Ny-2)),1);
            
            % Populate Sparse Matrix Structures
            for x_idx = 1:Nx
                for y_idx = 1:Ny
                    % Image of Stencil for Helmholtz Equation
                    if ((x_idx == 1) || (x_idx == Nx) || (y_idx == 1) || (y_idx == Ny))
                        % Dirichlet Boundary Condition
                        rows(val_idx) = lin_idx(x_idx,y_idx); 
                        cols(val_idx) = lin_idx(x_idx,y_idx);
                        vals(val_idx) = 1; 
                        val_idx = val_idx + 1;
                    else
                        % 9-Point Stencil 
                        % Center of Stencil
                        rows(val_idx) = lin_idx(x_idx,y_idx); 
                        cols(val_idx) = lin_idx(x_idx,y_idx);
                        vals(val_idx) = (1-d-e)*C(y_idx,x_idx)*(k(y_idx,x_idx)^2) ...
                            - b*(A(y_idx,x_idx)+A(y_idx,x_idx-1) + ...
                            B(y_idx,x_idx)/(g^2)+B(y_idx-1,x_idx)/(g^2))/(h^2);
                        val_idx = val_idx + 1;
                        % Left
                        rows(val_idx) = lin_idx(x_idx,y_idx);  
                        cols(val_idx) = lin_idx(x_idx-1,y_idx);
                        vals(val_idx) = (b*A(y_idx,x_idx-1) - ...
                            ((1-b)/2)*(B(y_idx,x_idx-1)/(g^2)+B(y_idx-1,x_idx-1)/(g^2)))/(h^2) + ...
                            (d/4)*C(y_idx,x_idx-1)*(k(y_idx,x_idx-1)^2);
                        val_idx = val_idx + 1;
                        % Right
                        rows(val_idx) = lin_idx(x_idx,y_idx); 
                        cols(val_idx) = lin_idx(x_idx+1,y_idx);
                        vals(val_idx) = (b*A(y_idx,x_idx) - ...
                            ((1-b)/2)*(B(y_idx,x_idx+1)/(g^2)+B(y_idx-1,x_idx+1)/(g^2)))/(h^2) + ...
                            (d/4)*C(y_idx,x_idx+1)*(k(y_idx,x_idx+1)^2);
                        val_idx = val_idx + 1;
                        % Down
                        rows(val_idx) = lin_idx(x_idx,y_idx);  
                        cols(val_idx) = lin_idx(x_idx,y_idx-1);
                        vals(val_idx) = (b*B(y_idx-1,x_idx)/(g^2) - ...
                            ((1-b)/2)*(A(y_idx-1,x_idx)+A(y_idx-1,x_idx-1)))/(h^2) + ...
                            (d/4)*C(y_idx-1,x_idx)*(k(y_idx-1,x_idx)^2);
                        val_idx = val_idx + 1;
                        % Up
                        rows(val_idx) = lin_idx(x_idx,y_idx); 
                        cols(val_idx) = lin_idx(x_idx,y_idx+1);
                        vals(val_idx) = (b*B(y_idx,x_idx)/(g^2) - ...
                            ((1-b)/2)*(A(y_idx+1,x_idx)+A(y_idx+1,x_idx-1)))/(h^2) + ...
                            (d/4)*C(y_idx+1,x_idx)*(k(y_idx+1,x_idx)^2);
                        val_idx = val_idx + 1;
                        % Bottom Left
                        rows(val_idx) = lin_idx(x_idx,y_idx); 
                        cols(val_idx) = lin_idx(x_idx-1,y_idx-1);
                        vals(val_idx) = (((1-b)/2)*(A(y_idx-1,x_idx-1)+B(y_idx-1,x_idx-1)/(g^2)))/(h^2) + ...
                            (e/4)*C(y_idx-1,x_idx-1)*(k(y_idx-1,x_idx-1)^2);
                        val_idx = val_idx + 1;
                        % Bottom Right
                        rows(val_idx) = lin_idx(x_idx,y_idx); 
                        cols(val_idx) = lin_idx(x_idx+1,y_idx-1);
                        vals(val_idx) = (((1-b)/2)*(A(y_idx-1,x_idx)+B(y_idx-1,x_idx+1)/(g^2)))/(h^2) + ...
                            (e/4)*C(y_idx-1,x_idx+1)*(k(y_idx-1,x_idx+1)^2);
                        val_idx = val_idx + 1;
                        % Top Left
                        rows(val_idx) = lin_idx(x_idx,y_idx); 
                        cols(val_idx) = lin_idx(x_idx-1,y_idx+1);
                        vals(val_idx) = (((1-b)/2)*(A(y_idx+1,x_idx-1)+B(y_idx,x_idx-1)/(g^2)))/(h^2) + ...
                            (e/4)*C(y_idx+1,x_idx-1)*(k(y_idx+1,x_idx-1)^2);
                        val_idx = val_idx + 1;
                        % Top Right
                        rows(val_idx) = lin_idx(x_idx,y_idx); 
                        cols(val_idx) = lin_idx(x_idx+1,y_idx+1);
                        vals(val_idx) = (((1-b)/2)*(A(y_idx+1,x_idx)+B(y_idx,x_idx+1)/(g^2)))/(h^2) + ...
                            (e/4)*C(y_idx+1,x_idx+1)*(k(y_idx+1,x_idx+1)^2);
                        val_idx = val_idx + 1;
                    end
                end
            end
            
            % Generate Left-Hand Side of Sparse Array
            obj.PML = C;
            obj.HelmholtzEqn = sparse(rows, cols, vals, Nx*Ny, Nx*Ny);
            if wfiUseGPU
                obj.virtualSourceMultiplier = ...
                    gpuArray(complex(single(obj.PML.*obj.derivativeSlowness)));
            end
            
            % Compute Block LU Factorization on GPU
            if wfiUseGPU % Check if GPU can be used to solve linear system
                % 1) Form the fixed-stencil block arrays directly on GPU.
                [Dd, Dl, Du, Ld, Ll, Lu, Ud, Ul, Uu] = ...
                    assembleBlockTridiagonalsGPU( ...
                    A, B, C, k, b, d, e, h, g, Nx, Ny);
                obj.Ld = Ld; obj.Ll = Ll; obj.Lu = Lu;
                obj.Ud = Ud; obj.Ul = Ul; obj.Uu = Uu;
                clear Ld Ll Lu Ud Ul Uu;
                % Two-Step Solution
                % B) Calculate the Schur Complements and Their Inverses
                if strcmp(obj.factorBackend, 'factors')
                    [obj.luFactors, obj.pivots] = decompBlockLUFactors( ...
                        obj.Ld, obj.Ll, obj.Lu, Dd, Dl, Du, ...
                        obj.Ud, obj.Ul, obj.Uu);
                else
                    obj.invT = decompBlockLU(obj.Ld, obj.Ll, obj.Lu, ...
                        Dd, Dl, Du, obj.Ud, obj.Ul, obj.Uu);
                end
            end
        end

        function [wvfield, virtSrcs] = solve(obj, src, adjoint)
            %SOLVE Solve Helmholtz Equation for Given Sources
            % USAGE:
            %   [wvfield, virtSrcs] = obj.solve(src, adjoint)
            % INPUTS:
            %   obj = HelmholtzSolver object containing fields:
            %       x, y = 1 x Nx and 1 x Ny arrays of x and y positions, respectively
            %       vel = Ny x Nx array of wave velocities [length/time]
            %       atten = Ny x Nx array of attenuations [Neper/(length/time)]
            %       f = Wave Frequency [1/time]
            %       signConvention = -1 for exp(-ikr), +1 for exp(+ikr)
            %       a0 = PML strength parameter from Chen/Cheng/Feng/Wu 2013 Paper
            %       L_PML = Length [length] of PML
            %       HelmholtzEqn = sparse array of dimension Ny*Nx x Ny*Nx
            %       if wfiUseGPU:
            %           Ld = Ny x (Nx-1) gpuArray of main diagonals on lower block matrices
            %           Ll = (Ny-1) x (Nx-1) gpuArray of lower diagonals on lower block matrices
            %           Lu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on lower block matrices
            %           Ud = Ny x (Nx-1) gpuArray of main diagonals on upper block matrices
            %           Ul = (Ny-1) x (Nx-1) gpuArray of lower diagonals on upper block matrices
            %           Uu = (Ny-1) x (Nx-1) gpuArray of upper diagonals on upper block matrices
            %           invT = Ny x Ny x Nx gpuArray of Schur Complement Inverses
            %   src = Ny x Nx x K array of K source images (Ny x Nx)
            %   adjoint = true if solving adjoint Helmholtz; 
            %             false if solving normal Helmholtz
            % OUTPUTS:
            %   wvfield = Ny x Nx x K array of K solved wavefields (Ny x Nx) vs space
            %   virtSrcs = Ny x Nx x K array of K solved virtual sources (Ny x Nx) vs space

            % Number of Grid Points and Sources
            [Ny, Nx, Nsrcs] = size(src);

            % Solve Helmholtz Equation Depending on Availability of GPU
            if wfiUseGPU
                % 2) Solve Helmholtz Equation Using Block LU Decomposition
                src = gpuArray(complex(single(src)));
                if strcmp(obj.factorBackend, 'factors')
                    wvfield = applyBlockLUFactors(src, obj.Ld, obj.Ll, obj.Lu, ...
                        obj.Ud, obj.Ul, obj.Uu, obj.luFactors, obj.pivots, adjoint);
                else
                    wvfield = applyBlockLU(src, obj.Ld, obj.Ll, obj.Lu, ...
                        obj.Ud, obj.Ul, obj.Uu, obj.invT, adjoint);
                end
                clearvars src Ld Ll Lu Ud Ul Uu invT;
                %wvfield = double(wvfield);
            else
                % Brute-force CPU solution of linear system
                src=double(src); % MATLAB sparse backslash requires double RHS.
                if adjoint
                    sol = (obj.HelmholtzEqn')\reshape(src,[Nx*Ny, Nsrcs]);
                else
                    sol = obj.HelmholtzEqn\reshape(src,[Nx*Ny, Nsrcs]);
                end
                wvfield = reshape(sol, size(src));
            end

            % Compute [dH/ds u] where H is Helmholtz matrix and u is the wavefield
            if nargout > 1
                if wfiUseGPU
                    % Create Sparse Matrix
                    % Keep virtual sources resident for adjoint and line search.
                    virtSrcs = obj.virtualSourceMultiplier.*wvfield;
                else
                    % Create Sparse Matrix
                    wvfield_matrix = obj.PML.*obj.derivativeSlowness;
                    % Compute Virtual Sources
                    virtSrcs = wvfield_matrix.*wvfield;
                end
            end

            % GPU solves remain resident. Callers gather only final diagnostics.

        end

        function out=massApply(obj,v,adjoint)
            % Exact derivative of the optimized nine-point mass stencil.
            % Legacy virtual-source multiplication alone drops off-diagonals.
            d=obj.massD;e=obj.massE;
            kernel=[e/4,d/4,e/4;d/4,1-d-e,d/4;e/4,d/4,e/4];
            if adjoint,v([1,end],:,:)=0;v(:,[1,end],:)=0;end
            out=convn(v,kernel,'same');
            if ~adjoint,out([1,end],:,:)=0;out(:,[1,end],:)=0;end
        end
    end
end
