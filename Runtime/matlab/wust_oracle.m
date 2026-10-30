function [loss, gradient, cache] = wust_oracle(c, obs, fi, cfg)
    % Source-projected least squares at one frequency on prepared observations.
    % The exact discrete gradient is with respect to slowness s = 1/c [s/m].
    % Per-source scale a=(u'*d)/(u'*u); the envelope theorem removes da/ds.
    timer = tic;
    ny = obs.shape_yx(1);
    nx = obs.shape_yx(2);
    nt = obs.n_tx;
    dispersion = [];
    if strcmp(cfg.wavenumber, 'kwave-ldr9')
        dispersion = cfg.dispersion;
    end
    solver = HelmholtzSolver(obs.x_m, obs.y_m, double(c), zeros(ny, nx), ...
                             obs.frequencies_hz(fi), -1, cfg.pml_strength, cfg.pml_m, ...
                             struct('backend', cfg.backend, 'stencil_bounds', cfg.stencil_bounds, ...
                                    'dispersion', dispersion));
    sync(cfg.backend);
    factorSeconds = toc(timer);
    timer = tic;
    gradient = zeros(ny, nx);
    loss = 0;
    virtual = cell(0);
    scales = cell(0);
    forwardSeconds = 0;
    adjointSeconds = 0;
    for first = 1:cfg.source_batch_size:nt
        ids = first:min(first + cfg.source_batch_size - 1, nt);
        nb = numel(ids);
        source = zeros(ny, nx, nb, 'single');
        for j = 1:nb
            source(obs.tx_index(ids(j)) + (j - 1) * ny * nx) = 1;
        end
        solveTimer = tic;
        [u, v] = solver.solve(source, false);
        sync(cfg.backend);
        forwardSeconds = forwardSeconds + toc(solveTimer);
        u = reshape(u, ny * nx, nb);
        predicted = u(obs.rx_index, :);
        mask = double(obs.mask(ids, :, fi).');
        observed = double(obs.Y(ids, :, fi).');
        if strcmp(cfg.backend, 'gpu')
            mask = gpuArray(single(mask));
            observed = gpuArray(single(observed));
        end
        predicted = predicted .* mask;
        observed = observed .* mask;
        denominator = sum(abs(predicted).^2, 1);
        numerator = sum(conj(predicted) .* observed, 1);
        assert(all(local(isfinite(denominator) & denominator > realmin(classUnderlyingLocal(denominator)))) && ...
               all(local(isfinite(numerator))), 'WUST:NumericalFailure', ...
               'Unusable source-scale denominator/numerator');
        scale = numerator ./ denominator;
        assert(all(local(isfinite(scale))), 'WUST:NumericalFailure', 'Nonfinite source scale');
        residual = (predicted .* scale - observed) .* mask;
        loss = loss + 0.5 * double(local(sum(abs(residual).^2, 'all')));
        rhs = zeros(ny * nx, nb, 'like', residual);
        rhs(obs.rx_index, :) = residual;
        solveTimer = tic;
        adjoint = solver.solve(reshape(rhs, ny, nx, nb), true);
        adjoint = solver.massApply(adjoint, true);
        gradient = gradient + double(local(sum(-real(conj(v .* reshape(scale, 1, 1, nb)) .* adjoint), 3)));
        sync(cfg.backend);
        adjointSeconds = adjointSeconds + toc(solveTimer);
        virtual{end + 1} = v; %#ok<AGROW>
        scales{end + 1} = scale; %#ok<AGROW>
    end
    assert(isfinite(loss) && all(isfinite(gradient(:))), ...
           'WUST:NumericalFailure', 'Nonfinite objective/gradient');
    cache = struct('solver', solver, 'virtual', {virtual}, 'scales', {scales}, ...
                   'data_norm', obs.frequency_data_norm(fi), 'factor_seconds', factorSeconds, ...
                   'forward_seconds', forwardSeconds, 'adjoint_seconds', adjointSeconds, ...
                   'oracle_seconds', factorSeconds + toc(timer), 'forward_rhs', nt, 'adjoint_rhs', nt);
end

function name = classUnderlyingLocal(a)
    if isa(a, 'gpuArray')
        name = classUnderlying(a);
    else
        name = class(a);
    end
end

function a = local(a)
    if isa(a, 'gpuArray')
        a = gather(a);
    end
end

function sync(backend)
    if strcmp(backend, 'gpu')
        wait(gpuDevice);
    end
end
