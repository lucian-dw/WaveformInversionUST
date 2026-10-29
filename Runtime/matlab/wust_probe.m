function facts = wust_probe(output, backend)
% Explicit deployment diagnostic; static discovery does not call this function.
if nargin < 2
    backend = 'cpu';
end
assert(ismember(string(backend), ["cpu", "gpu"]), 'WUST:InvalidInput', ...
    'Probe backend must be cpu or gpu');
facts = struct('matlab_available', true, 'matlab_version', version, ...
    'platform', computer, 'requested_backend', backend, ...
    'cpu_available', [], 'gpu_available', [], 'failure_reason', '', 'mex', {{}});
try
    wust_setup(backend);
    if strcmp(backend, 'gpu')
        device = gpuDevice;
        facts.gpu_device = device.Name;
    end
    x = (-3:3)*.001;
    speed = 1500*ones(7);
    options = struct('stencil_bounds', [1400, 1700], 'backend', backend);
    solver = HelmholtzSolver(x, x, speed, zeros(7), 1e5, -1, 10, .001, options);
    source = zeros(7);
    source(4,4) = 1;
    wave = solver.solve(source, false);
    if isa(wave, 'gpuArray')
        wave = gather(wave);
    end
    assert(all(isfinite(wave(:))), 'WUST:NumericalFailure', 'Nonfinite probe result');
    facts.([backend '_available']) = true;
catch err
    facts.([backend '_available']) = false;
    facts.failure_reason = err.message;
end
if strcmp(backend, 'gpu')
    for name = {'decompBlockLU', 'applyBlockLU'}
        facts.mex{end+1} = struct('name', name{1}, 'path', which(name{1}), ...
            'available', exist(name{1}, 'file') == 3);
    end
end
if nargin > 0 && ~isempty(output)
    fid = fopen(output, 'w');
    assert(fid >= 0, 'WUST:InvalidInput', 'Cannot write diagnostic output');
    close = onCleanup(@() fclose(fid));
    fprintf(fid, '%s', jsonencode(facts));
end
end
