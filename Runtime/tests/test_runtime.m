function test_runtime(backend)
% One backend-explicit gate for derivatives, batching and reconstruction.
if nargin < 1
    backend = 'cpu';
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));
wust_setup(backend);
check_mex_boundary(backend);
x = (-15:15)*1e-3;
spec = struct('x_m', x, 'y_m', x, ...
    'tx_index', sub2ind([31,31], [10,22,16], [8,23,25]).', ...
    'rx_index', sub2ind([31,31], [8,15,23,18], [13,24,15,8]).', ...
    'frequencies_hz', [1e5,1.2e5]);
[obs, sound_speed, source] = wust_test_measurements(spec, backend, spec.frequencies_hz);
[X, Y] = meshgrid(obs.x_m, obs.y_m);
assert(isequal(ringingRemovalFilt(x,x,sound_speed,1500,1e5,0,4), sound_speed));
filtered = ringingRemovalFilt(x,x,sound_speed,1500,1e5,.5,4);
assert(all(isfinite(filtered(:))) && norm(filtered-sound_speed,'fro') > 0);
cfg = struct('backend', backend, 'pml_m', .004, 'pml_strength', 10, ...
    'stencil_bounds', [1400,1700], 'wavenumber', 'continuum', 'source_batch_size', 2, ...
    'step_damping', .25, 'max_update_mps', 12, 'bounds_mps', [1300,1800], ...
    'filter_cutoff', 0, 'filter_order', 4, ...
    'frequency_schedule_hz', [1e5,1e5,1.2e5,1.2e5], 'update_mask', true(size(sound_speed)));
cfg.update_mask([1:5,end-4:end], :) = false;
cfg.update_mask(:, [1:5,end-4:end]) = false;
check_update_rules(x, cfg);
options = struct('backend', backend, 'stencil_bounds', cfg.stencil_bounds);
solver = HelmholtzSolver(x, x, sound_speed, zeros(size(sound_speed)), ...
    1e5, -1, 10, .004, options);
% Setup cannot change an already constructed instance's backend.
wust_setup('cpu');
wave = solver.solve(source, false);
assert(isa(wave,'gpuArray') == strcmp(backend,'gpu'));
initial = 1500*ones(size(sound_speed));
direction = exp(-((X-.002).^2+(Y+.003).^2)/2e-5).*cfg.update_mask*1e-6;
step = 1e-3;
tolerance = 1e-4;
if strcmp(backend,'gpu')
    % Complex-single LU needs a resolvable finite difference.
    step = .1;
    tolerance = .03;
end
[loss, gradient] = wust_oracle(initial, obs, 1, cfg);
check_derivative(initial, obs, cfg, direction, step, tolerance);
batched = cfg;
batched.source_batch_size = 3;
[batch_loss, batch_gradient] = wust_oracle(initial, obs, 1, batched);
assert(abs(batch_loss-loss)/max(loss,eps) < 1e-4);
assert(norm(batch_gradient-gradient,'fro')/norm(gradient,'fro') < 1e-4);
result = wust_reconstruct(obs, initial, cfg);
assert(all(isfinite(result.c_mps(:))));
assert(strcmp(result.completion.reason, 'schedule_complete'));
assert(~result.completion.converged && result.completed_updates == 4);
assert(result.planned_schedule_length == 4);
assert(isempty(result.final_data_residual) && result.records(end).loss_model_step == 3);
final_loss = wust_oracle(result.c_mps, obs, 1, cfg);
assert(final_loss < loss, 'Training objective did not decrease');
fprintf('%s FWI loss %.6g -> %.6g\n', backend, loss, final_loss);
short = cfg;
short.max_updates = 1;
short.save_history = true;
short.evaluate_final_residual = true;
result = wust_reconstruct(obs, initial, short);
assert(strcmp(result.completion.reason,'caller_truncated_schedule'));
assert(result.planned_schedule_length == 4 && result.completed_updates == 1);
assert(isequal(size(result.history_mps), [31,31,2]));
assert(result.final_data_residual.oracle_calls == numel(obs.frequencies_hz));
assert(result.final_data_residual.forward_rhs == numel(obs.tx_index)*numel(obs.frequencies_hz));
assert(isfinite(result.final_data_residual.relative));
zero = cfg;
zero.max_updates = 0;
result = wust_reconstruct(obs, initial, zero);
assert(isequal(result.c_mps,initial) && isempty(result.records));
assert(result.completed_updates == 0);
zero_data = obs;
zero_data.Y(:) = 0;
zero_run = cfg;
zero_run.max_updates = 1;
result = wust_reconstruct(zero_data, initial, zero_run);
assert(isequal(result.c_mps, initial));
assert(result.records.alpha == 0 && result.records.absolute_residual_before_update == 0);
assert(isnan(result.records.relative_residual_before_update));
assert(strcmp(result.records.relative_residual_status, 'zero_observation_energy'));
encoded = jsonencode(result.records);
assert(contains(encoded, '"relative_residual_before_update":null'));
default = rmfield(cfg, 'frequency_schedule_hz');
result = wust_reconstruct(obs, initial, default);
assert(result.planned_schedule_length == 2);
assert(isequal(result.executed_frequencies_hz(:), [1e5;1.2e5]));
bad = obs;
bad.mask(1,:,1) = false;
bad.mask(1,1,1) = true;
reject(@() wust_reconstruct(bad,initial,cfg), 'WUST:InvalidInput');
bad = obs;
bad.Y(:) = realmax/2;
reject(@() wust_reconstruct(bad,initial,cfg), 'WUST:NumericalFailure');
cfg.wavenumber = 'kwave-ldr9';
cfg.dispersion = struct('time_step_s',2e-8, ...
    'reference_speed_mps',1700);
check_derivative(initial, obs, cfg, direction, step, tolerance);
end

function check_update_rules(x, cfg)
mask = cfg.update_mask;
g = ones(size(mask));
g(~mask) = -1e10;
[roi, z] = wust_roi_gradient(g, mask, x, x, 1500, 1e5, .5, 4);
g(~mask) = 1e20;
[~, changed] = wust_roi_gradient(g, mask, x, x, 1500, 1e5, .5, 4);
assert(isequal(z, changed) && all(z(~mask) == 0));
assert(sum(roi(:).*(-z(:))) < 0, 'Masked filtered direction must descend');
[d, beta, gz] = wust_ncg_direction([1;2], [2;1], [1;.5], 1.5, [-1;-2], true);
assert(abs(beta-4/3) < 1e-14 && gz == 4 && sum([1;2].*d) < 0);
[d, beta] = wust_ncg_direction([1;2], [2;1], [1;.5], 1.5, [100;100], true);
assert(beta == 0 && isequal(d, [-2;-1]));
[~, beta] = wust_ncg_direction([1;2], [2;1], [1;.5], 1.5, [-1;-2], false);
assert(beta == 0);
initial = 1500*ones(size(mask));
crossing = -ones(size(mask));
updated = wust_project_slowness_update(initial, crossing, 1, cfg);
assert(all(updated(mask) == 1512) && isequal(updated(~mask), initial(~mask)));
assert(all(isfinite(updated(:)) & updated(:) > 0));
end

function check_derivative(initial, obs, cfg, direction, step, tolerance)
[~, gradient] = wust_oracle(initial, obs, 1, cfg);
plus = wust_oracle(1./(1./initial+step*direction), obs, 1, cfg);
minus = wust_oracle(1./(1./initial-step*direction), obs, 1, cfg);
finite_difference = (plus-minus)/(2*step);
adjoint = sum(gradient(:).*direction(:));
error = abs(finite_difference-adjoint)/max(abs(finite_difference),abs(adjoint));
assert(error < tolerance, 'Discrete gradient failure: %.6g', error);
fprintf('%s %s gradient relative error %.6g\n',cfg.backend,cfg.wavenumber,error);
end

function reject(call, identifier)
rejected = false;
try
    call();
catch err
    rejected = strcmp(err.identifier, identifier);
end
assert(rejected, 'Expected %s', identifier);
end

function check_mex_boundary(backend)
if strcmp(backend,'gpu')
    rhs=gpuArray(complex(zeros(2,2,'single')));
    diagonal=gpuArray(complex(zeros(2,1,'single')));
    offdiagonal=gpuArray(complex(zeros(1,1,'single')));
    inverse=gpuArray(complex(zeros(2,2,2,'single')));
    for bad={real(rhs),gpuArray(complex(zeros(2,2,2,2,'single')))}
        rejected=false;
        try
            unused=applyBlockLU(bad{1},diagonal,offdiagonal,offdiagonal, ...
                diagonal,offdiagonal,offdiagonal,inverse,false); %#ok<NASGU>
        catch err
            rejected=strcmp(err.identifier,'WUST:InvalidInput');
        end
        assert(rejected,'MEX must reject real storage and rank-4 input');
    end
    for bad={real(diagonal),gpuArray(complex(zeros(2,1,2,'single')))}
        rejected=false;
        try
            unused=decompBlockLU(bad{1},offdiagonal,offdiagonal,rhs, ...
                diagonal,diagonal,diagonal,offdiagonal,offdiagonal); %#ok<NASGU>
        catch err
            rejected=strcmp(err.identifier,'WUST:InvalidInput');
        end
        assert(rejected,'MEX decomposition must validate storage before access');
    end
end
end
