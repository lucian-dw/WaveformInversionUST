function result = wust_reconstruct(obs, initial, cfg)
% Frequency-continuation FWI in slowness, using clipped PR/FR NCG.
% The linearized step uses the exact discrete mass-stencil derivative.
[obs, cfg] = prepare_reconstruction(obs, initial, cfg);
wust_setup(cfg.backend);
c = double(initial);
mask = cfg.update_mask;
allTimer = tic;
n = numel(cfg.schedule);
if cfg.save_history
    history = zeros([size(c), n + 1], 'single');
    history(:, :, 1) = c;
end
records = struct([]);
prev_z = zeros(size(c));
prev_gz = 0;
direction = prev_z;
last = 0;
for step = 1:n
    fi = cfg.schedule(step);
    stepTimer = tic;
    [loss, g, state] = wust_oracle(c, obs, fi, cfg);
    [roi_gradient, z] = wust_roi_gradient(g, mask, obs.x_m, obs.y_m, ...
        mean(initial(:)), obs.frequencies_hz(fi), cfg.filter_cutoff, cfg.filter_order);
    [direction, beta, gz] = wust_ncg_direction(roi_gradient, z, prev_z, ...
        prev_gz, direction, fi == last);
    prev_z = z;
    prev_gz = gz;
    last = fi;

    [curvature, linearSeconds] = directional_curvature( ...
        state, direction, obs.rx_index, obs.mask(:, :, fi));
    slope = sum(roi_gradient(:).*direction(:));
    alpha = linearized_step_length(slope, curvature);
    oldSlowness = 1./c;
    c = wust_project_slowness_update(c, double(direction), alpha, cfg);
    if cfg.save_history
        history(:, :, step + 1) = single(c);
    end
    absolute_residual = sqrt(2*loss);
    relative_residual = NaN;
    residual_status = 'zero_observation_energy';
    if state.data_norm > 0
        relative_residual = absolute_residual/sqrt(state.data_norm);
        residual_status = 'available';
    end
    record = struct('step', step, 'frequency_hz', obs.frequencies_hz(fi), ...
        'loss_before_update', loss, ...
        'absolute_residual_before_update', absolute_residual, ...
        'relative_residual_before_update', relative_residual, ...
        'relative_residual_status', residual_status, ...
        'alpha', alpha, 'beta', beta, 'wall_seconds', toc(stepTimer), ...
        'assembly_factor_seconds', state.factor_seconds, ...
        'assembly_seconds', state.solver.assembly_seconds, ...
        'factor_seconds', state.solver.factor_seconds, ...
        'forward_seconds', state.forward_seconds, ...
        'adjoint_seconds', state.adjoint_seconds, ...
        'linearized_seconds', linearSeconds, 'fresh_factorizations', 1, ...
        'forward_rhs', state.forward_rhs, 'adjoint_rhs', state.adjoint_rhs, ...
        'linearized_rhs', numel(obs.tx_index));
    record.relative_slowness_update = norm(1./c(mask) - oldSlowness(mask)) ...
        /max(norm(oldSlowness(mask)), realmin);
    record.stage_id = fi;
    record.loss_model_step = step - 1;
    if isempty(records)
        records = record;
    else
        records(step) = record; %#ok<AGROW>
    end
    fprintf('FWI %d/%d f=%.3f MHz residual(pre)=%.6g wall=%.3fs\n', ...
        step, n, obs.frequencies_hz(fi)/1e6, ...
        records(step).relative_residual_before_update, records(step).wall_seconds);
end

reason = 'schedule_complete';
category = 'completion';
if n == 0
    reason = 'zero_updates';
    category = 'budget';
elseif n < cfg.planned_schedule_length
    reason = 'caller_truncated_schedule';
    category = 'budget';
end
result = struct('schema', 'wust.reconstruction', 'schema_version', 2, ...
    'c_mps', c, 'records', records, 'config', cfg, ...
    'wall_seconds', toc(allTimer), ...
    'selection', 'final (no target labels or best-GT selection)', ...
    'gradient_contract', 'exact discrete nine-point mass stencil');
if cfg.save_history
    result.history_mps = history;
end
result.executed_schedule = cfg.schedule;
result.completed_updates = n;
result.planned_schedule_length = cfg.planned_schedule_length;
result.executed_frequencies_hz = obs.frequencies_hz(cfg.schedule);
result.canonical_frequencies_hz = obs.frequencies_hz;
result.observation_preparation = obs.preparation;
result.completion = struct('category', category, 'reason', reason, 'converged', false);
result.final_data_residual = [];
if cfg.evaluate_final_residual
    final_loss = 0;
    final_energy = 0;
    finalTimer = tic;
    for fi = 1:numel(obs.frequencies_hz)
        [value, ~, final_state] = wust_oracle(c, obs, fi, cfg);
        final_loss = final_loss + value;
        final_energy = final_energy + final_state.data_norm;
    end
    relative = NaN;
    status = 'zero_observation_energy';
    if final_energy > 0
        relative = sqrt(2*final_loss/final_energy);
        status = 'available';
    end
    result.final_data_residual = struct('absolute', sqrt(2*final_loss), ...
        'relative', relative, 'status', status, ...
        'oracle_calls', numel(obs.frequencies_hz), ...
        'forward_rhs', numel(obs.tx_index)*numel(obs.frequencies_hz), ...
        'adjoint_rhs', numel(obs.tx_index)*numel(obs.frequencies_hz), ...
        'wall_seconds', toc(finalTimer));
end
result.wall_seconds = toc(allTimer);
result.update_semantics = struct('optimization_variable', 'slowness', ...
    'update_variable', 'full_slowness', 'update_units', 's/m', 'update_norm', 'l2', ...
    'update_norm_scope', 'declared_update_mask', ...
    'update_normalization', 'norm(s_new-s_old)/norm(s_old)', ...
    'iteration_unit', 'frequency_schedule_update', 'update_rtol_enabled', false);
end

function [obs, cfg] = prepare_reconstruction(obs, initial, cfg)
obs = wust_prepare_observations(obs);
required = {'backend', 'bounds_mps', 'max_update_mps', 'step_damping', ...
    'source_batch_size', 'pml_strength', 'pml_m', 'stencil_bounds', 'wavenumber', ...
    'filter_cutoff', 'filter_order', 'update_mask'};
allowed = [required, {'dispersion', 'frequency_schedule_hz', 'max_updates', 'save_history', 'evaluate_final_residual'}];
assert(all(ismember(required, fieldnames(cfg))) && all(ismember(fieldnames(cfg), allowed)), ...
    'WUST:InvalidInput', 'Missing or unsupported reconstruction config fields');
for name = {'pml_m', 'pml_strength', 'max_update_mps', 'step_damping', 'source_batch_size', 'filter_order'}
    v = cfg.(name{1});
    assert(isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v) && v > 0, ...
        'WUST:InvalidInput', 'Expected positive finite %s', name{1});
end
assert(cfg.source_batch_size == round(cfg.source_batch_size) ...
    && cfg.filter_order == round(cfg.filter_order) ...
    && isscalar(cfg.filter_cutoff) && isfinite(cfg.filter_cutoff) && cfg.filter_cutoff >= 0, ...
    'WUST:InvalidInput', 'Invalid batch/filter controls');
for name = {'bounds_mps', 'stencil_bounds'}
    b = cfg.(name{1});
    assert(numel(b) == 2 && all(isfinite(b)) && b(1) > 0 && b(2) > b(1), ...
        'WUST:InvalidInput', 'Invalid speed bounds');
end
assert(ismember(string(cfg.backend), ["cpu", "gpu"]) ...
    && ismember(string(cfg.wavenumber), ["continuum", "kwave-ldr9"]), ...
    'WUST:InvalidInput', 'Unsupported backend/wavenumber');
if strcmp(cfg.wavenumber, 'kwave-ldr9')
    assert(isfield(cfg, 'dispersion'), ...
        'WUST:InvalidInput', 'LDR9 requires explicit simulation dispersion facts');
end
assert(isequal(size(initial), [numel(obs.y_m), numel(obs.x_m)]) && isreal(initial) ...
    && all(isfinite(initial(:))) ...
    && all(initial(:) >= cfg.bounds_mps(1) & initial(:) <= cfg.bounds_mps(2)), ...
    'WUST:InvalidInput', 'Initial map must match grid and speed bounds; no resize or scalar expansion');
assert(isequal(size(cfg.update_mask), size(initial)) ...
    && all(cfg.update_mask(:) == 0 | cfg.update_mask(:) == 1), ...
    'WUST:InvalidInput', 'Invalid update mask');
cfg.update_mask = logical(cfg.update_mask);
[X, Y] = meshgrid(obs.x_m, obs.y_m);
tol = 16*eps(max(abs([obs.x_m(:); obs.y_m(:)])) + 1);
interior = X > min(obs.x_m) + cfg.pml_m - tol & X < max(obs.x_m) - cfg.pml_m + tol ...
    & Y > min(obs.y_m) + cfg.pml_m - tol & Y < max(obs.y_m) - cfg.pml_m + tol;
interior([1, end], :) = false;
interior(:, [1, end]) = false;
assert(all(~cfg.update_mask(:) | interior(:)), ...
    'WUST:InvalidInput', 'Update mask enters PML/Dirichlet boundary');

if ~isfield(cfg, 'frequency_schedule_hz')
    full_hz = obs.frequencies_hz;
else
    full_hz = cfg.frequency_schedule_hz(:);
end
assert(isnumeric(full_hz) && isreal(full_hz) ...
    && (isempty(full_hz) || isvector(full_hz)) && all(isfinite(full_hz) & full_hz > 0), ...
    'WUST:InvalidInput', 'frequency_schedule_hz must contain finite positive Hz');
[matched, schedule] = ismember(full_hz, obs.frequencies_hz);
assert(all(matched), 'WUST:InvalidInput', 'Schedule frequencies must exactly match observations');
cfg.planned_schedule_length = numel(schedule);
if isfield(cfg, 'max_updates') && ~isempty(cfg.max_updates)
    n = cfg.max_updates;
    assert(isnumeric(n) && isreal(n) && isscalar(n) && isfinite(n) && n >= 0 && n == round(n), ...
        'WUST:InvalidInput', 'max_updates must be a nonnegative integer');
    schedule = schedule(1:min(n, numel(schedule)));
end
cfg.schedule = schedule(:).';
counts = obs.valid_receiver_counts(:, unique(cfg.schedule));
assert(all(counts(:) >= 2), 'WUST:InvalidInput', ...
    'Source-scale fitting requires two valid receivers per scheduled TX/frequency');
if ~isfield(cfg, 'evaluate_final_residual')
    cfg.evaluate_final_residual = false;
end
assert(islogical(cfg.evaluate_final_residual) && isscalar(cfg.evaluate_final_residual), ...
    'WUST:InvalidInput', 'evaluate_final_residual must be boolean');
if ~isfield(cfg, 'save_history')
    cfg.save_history = false;
end
assert(islogical(cfg.save_history) && isscalar(cfg.save_history), ...
    'WUST:InvalidInput', 'save_history must be boolean');
end
