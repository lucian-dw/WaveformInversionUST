function simulate_and_reconstruct(output_dir)
% Small k-Wave -> anti-alias -> DTFT -> FWI example, not a calibrated benchmark.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'Runtime', 'matlab'));
assert(nargin == 1 && ~isempty(output_dir), 'Specify an output directory.');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
x = (-15:16)*.001;
y = x;
[X, Y] = meshgrid(x, y);
model = struct('x_m', x, 'y_m', y, ...
    'c_mps', 1500 + 15*exp(-(X.^2 + Y.^2)/1e-5), ...
    'dt_s', .15e-6, ...
    'source_pressure', sin(2*pi*.1e6*(0:399)*.15e-6).*exp(-(((0:399) - 30)/12).^2));
model.tx_index = sub2ind(size(X), [10, 24, 17], [8, 22, 25]).';
model.rx_index = model.tx_index;
cfg = struct('backend', 'matlab-serial', 'pml_size', 8, ...
    'density_kg_m3', 1000, 'sound_speed_ref_mps', 1550, ...
    'downsample_factor', 2, 'time_offset_s', 0, 'work_dir', output_dir);
rf = wust_simulate(model, cfg);
grid = struct('shape_yx', size(X), 'spacing_yx_m', [.001, .001], ...
    'origin_yx_m', [y(1), x(1)] - .0005, 'origin_kind', 'pixel_edge');
prepare = struct('grid', grid, 'frequencies_hz', [.1e6, .125e6], ...
    'c_geom_mps', 1500, 'phase_correction', 'none', 'mask', ~eye(3), ...
    'data_units', 'Pa*s', 'measurement_provenance', 'self_simulated');
obs = wust_prepare(rf, prepare);
assert(isequal(size(obs.Y), [3, 3, 2]));
assert(isequal(obs.tx_index, model.tx_index));
assert(isequal(obs.rx_index, model.rx_index));
mask = true(size(X));
mask([1:5, end-4:end], :) = false;
mask(:, [1:5, end-4:end]) = false;
fwi = struct('backend', 'cpu', 'frequency_schedule_hz', [1e5, 1.25e5], ...
    'bounds_mps', [1300, 1800], 'max_update_mps', 12, 'step_damping', .25, ...
    'source_batch_size', 2, 'pml_strength', 10, 'pml_m', .004, ...
    'stencil_bounds', [1400, 1700], 'wavenumber', 'continuum', ...
    'filter_cutoff', 0, 'filter_order', 4, 'update_mask', mask);
result = wust_reconstruct(obs, 1500*ones(size(X)), fwi);
assert(all(isfinite(result.c_mps(:))));
save(fullfile(output_dir, 'example.mat'), 'rf', 'obs', 'result', '-v7.3');
fprintf('Saved k-Wave -> FIR -> DTFT -> FWI example to %s\n', output_dir);
end
