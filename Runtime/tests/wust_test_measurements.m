function [obs, sound_speed, source] = wust_test_measurements(spec, backend, frequencies_hz, outputPath)
    % Small coherent fixture; production reconstruction never generates observations.
    if nargin < 2
        backend = 'cpu';
    end
    if nargin < 3 || isempty(frequencies_hz)
        frequencies_hz = spec.frequencies_hz;
    end
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'matlab'));
    wust_setup(backend);
    if ~isfield(spec, 'grid')
        dx = mean(diff(spec.x_m));
        dy = mean(diff(spec.y_m));
        spec.grid = struct('shape_yx', [numel(spec.y_m), numel(spec.x_m)], ...
                           'origin_yx_m', [spec.y_m(1) - dy / 2, spec.x_m(1) - dx / 2], ...
                           'spacing_yx_m', [dy, dx], 'origin_kind', 'pixel_edge');
        [row, col] = ind2sub(spec.grid.shape_yx, spec.tx_index);
        spec.tx_xy_m = [reshape(spec.x_m(col), [], 1), reshape(spec.y_m(row), [], 1)];
        [row, col] = ind2sub(spec.grid.shape_yx, spec.rx_index);
        spec.rx_xy_m = [reshape(spec.x_m(col), [], 1), reshape(spec.y_m(row), [], 1)];
    end
    geo = wust_geometry(spec.grid, spec.tx_xy_m, spec.rx_xy_m);
    [X, Y] = meshgrid(geo.x_m, geo.y_m);
    sound_speed = 1500 + 30 * exp(-(X.^2 + Y.^2) / 3e-5);
    n_tx = numel(geo.tx_index);
    n_rx = numel(geo.rx_index);
    source = zeros([size(sound_speed), n_tx]);
    for tx = 1:n_tx
        source(geo.tx_index(tx) + (tx - 1) * numel(sound_speed)) = 1;
    end
    if isfield(spec, 'Y')
        pressure = spec.Y;
    else
        pressure = complex(zeros(n_tx, n_rx, numel(frequencies_hz)));
        for fi = 1:numel(frequencies_hz)
            options = struct('backend', backend, 'stencil_bounds', [1400, 1700]);
            solver = HelmholtzSolver(geo.x_m, geo.y_m, sound_speed, ...
                                     zeros(size(sound_speed)), frequencies_hz(fi), -1, 10, .004, options);
            wave = solver.solve(source, false);
            if isa(wave, 'gpuArray')
                wave = gather(wave);
            end
            wave = reshape(wave, [], n_tx);
            pressure(:, :, fi) = wave(geo.rx_index, :).';
        end
    end
    if isfield(spec, 'mask')
        mask = spec.mask;
        if isfield(spec, 'mask_axes') && strcmp(spec.mask_axes, 'frequency,tx,rx')
            mask = permute(mask, [2, 3, 1]);
        end
    else
        mask = true(size(pressure));
    end
    input = struct('schema', 'wust.frequency_input', 'schema_version', 2, ...
                   'grid', spec.grid, 'tx_xy_m', geo.tx_xy_m, 'rx_xy_m', geo.rx_xy_m, ...
                   'frequencies_hz', frequencies_hz, 'pressure', pressure, ...
                   'pressure_axes', 'tx,rx,frequency', 'mask', mask, 'mask_axes', 'tx,rx,frequency', ...
                   'fourier_sign', -1, 'real_pressure', true, 'pressure_type', 'total_pressure', ...
                   'data_units', 'unknown', 'spectrum_normalization', 'unknown', ...
                   'measurement_provenance', 'coherent_Helmholtz_fixture');
    obs = wust_prepare_observations(input);
    if nargin >= 4 && ~isempty(outputPath)
        wust_write_artifact(outputPath, input);
    end
end
