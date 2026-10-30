function obs = wust_prepare(rf, cfg)
    % RF [time,rx,tx] -> canonical complex pressure, with explicit Grid.
    % DTFT uses physical sample times and dt. No hidden window or normalization.
    required = {'grid', 'frequencies_hz', 'c_geom_mps', 'phase_correction', ...
                'mask', 'data_units', 'measurement_provenance'};
    assert(all(ismember(required, fieldnames(cfg))) && ...
           all(ismember(fieldnames(cfg), required)), ...
           'WUST:InvalidInput', 'Missing or unsupported RF preparation fields');
    time_s = double(rf.time_s(:));
    assert(numel(time_s) >= 2 && all(isfinite(time_s)), 'Invalid sample times');
    dt = time_s(2) - time_s(1);
    assert(dt > 0 && all(abs(diff(time_s) - dt) <= dt * 1e-6), 'Nonuniform time');
    frequency_hz = double(cfg.frequencies_hz(:));
    assert(all(isfinite(frequency_hz) & frequency_hz > 0 & frequency_hz < 0.5 / dt), ...
           'Frequency outside saved Nyquist');
    nt = size(rf.tx_xy_m, 1);
    nr = size(rf.rx_xy_m, 1);
    nf = numel(frequency_hz);
    assert(isreal(rf.pressure) && isequal( ...
                                          [size(rf.pressure, 1), size(rf.pressure, 2), size(rf.pressure, 3)], ...
                                          [numel(time_s), nr, nt]), 'RF must be real [time,rx,tx] pressure');
    assert(all(cfg.mask(:) == 0 | cfg.mask(:) == 1), 'Mask must contain 0/1');
    mask = logical(cfg.mask);
    if isequal(size(mask), [nt, nr])
        mask = repmat(mask, 1, 1, nf);
    end
    assert(isequal([size(mask, 1), size(mask, 2), size(mask, 3)], [nt, nr, nf]), ...
           'RF mask must be [tx,rx] or [tx,rx,frequency]');
    pressure = complex(zeros(nt, nr, nf, 'single'));
    kernel = exp(-2i * pi * frequency_hz * time_s.') * dt;
    for tx = 1:nt
        traces = double(rf.pressure(:, :, tx));
        active = reshape(any(mask(tx, :, :), 3), 1, nr);
        traces(:, ~active) = 0;
        assert(all(isfinite(traces(:))), 'Nonfinite RF on an active channel');
        pressure(tx, :, :) = permute(single(kernel * traces), [3, 2, 1]);
    end
    input = struct('schema', 'wust.frequency_input', 'schema_version', 2, ...
                   'grid', cfg.grid, 'pressure', pressure, 'mask', mask, ...
                   'pressure_axes', 'tx,rx,frequency', 'mask_axes', 'tx,rx,frequency', ...
                   'frequencies_hz', frequency_hz, 'tx_xy_m', rf.tx_xy_m, 'rx_xy_m', rf.rx_xy_m, ...
                   'fourier_sign', -1, 'pressure_type', 'total_pressure', ...
                   'data_units', cfg.data_units, 'spectrum_normalization', 'dtft_dt', ...
                   'measurement_provenance', cfg.measurement_provenance);
    obs = wust_prepare_observations(input);
    assert(ismember(string(cfg.phase_correction), ["none", "homogeneous-tof"]), ...
           'Unsupported phase policy');
    if strcmp(cfg.phase_correction, 'homogeneous-tof')
        validateattributes(cfg.c_geom_mps, {'numeric'}, {'scalar', 'positive', 'finite'});
        measured_distance = pair_distance(obs.tx_xy_m, obs.rx_xy_m);
        snapped_distance = pair_distance(obs.tx_snapped_xy_m, obs.rx_snapped_xy_m);
        delay_s = (snapped_distance - measured_distance) / cfg.c_geom_mps;
        obs.Y = obs.Y .* exp(-2i * pi * delay_s .* reshape(obs.frequencies_hz, 1, 1, []));
    end
    obs.preparation.rf_config = cfg;
    obs.preparation.phase_correction = cfg.phase_correction;
    obs.saved_dt_s = dt;
    if isfield(rf, 'simulation')
        obs.simulation = rf.simulation;
    end
end

function distance = pair_distance(a, b)
    distance = hypot(a(:, 1) - b(:, 1).', a(:, 2) - b(:, 2).');
end
