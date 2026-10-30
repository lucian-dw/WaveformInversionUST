function make_native_fixture(path, spatial_dimension)
    % Small homogeneous-water native input, including a genuine 3D z axis.
    if spatial_dimension == 2
        grid = kWaveGrid(12, 1e-3, 14, 1e-3);
        source.p_mask = zeros(12, 14);
        source.p_mask(5, 6) = 1;
        sensor.mask = zeros(12, 14);
        sensor.mask(8, 10) = 1;
        forward = @kspaceFirstOrder2D;
    elseif spatial_dimension == 3
        grid = kWaveGrid(12, 1e-3, 14, 1e-3, 10, 1e-3);
        source.p_mask = zeros(12, 14, 10);
        source.p_mask(5, 6, 4) = 1;
        sensor.mask = zeros(12, 14, 10);
        sensor.mask(8, 10, 7) = 1;
        forward = @kspaceFirstOrder3D;
    else
        error('WUST:InvalidInput', 'spatial_dimension must be 2 or 3');
    end
    grid.setTime(100, 1e-7);
    medium.sound_speed = 1500;
    medium.density = 1000;
    source.p = sin(2 * pi * 2e5 * grid.t_array) .* exp(-((grid.t_array - 2e-6) / 1e-6).^2);
    source.p_mode = 'additive';
    forward(grid, medium, source, sensor, 'PMLSize', 4, 'PMLInside', false, ...
            'PlotSim', false, 'SaveToDisk', path, 'SaveToDiskExit', true);
end
