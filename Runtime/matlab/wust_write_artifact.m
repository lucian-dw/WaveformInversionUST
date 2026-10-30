function wust_write_artifact(path, result)
    % Arrays close first; the manifest is the success index of this run directory.
    file = fullfile(fileparts(path), 'arrays.h5');
    assert(~isfile(file) && ~isfile(path), 'WUST:InvalidInput', 'Output exists');
    fields = wust_array_fields(result.schema);
    names = fieldnames(fields);
    meta = result;
    specs = struct;
    % Coordinates below are derived from grid/positions, never separate protocol fields.
    derived = {'x_m', 'y_m', 'tx_snapped_xy_m', 'rx_snapped_xy_m', 'pressure_axes', 'mask_axes'};
    for i = 1:numel(derived)
        if isfield(meta, derived{i})
            meta = rmfield(meta, derived{i});
        end
    end
    for i = 1:numel(names)
        key = names{i};
        field = fields.(key);
        if ~isfield(result, key)
            assert(field.optional, 'WUST:InvalidInput', 'Missing output array: %s', key);
            continue
        end
        a = result.(key);
        meta = rmfield(meta, key);
        axes = field.axes{1};
        if isfield(result, [key '_axes'])
            axes = result.([key '_axes']);
        end
        units = field.units;
        if strcmp(units, 'data_units')
            units = result.data_units;
        end
        rank = numel(strsplit(axes, ','));
        shape = zeros(1, rank);
        for k = 1:rank
            shape(k) = size(a, k);
        end
        if rank == 1
            shape = numel(a);
        end
        types = struct('double', 'float64', 'single', 'float32', 'logical', 'bool', 'int64', 'int64', 'uint8', 'uint8');
        dtype = types.(class(a));
        spec = struct('axes', axes, 'shape', {num2cell(shape)}, 'dtype', dtype, 'units', units);
        if all(startsWith(string(field.dtypes), 'complex'))
            if isa(a, 'single')
                spec.dtype = 'complex64';
            else
                spec.dtype = 'complex128';
            end
            spec.real_dataset = ['/' key '_real'];
            spec.imag_dataset = ['/' key '_imag'];
            write(file, spec.real_dataset, real(a), shape);
            write(file, spec.imag_dataset, imag(a), shape);
        else
            spec.dataset = ['/' key];
            write(file, spec.dataset, a, shape);
        end
        specs.(key) = spec;
    end
    if strcmp(result.schema, 'wust.reconstruction')
        meta.config = rmfield(meta.config, 'update_mask');
        meta.config.schedule = num2cell(meta.config.schedule(:).');
        for key = {'executed_schedule', 'executed_frequencies_hz', 'canonical_frequencies_hz'}
            meta.(key{1}) = num2cell(meta.(key{1})(:).');
        end
        meta.records = arrayfun(@(r)r, meta.records, 'UniformOutput', false);
    end
    meta = rmfield(meta, {'schema', 'schema_version'});
    doc = struct('schema', result.schema, 'schema_version', 2, 'metadata', meta, ...
                 'arrays_file', 'arrays.h5', 'arrays', specs);
    fid = fopen(path, 'w');
    assert(fid >= 0, 'WUST:InvalidInput', 'Cannot create output');
    close = onCleanup(@()fclose(fid));
    fprintf(fid, '%s', jsonencode(doc));
end

function write(file, name, a, shape)
    if islogical(a)
        a = uint8(a);
    end
    if numel(shape) > 1
        a = permute(a, numel(shape):-1:1);
    else
        a = a(:);
    end
    h5create(file, name, fliplr(shape), 'Datatype', class(a));
    h5write(file, name, a);
end
