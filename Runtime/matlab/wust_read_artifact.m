function value = wust_read_artifact(path, expected)
    % Decode declared Python axes at the HDF5/MATLAB storage boundary, once.
    doc = jsondecode(fileread(path));
    wust_assert_schema(doc, expected);
    fields = wust_array_fields(doc.schema);
    names = fieldnames(doc.arrays);
    allowed = fieldnames(fields);
    required = allowed(~cellfun(@(n)fields.(n).optional, allowed));
    assert(all(ismember(required, names)) && all(ismember(names, allowed)), ...
           'WUST:InvalidInput', 'Missing or unexpected array fields');
    [~, name, ext] = fileparts(doc.arrays_file);
    assert(strcmp([name ext], doc.arrays_file), 'WUST:InvalidInput', 'Array file must be a sibling filename');
    file = fullfile(fileparts(path), doc.arrays_file);
    value = doc.metadata;
    value.schema = doc.schema;
    value.schema_version = doc.schema_version;
    for i = 1:numel(names)
        key = names{i};
        spec = doc.arrays.(key);
        field = fields.(key);
        units = field.units;
        if strcmp(units, 'data_units')
            units = value.data_units;
        end
        assert(any(strcmp(spec.axes, field.axes)) && strcmp(spec.units, units) && ...
               any(strcmp(spec.dtype, field.dtypes)), 'WUST:InvalidInput', 'Invalid axes/units/dtype: %s', key);
        shape = double(spec.shape(:).');
        assert(numel(shape) == numel(strsplit(spec.axes, ',')) && ...
               all(isfinite(shape) & shape >= 1 & shape == round(shape)), ...
               'WUST:InvalidInput', 'Invalid array shape');
        if startsWith(spec.dtype, 'complex')
            assert(strcmp(spec.real_dataset, ['/' key '_real']) && ...
                   strcmp(spec.imag_dataset, ['/' key '_imag']), 'WUST:InvalidInput', 'Invalid complex dataset names');
            if strcmp(spec.dtype, 'complex64')
                storage = 'single';
            else
                storage = 'double';
            end
            a = read(file, spec.real_dataset, shape, storage) + 1i * read(file, spec.imag_dataset, shape, storage);
        else
            assert(strcmp(spec.dataset, ['/' key]), 'WUST:InvalidInput', 'Invalid dataset name');
            types = struct('float32', 'single', 'float64', 'double', 'int64', 'int64', 'bool', 'uint8', 'uint8', 'uint8');
            a = read(file, spec.dataset, shape, types.(spec.dtype));
            if strcmp(spec.dtype, 'bool')
                assert(all(a(:) == 0 | a(:) == 1), 'WUST:InvalidInput', 'Nonbinary mask');
                a = logical(a);
            end
        end
        value.(key) = a;
        if ismember(key, {'pressure', 'mask'})
            value.([key '_axes']) = spec.axes;
        end
    end
end

function a = read(file, dataset, shape, storage)
    info = h5info(file, dataset);
    assert(isequal(double(info.Dataspace.Size(:).'), fliplr(shape)), ...
           'WUST:InvalidInput', 'HDF5 dimensions disagree with declared axes');
    a = h5read(file, dataset);
    assert(isa(a, storage), 'WUST:InvalidInput', 'HDF5 storage disagrees with declared dtype');
    if numel(shape) == 1
        a = reshape(a, shape(1), 1);
    else
        a = permute(reshape(a, fliplr(shape)), numel(shape):-1:1);
    end
end
