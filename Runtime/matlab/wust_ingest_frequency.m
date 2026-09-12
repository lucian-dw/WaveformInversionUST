function obs = wust_ingest_frequency(input)
% One owner for geometry, frequency order, masks and Fourier conversion.
wust_assert_schema(input, 'wust.frequency_input');
geo = wust_geometry(input.grid, input.tx_xy_m, input.rx_xy_m);
f = double(input.frequencies_hz(:));
assert(~isempty(f) && all(isfinite(f)&f>0) && numel(unique(f))==numel(f), ...
    'WUST:InvalidInput', 'Frequencies must be finite, positive and unique');
assert(strcmp(input.pressure_type,'total_pressure'), 'WUST:InvalidInput', ...
    'Ingestion requires total complex pressure, not scattered fields or ratios');
assert(ismember(string(input.data_units),["Pa*s","instrument_units","unknown"]) && ...
    ismember(string(input.spectrum_normalization),["dtft_dt","discrete_sum","unknown"]), ...
    'WUST:InvalidInput', 'Explicit supported units and normalization required');
assert(~strcmp(input.data_units,'Pa*s') || strcmp(input.spectrum_normalization,'dtft_dt'), ...
    'WUST:InvalidInput', 'Pa*s requires the dt-weighted DTFT normalization');
nt = numel(geo.tx_index); nr = numel(geo.rx_index); nf = numel(f);
Y = layout(input.pressure, input.pressure_axes);
assert((isa(Y,'single') || isa(Y,'double')) && ...
    isequal([size(Y,1),size(Y,2),size(Y,3)],[nt,nr,nf]), ...
    'WUST:InvalidInput', 'Pressure axes/shape mismatch');
if strcmp(input.mask_axes,'tx,rx')
    assert(isequal(size(input.mask),[nt,nr]), 'WUST:InvalidInput', 'Mask shape mismatch');
    mask = repmat(input.mask,1,1,nf);
else
    mask = layout(input.mask,input.mask_axes);
end
assert(isequal([size(mask,1),size(mask,2),size(mask,3)],[nt,nr,nf]) && ...
    all(mask(:)==0 | mask(:)==1), 'WUST:InvalidInput', 'Mask must contain only booleans/0/1');
mask = logical(mask);
assert(any(mask(:)) && all(isfinite(Y(mask))), 'WUST:InvalidInput', ...
    'Empty observations or nonfinite pressure on a valid channel');
Y(~mask) = 0;
conversion = 'none';
assert(isscalar(input.fourier_sign) && ismember(input.fourier_sign,[-1,1]), ...
    'WUST:InvalidInput', 'Declare Fourier sign -1 or +1; unknown conventions are rejected');
if input.fourier_sign==1
    assert(isfield(input,'real_pressure') && isequal(input.real_pressure,true), ...
        'WUST:InvalidInput', 'Conjugation requires declared real time-domain pressure');
    Y = conj(Y); conversion = 'conjugate';
end
[sorted, order] = sort(f); Y = Y(:,:,order); mask = mask(:,:,order);
obs = geo;
obs.schema = 'wust.measurements'; obs.schema_version = 1;
obs.Y = Y; obs.mask = mask; obs.frequencies_hz = sorted; obs.fourier_sign = -1;
obs.grid = input.grid; obs.data_units = input.data_units;
obs.spectrum_normalization = input.spectrum_normalization;
obs.pressure_type = 'total_pressure';
obs.preparation = struct('input_fourier_sign',input.fourier_sign, ...
    'conversion',conversion,'input_frequencies_hz',f, ...
    'canonical_to_input_permutation_1based',order, ...
    'input_pressure_axes',input.pressure_axes,'input_mask_axes',input.mask_axes, ...
    'phase_correction','none','masked_values_policy','zero_after_preserving_mask', ...
    'input_precision',class(Y),'measurement_provenance',input.measurement_provenance);
end

function a = layout(a, axes)
switch axes
    case 'tx,rx,frequency'
    case 'frequency,tx,rx'
        a = permute(a,[2,3,1]);
    otherwise
        error('WUST:InvalidInput','Unsupported declared axes: %s',axes);
end
end
