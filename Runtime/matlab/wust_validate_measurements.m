function obs = wust_validate_measurements(obs)
wust_assert_schema(obs,'wust.measurements');
geo = wust_geometry(obs.grid,obs.tx_xy_m,obs.rx_xy_m);
for name = {'tx_index','rx_index'}
    assert(isequal(obs.(name{1})(:),geo.(name{1})(:)), ...
        'WUST:InvalidInput','Canonical sampling indices do not match geometry');
end
for name = {'x_m','y_m'}
    a=obs.(name{1})(:);b=geo.(name{1})(:);
    assert(numel(a)==numel(b) && all(isfinite(a)) && ...
        max(abs(a-b))<=16*eps(max(abs(b))+1), 'WUST:InvalidInput','Grid coordinates disagree');
end
f=obs.frequencies_hz(:); shape=[numel(geo.tx_index),numel(geo.rx_index),numel(f)];
assert(~isempty(f)&&all(isfinite(f)&f>0)&&all(diff(f)>0)&&obs.fourier_sign==-1, ...
    'WUST:InvalidInput','Canonical frequencies/sign invalid');
assert(isequal([size(obs.Y,1),size(obs.Y,2),size(obs.Y,3)],shape) && ...
    isequal([size(obs.mask,1),size(obs.mask,2),size(obs.mask,3)],shape) && ...
    all(obs.mask(:)==0|obs.mask(:)==1), 'WUST:InvalidInput','Canonical arrays invalid');
obs.mask=logical(obs.mask);
assert(all(isfinite(obs.Y(obs.mask))), 'WUST:InvalidInput','Nonfinite valid pressure');
obs.Y(~obs.mask)=0;
end
