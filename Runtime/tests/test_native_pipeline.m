function test_native_pipeline(binaryPath, outputDir, device)
% Small non-square RF/DTFT comparison against independent MATLAB propagation.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'matlab'));
assert(~exist(outputDir,'dir'),'Use a fresh output directory');
mkdir(outputDir);
x = (-13:13)*.5e-3;
y = (-11:11)*.5e-3;
[X,Y] = meshgrid(x,y);
c = 1500 + 20*exp(-((X-.001).^2+(Y+.0015).^2)/2e-6);
tx = sub2ind(size(c),[16,7,12],[8,19,21]).';
rx = sub2ind(size(c),[7,17,6,13],[17,11,9,20]).';
dt = 60e-9;
t = (0:399)*dt;
model = struct('c_mps',c,'x_m',x,'y_m',y,'tx_index',tx,'rx_index',rx, ...
    'dt_s',dt,'source_pressure',sin(2*pi*.35e6*t).*exp(-((t-3e-6)/1e-6).^2));
cfg = struct('backend','native-batch','pml_size',10,'density_kg_m3',1000, ...
    'sound_speed_ref_mps',1550,'downsample_factor',2,'time_offset_s',-3e-6, ...
    'work_dir',outputDir,'binary_path',binaryPath,'device_num',device);
native = wust_simulate(model,cfg);
cfg.backend = 'matlab-serial';
serial = wust_simulate(model,cfg);
assert(isequal(native.tx_xy_m,serial.tx_xy_m) && isequal(native.rx_xy_m,serial.rx_xy_m));
assert(isequal(size(native.pressure),size(serial.pressure)));
assert(max(abs(native.time_s-serial.time_s))<1e-15);
rferr = norm(double(native.pressure(:))-double(serial.pressure(:)))/norm(double(serial.pressure(:)));
grid = struct('shape_yx',size(c),'spacing_yx_m',[.0005,.0005], ...
    'origin_yx_m',[y(1),x(1)]-.00025,'origin_kind','pixel_edge');
prep = struct('frequencies_hz',[.2,.35,.45]*1e6,'grid',grid,'c_geom_mps',1500, ...
    'phase_correction','none','mask',true(3,4), ...
    'data_units','Pa*s','measurement_provenance','self_simulated');
dn = wust_prepare(native,prep);
ds = wust_prepare(serial,prep);
derr = norm(double(dn.Y(:))-double(ds.Y(:)))/norm(double(ds.Y(:)));
report = struct('grid',size(c),'transmitters',3,'receivers',4, ...
    'rf_relative_l2',rferr,'frequency_relative_l2',derr, ...
    'native_seconds',native.simulation.generation_seconds, ...
    'serial_seconds',serial.simulation.generation_seconds, ...
    'passed',rferr<1e-4 && derr<1e-4);
save(fullfile(outputDir,'result.mat'),'report','native','serial','-v7.3');
fid = fopen(fullfile(outputDir,'report.json'),'w');
cleanup = onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',jsonencode(report));
disp(report);
assert(report.passed,'Native wrapper RF/DTFT mismatch');
end
