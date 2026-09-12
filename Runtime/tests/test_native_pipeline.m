function test_native_pipeline(binaryPath,outputDir)
% Full 128 TX end-to-end gate. Run only on an explicitly assigned idle GPU.
% binaryPath must be the freshly built runtime's launcher directory.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));
assert(~exist(outputDir,'dir'),'Use a fresh output directory');mkdir(outputDir);
x=(-63:64)*.5e-3;y=x;[X,Y]=meshgrid(x,y);
c=1500+30*exp(-((X-.003).^2+(Y+.004).^2)/1e-5)-15*exp(-((X+.008).^2+(Y-.007).^2)/8e-6);
theta=(0:127)*2*pi/128;cx=round(64+.024*cos(theta)/.0005);cy=round(64+.024*sin(theta)/.0005);
indices=sub2ind(size(c),cy,cx).';assert(numel(unique(indices))==128);
% Intentionally unsorted order exposes TX/RX order mistakes.
indices=indices([65:128,1:64]);dt=70e-9;t=(0:1599)*dt;
model=struct('c_mps',c,'x_m',x,'y_m',y,'tx_index',indices,'rx_index',indices, ...
    'dt_s',dt,'source_pressure',sin(2*pi*.5e6*t).*exp(-((t-4e-6)/1.5e-6).^2));
cfg=struct('backend','native128','pml_size',10,'density_kg_m3',1000, ...
    'sound_speed_ref_mps',1550,'downsample_factor',2,'time_offset_s',-4e-6, ...
    'work_dir',outputDir,'binary_path',binaryPath,'device_num',0);
native=wust_simulate(model,cfg);
selected=[1,2,17,33,65,81,97,128];
save(fullfile(outputDir,'native_checkpoint.mat'),'native','model','cfg','-v7.3');
referenceModel=model;referenceModel.tx_index=model.tx_index(selected);
cfg.backend='matlab-serial';serial=wust_simulate(referenceModel,cfg);
assert(isequal(native.tx_xy_m(selected,:),serial.tx_xy_m)&&isequal(native.rx_xy_m,serial.rx_xy_m));
subset=native.pressure(:,:,selected);
assert(isequal(size(subset),size(serial.pressure)));
assert(max(abs(native.time_s-serial.time_s))<1e-15);
rferr=norm(double(subset(:))-double(serial.pressure(:)))/norm(double(serial.pressure(:)));
prep=struct('frequencies_hz',(.3:.025:1)*1e6,'x_m',x,'y_m',y,'c_geom_mps',1500, ...
    'window','none','phase_correction','none','mask',~eye(128));
prep.data_units='Pa*s';prep.measurement_provenance='self_simulated';
dn=wust_prepare(native,prep);prep.mask=prep.mask(selected,:);ds=wust_prepare(serial,prep);
subset=dn.Y(selected,:,:);derr=norm(double(subset(:))-double(ds.Y(:)))/norm(double(ds.Y(:)));
report=struct('schema','wust.native_pipeline_test','schema_version',1,'grid',[128,128],'transmitters',128, ...
    'rf_relative_l2',rferr,'frequency_relative_l2',derr,'passed',rferr<1e-4&&derr<1e-4, ...
    'native_generation_seconds',native.simulation.generation_seconds, ...
    'matlab_serial_generation_seconds',serial.simulation.generation_seconds, ...
    'native_filter_seconds',native.simulation.filter_seconds, ...
    'serial_filter_seconds',serial.simulation.filter_seconds, ...
    'matlab_reference_tx_indices',selected, ...
    'note','128 native TX vs eight independent MATLAB TX; separate CUDA serial gate covers all 128; no extrapolated speedup');
save(fullfile(outputDir,'result.mat'),'report','native','serial','-v7.3');
fid=fopen(fullfile(outputDir,'report.json'),'w');fprintf(fid,'%s',jsonencode(report));fclose(fid);
disp(report);assert(report.passed,'Native wrapper RF/DTFT mismatch');
end
