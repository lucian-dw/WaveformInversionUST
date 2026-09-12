function smoke_pipeline
% Tiny external-k-Wave -> anti-alias -> DTFT -> FWI end-to-end example.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));
x=(-15:16)*.001;y=x;[X,Y]=meshgrid(x,y);
model=struct('x_m',x,'y_m',y,'c_mps',1500+15*exp(-(X.^2+Y.^2)/1e-5), ...
    'dt_s',.15e-6,'source_pressure',sin(2*pi*.1e6*(0:399)*.15e-6).*exp(-(((0:399)-30)/12).^2));
model.tx_index=sub2ind(size(X),[10,24,17],[8,22,25]).';model.rx_index=model.tx_index;
cfg=struct('backend','matlab-serial','pml_size',8,'density_kg_m3',1000, ...
    'sound_speed_ref_mps',1550,'downsample_factor',2,'time_offset_s',0,'work_dir',tempdir);
rf=wust_simulate(model,cfg);
prepare=struct('x_m',x,'y_m',y,'frequencies_hz',[.1e6,.125e6], ...
    'c_geom_mps',1500,'window','none','phase_correction','none','mask',~eye(3));
obs=wust_prepare(rf,prepare);
assert(isequal(size(obs.Y),[3,3,2]));
assert(isequal(obs.tx_index,model.tx_index));
assert(isequal(obs.rx_index,model.rx_index));
mask=true(size(X));mask([1:5,end-4:end],:)=false;mask(:,[1:5,end-4:end])=false;
fwi=struct('backend','cpu','schedule',[1,2],'bounds_mps',[1300,1800],'max_update_mps',12, ...
    'step_damping',.25,'source_batch_size',2,'pml_strength',10,'pml_m',.004, ...
    'stencil_bounds',[1400,1700],'wavenumber','continuum','filter_cutoff',0,'filter_order',4,'update_mask',mask);
result=wust_reconstruct(obs,1500*ones(size(X)),fwi);assert(all(isfinite(result.c_mps(:))));
out=fullfile(root,'tests','artifacts');if ~exist(out,'dir'),mkdir(out);end
save(fullfile(out,'smoke.mat'),'rf','obs','result','-v7.3');
% Exercise the thin adapter's MAT contract separately from direct MATLAB calls.
initial_mps=1500*ones(size(X));update_mask=mask;
save(fullfile(out,'adapter_input.mat'),'obs','initial_mps','update_mask');
req=struct('schema','wfi.request.v1','operation','reconstruct','input_mat', ...
    fullfile(out,'adapter_input.mat'),'output_mat',fullfile(out,'adapter_output.mat'),'config',rmfield(fwi,'update_mask'));
fid=fopen(fullfile(out,'request.json'),'w');fprintf(fid,'%s',jsonencode(req));fclose(fid);
fprintf('End-to-end k-Wave serial -> RF FIR -> DTFT -> FWI smoke passed\n');
end
