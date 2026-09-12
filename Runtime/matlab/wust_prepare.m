function out=wust_prepare(rf, cfg)
% RF: time_s [T,1], pressure [T,RX,TX], tx_xy_m/rx_xy_m [N,2].
% Never crops/resizes SoS. Caller owns acquisition selection and spatial grid.
% DTFT uses physical t (including pulse offset), not an FFT-bin approximation.
required={'frequencies_hz','x_m','y_m','c_geom_mps','window','phase_correction','mask', ...
    'data_units','measurement_provenance'};
for i=1:numel(required),assert(isfield(cfg,required{i}),['Missing ' required{i}]);end
t=double(rf.time_s(:)); dt=mean(diff(t)); f=double(cfg.frequencies_hz(:));
assert(numel(t)>1 && dt>0 && max(abs(diff(t)-dt))<dt*1e-6,'Nonuniform time');
assert(all(isfinite(f)&f>0&f<0.5/dt),'Frequency outside saved Nyquist');
assert(all(diff(f)>0),'Frequencies must increase');
assert(size(rf.pressure,1)==numel(t) && size(rf.pressure,2)==size(rf.rx_xy_m,1) ...
    && size(rf.pressure,3)==size(rf.tx_xy_m,1),'RF axes must be T,RX,TX');
nt=size(rf.tx_xy_m,1); nr=size(rf.rx_xy_m,1);
assert(isreal(rf.pressure),'WUST:InvalidInput','RF must be real time-domain pressure');
assert(all(cfg.mask(:)==0|cfg.mask(:)==1),'WUST:InvalidInput','Mask must contain 0/1');
mask=logical(cfg.mask);
if isequal(size(mask),[nt,nr]),mask=repmat(mask,1,1,numel(f));end
assert(isequal([size(mask,1),size(mask,2),size(mask,3)],[nt,nr,numel(f)]), ...
    'WUST:InvalidInput','RF mask shape mismatch');
grid=struct('shape_yx',[numel(cfg.y_m),numel(cfg.x_m)], ...
    'origin_yx_m',[cfg.y_m(1)-mean(diff(cfg.y_m))/2,cfg.x_m(1)-mean(diff(cfg.x_m))/2], ...
    'spacing_yx_m',[mean(diff(cfg.y_m)),mean(diff(cfg.x_m))],'origin_kind','pixel_edge');
geo=wust_geometry(grid,rf.tx_xy_m,rf.rx_xy_m);
txsnap=geo.tx_snapped_xy_m;rxsnap=geo.rx_snapped_xy_m;
Y=complex(zeros(nt,nr,numel(f),'single'));
kernel=exp(-2i*pi*f*t.')*dt;
dist=pdistLocal(rf.tx_xy_m,rf.rx_xy_m); snapdist=pdistLocal(txsnap,rxsnap);
tof=dist/cfg.c_geom_mps;
assert(ismember(string(cfg.window),["none","legacy-nominal"]),'Unsupported window');
assert(ismember(string(cfg.phase_correction),["none","homogeneous-tof"]),'Unsupported phase policy');
for tx=1:nt
    traces=double(rf.pressure(:,:,tx));
    active=reshape(any(mask(tx,:,:),3),1,nr);traces(:,~active)=0;
    assert(all(isfinite(traces(:))),'Non-finite RF, supply cleaned RF explicitly');
    if strcmp(cfg.window,'legacy-nominal')
        width=max(0.05*max(tof(:)),dt);
        traces=traces.*exp(-0.5*(max(tof(tx,:)-t,0)/width).^2);
    end
    values=kernel*traces;
    if strcmp(cfg.phase_correction,'homogeneous-tof')
        values=values.*exp(-2i*pi*f*((snapdist(tx,:)-dist(tx,:))/cfg.c_geom_mps));
    end
    Y(tx,:,:)=permute(single(values),[3,2,1]);
end
assert(isequal(size(mask),size(Y)),'Mask must be TX,RX or TX,RX,F');
assert(any(mask(:)),'Empty observations');
input=struct('schema','wust.frequency_input','schema_version',1,'grid',grid, ...
    'pressure',Y,'mask',mask,'pressure_axes','tx,rx,frequency', ...
    'mask_axes','tx,rx,frequency','frequencies_hz',f, ...
    'tx_xy_m',rf.tx_xy_m,'rx_xy_m',rf.rx_xy_m,'fourier_sign',-1, ...
    'pressure_type','total_pressure','data_units',cfg.data_units, ...
    'spectrum_normalization','dtft_dt','measurement_provenance',cfg.measurement_provenance);
out=wust_ingest_frequency(input);
out.preparation.rf_config=cfg;out.preparation.phase_correction=cfg.phase_correction;
out.saved_dt_s=dt;
if isfield(rf,'simulation'),out.simulation=rf.simulation;end
end
function d=pdistLocal(a,b)
d=sqrt((a(:,1)-b(:,1).').^2+(a(:,2)-b(:,2).').^2);
end
