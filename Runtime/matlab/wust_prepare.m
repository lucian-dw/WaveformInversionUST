function out=wust_prepare(rf, cfg)
% RF: time_s [T,1], pressure [T,RX,TX], tx_xy_m/rx_xy_m [N,2].
% Never crops/resizes SoS. Caller owns acquisition selection and spatial grid.
% DTFT uses physical t (including pulse offset), not an FFT-bin approximation.
required={'frequencies_hz','x_m','y_m','c_geom_mps','window','phase_correction','mask'};
for i=1:numel(required),assert(isfield(cfg,required{i}),['Missing ' required{i}]);end
t=double(rf.time_s(:)); dt=mean(diff(t)); f=double(cfg.frequencies_hz(:));
assert(numel(t)>1 && dt>0 && max(abs(diff(t)-dt))<dt*1e-6,'Nonuniform time');
assert(all(isfinite(f)&f>0&f<0.5/dt),'Frequency outside saved Nyquist');
assert(all(diff(f)>0),'Frequencies must increase');
assert(size(rf.pressure,1)==numel(t) && size(rf.pressure,2)==size(rf.rx_xy_m,1) ...
    && size(rf.pressure,3)==size(rf.tx_xy_m,1),'RF axes must be T,RX,TX');
nt=size(rf.tx_xy_m,1); nr=size(rf.rx_xy_m,1);
[txind,txsnap]=snap(rf.tx_xy_m,cfg.x_m,cfg.y_m);
[rxind,rxsnap]=snap(rf.rx_xy_m,cfg.x_m,cfg.y_m);
Y=complex(zeros(nt,nr,numel(f),'single'));
kernel=exp(-2i*pi*f*t.')*dt;
dist=pdistLocal(rf.tx_xy_m,rf.rx_xy_m); snapdist=pdistLocal(txsnap,rxsnap);
tof=dist/cfg.c_geom_mps;
assert(ismember(string(cfg.window),["none","legacy-nominal"]),'Unsupported window');
assert(ismember(string(cfg.phase_correction),["none","homogeneous-tof"]),'Unsupported phase policy');
for tx=1:nt
    traces=double(rf.pressure(:,:,tx));
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
mask=logical(cfg.mask);
if isequal(size(mask),[nt,nr]),mask=repmat(mask,1,1,numel(f));end
assert(isequal(size(mask),size(Y)),'Mask must be TX,RX or TX,RX,F');
assert(any(mask(:)),'Empty observations');
out=struct('schema','wfi.measurements.v1','Y',Y,'mask',mask,'frequencies_hz',f, ...
    'x_m',double(cfg.x_m(:).'),'y_m',double(cfg.y_m(:).'), ...
    'tx_index',txind,'rx_index',rxind,'tx_xy_m',rf.tx_xy_m,'rx_xy_m',rf.rx_xy_m, ...
    'tx_snapped_xy_m',txsnap,'rx_snapped_xy_m',rxsnap,'preparation',cfg, ...
    'saved_dt_s',dt,'fourier_sign',-1);
if isfield(rf,'simulation'),out.simulation=rf.simulation;end
end
function [index,xy]=snap(xy,x,y)
assert(size(xy,2)==2 && all(isfinite(xy(:))),'Expected physical [x,y]');
x=double(x(:));y=double(y(:));
assert(all(xy(:,1)>=min(x)&xy(:,1)<=max(x)&xy(:,2)>=min(y)&xy(:,2)<=max(y)),'Array outside grid');
[~,ix]=min(abs(x-xy(:,1).'),[],1);[~,iy]=min(abs(y-xy(:,2).'),[],1);
index=sub2ind([numel(y),numel(x)],iy,ix).';
assert(numel(unique(index))==numel(index),'Grid merges distinct array elements');
xy=[x(ix),y(iy)];
end
function d=pdistLocal(a,b)
d=sqrt((a(:,1)-b(:,1).').^2+(a(:,2)-b(:,2).').^2);
end
