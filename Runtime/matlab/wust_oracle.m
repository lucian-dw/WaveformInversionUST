function [loss,gradient,cache]=wust_oracle(c,obs,fi,cfg)
% Single-frequency source-projected least squares, exact discrete adjoint.
% Output gradient is w.r.t. SLOWNESS, not velocity; no filtering here.
% Per-TX complex scale a=(u'*d)/(u'*u); envelope theorem removes da/ds.
t=tic; x=obs.x_m;y=obs.y_m;[ny,nx]=size(c);nt=numel(obs.tx_index);
assert(isequal(size(c),[numel(y),numel(x)]) && all(isfinite(c(:))&c(:)>0),'Invalid medium grid');
assert(all(diff(x)>0) && all(diff(y)>0),'Grid must increase');
assert(max(abs(diff(x)-mean(diff(x))))<mean(diff(x))*1e-7 && ...
    max(abs(diff(y)-mean(diff(y))))<mean(diff(y))*1e-7,'Uniform grids required');
spec=[];
if strcmp(cfg.wavenumber,'kwave-ldr9')
    spec=cfg.dispersion;
else
    assert(strcmp(cfg.wavenumber,'continuum'),'Unknown wavenumber model');
end
solver=HelmholtzSolver(x,y,double(c),zeros(ny,nx),obs.frequencies_hz(fi), ...
    -1,cfg.pml_strength,cfg.pml_m,cfg.stencil_bounds,'inverse',spec);
sync; factorSeconds=toc(t);t=tic;
gradient=zeros(ny,nx);loss=0;normdata=0;virtual=cell(0);scales=cell(0);
batch=cfg.source_batch_size;
assert(batch>=1 && batch==round(batch),'Integer source_batch_size required');
forwardSeconds=0;adjointSeconds=0;
for first=1:batch:nt
    ids=first:min(first+batch-1,nt);nb=numel(ids);src=zeros(ny,nx,nb,'single');
    for j=1:nb,src(obs.tx_index(ids(j))+(j-1)*ny*nx)=1;end
    timer=tic;[u,v]=solver.solve(src,false);sync;forwardSeconds=forwardSeconds+toc(timer);
    u=reshape(u,ny*nx,nb);pred=u(obs.rx_index,:);
    mask=double(obs.mask(ids,:,fi).');data=double(obs.Y(ids,:,fi).');
    if wustUseGPU,mask=gpuArray(single(mask));data=gpuArray(single(data));end
    pred=pred.*mask;data=data.*mask;
    den=sum(abs(pred).^2,1);num=sum(conj(pred).*data,1);
    a=zeros(size(num),'like',num);good=den>0;a(good)=num(good)./den(good);
    r=(pred.*a-data).*mask;
    loss=loss+0.5*double(local(sum(abs(r).^2,'all')));
    normdata=normdata+double(local(sum(abs(data).^2,'all')));
    rhs=zeros(ny*nx,nb,'like',r);rhs(obs.rx_index,:)=r;
    timer=tic;adj=solver.solve(reshape(rhs,ny,nx,nb),true);
    adj=solver.massApply(adj,true);
    gradient=gradient+double(local(sum(-real(conj(v.*reshape(a,1,1,nb)).*adj),3)));
    sync;adjointSeconds=adjointSeconds+toc(timer);
    virtual{end+1}=v;scales{end+1}=a; %#ok<AGROW>
end
cache=struct('solver',solver,'virtual',{virtual},'scales',{scales}, ...
    'data_norm',normdata,'factor_seconds',factorSeconds, ...
    'forward_seconds',forwardSeconds,'adjoint_seconds',adjointSeconds, ...
    'oracle_seconds',factorSeconds+toc(t),'forward_rhs',nt,'adjoint_rhs',nt);
end
function a=local(a)
if isa(a,'gpuArray'),a=gather(a);end
end
function sync
if wustUseGPU,wait(gpuDevice);end
end
