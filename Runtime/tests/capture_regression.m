function capture_regression(root, output)
backend='cpu';addpath(fullfile(root,'Runtime','matlab'));wust_setup(backend);
rng(7);x=(-15:15)*1e-3;y=x;[X,Y]=meshgrid(x,y);c=1500+30*exp(-(X.^2+Y.^2)/3e-5);
cfg=struct('backend',backend,'pml_m',.004,'pml_strength',10, ...
    'stencil_bounds',[1400,1700],'wavenumber','continuum','source_batch_size',2, ...
    'step_damping',.25,'max_update_mps',12,'bounds_mps',[1300,1800], ...
    'filter_cutoff',0,'filter_order',4,'schedule',[1,1,2,2],'update_mask',true(size(c)));
cfg.update_mask([1:5,end-4:end],:)=false;cfg.update_mask(:,[1:5,end-4:end])=false;
tx=sub2ind(size(c),[10,22,16],[8,23,25]).';rx=sub2ind(size(c),[8,15,23,18],[13,24,15,8]).';
obs=struct('x_m',x,'y_m',y,'tx_index',tx,'rx_index',rx, ...
    'frequencies_hz',[.1e6,.12e6],'fourier_sign',-1,'mask',true(3,4,2));
src=zeros([size(c),3]);for j=1:3,src(tx(j)+(j-1)*numel(c))=1;end
for k=1:2
    s=HelmholtzSolver(x,y,c,zeros(size(c)),obs.frequencies_hz(k),-1,10,.004,[1400,1700]);
    u=s.solve(src,false);if isa(u,'gpuArray'),u=gather(u);end;u=reshape(u,[],3);
    obs.Y(:,:,k)=u(rx,:).';
end
obs=wust_test_measurements(obs);
initial=1500*ones(size(c));
[loss,g]=wust_oracle(initial,obs,1,cfg);
initial_gradient=g;
direction=exp(-((X-.002).^2+(Y+.003).^2)/2e-5).*cfg.update_mask;direction=direction*1e-6;
h=1e-3;
if strcmp(backend,'gpu'),h=.1;end % complex-single LU needs a resolvable finite difference
lp=wust_oracle(1./(1./initial+h*direction),obs,1,cfg);
lm=wust_oracle(1./(1./initial-h*direction),obs,1,cfg);
fd=(lp-lm)/(2*h);ad=sum(g(:).*direction(:));err=abs(fd-ad)/max(abs(fd),abs(ad));
if strcmp(backend,'cpu'),tolerance=1e-4;else,tolerance=0.03;end
assert(err<tolerance,'Discrete gradient failure: %.6g',err);
fprintf('%s discrete projected gradient relative error %.6g\n',backend,err);
% Independent TX batching must not change loss or gradient.
cfg2=cfg;cfg2.source_batch_size=3;[l2,g2]=wust_oracle(initial,obs,1,cfg2);
assert(abs(l2-loss)/max(loss,eps)<1e-4 && norm(g2-g,'fro')/norm(g,'fro')<1e-4,'Batch mismatch');
result=wust_reconstruct(obs,initial,cfg);assert(all(isfinite(result.c_mps(:))));
final=wust_oracle(result.c_mps,obs,1,cfg);assert(final<loss,'Tiny training case did not improve');
fprintf('%s tiny FWI loss %.6g -> %.6g\n',backend,loss,final);
rf=struct('time_s',(0:255)'*1e-7+2e-7, ...
    'tx_xy_m',[x([8,23,25])',y([10,22,16])'], ...
    'rx_xy_m',[x([13,24,15,8])',y([8,15,23,18])']);
rf.pressure=randn(256,4,3);
prep=struct('frequencies_hz',obs.frequencies_hz,'x_m',x,'y_m',y, ...
    'c_geom_mps',1500,'window','none','phase_correction','none','mask',true(3,4));
prep.data_units='unknown';prep.measurement_provenance='numerical_fixture';
prepared=wust_prepare(rf,prep);
save(output,'prepared','loss','initial_gradient','result','-v7');
% Exact LDR9 derivative using explicit simulation metadata.
cfg.wavenumber='kwave-ldr9';cfg.dispersion=struct('time_step_s',2e-8, ...
    'reference_speed_mps',1700,'model_reference_speed_mps',1500);
[~,g]=wust_oracle(initial,obs,1,cfg);
lp=wust_oracle(1./(1./initial+h*direction),obs,1,cfg);
lm=wust_oracle(1./(1./initial-h*direction),obs,1,cfg);
fd=(lp-lm)/(2*h);ad=sum(g(:).*direction(:));err=abs(fd-ad)/max(abs(fd),abs(ad));
assert(err<tolerance,'LDR9 derivative failed');fprintf('%s LDR9 derivative error %.6g\n',backend,err);
end
