function test_batch_contract
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));wust_setup('cpu');
x=(-10:10)*.001;[X,Y]=meshgrid(x,x);c=1500+5*exp(-(X.^2+Y.^2)/1e-5);
obs=struct('x_m',x,'y_m',x,'tx_index',[6+21*5;15+21*14], ...
    'rx_index',[14+21*5;6+21*13;11+21*16],'frequencies_hz',[1e5,1.2e5], ...
    'fourier_sign',-1,'mask',true(2,3,2),'Y',complex(ones(2,3,2)));
obs=wust_test_measurements(obs);
src=zeros(21,21,2);for i=1:2,src(obs.tx_index(i)+(i-1)*numel(c))=1;end
for fi=1:2
    solver=HelmholtzSolver(x,x,c,zeros(size(c)),obs.frequencies_hz(fi),-1,10,.003,[1400,1700]);
    u=reshape(solver.solve(src,false),[],2);obs.Y(:,:,fi)=u(obs.rx_index,:).';
end
mask=false(21);mask(5:17,5:17)=true;
cfg=struct('backend','cpu','schedule',[1,1,2],'bounds_mps',[1300,1800],'max_update_mps',12, ...
    'step_damping',.25,'source_batch_size',1,'pml_strength',10,'pml_m',.003, ...
    'stencil_bounds',[1400,1700],'wavenumber','continuum','filter_cutoff',0,'filter_order',4,'update_mask',mask);
initial=1500*ones(21);r=wust_reconstruct(obs,initial,cfg);
assert(strcmp(r.completion.reason,'schedule_complete')&&~r.completion.converged&&r.completed_updates==3);
assert(isempty(r.final_data_residual)&&r.records(end).loss_model_step==2);
cfg0=cfg;cfg0.schedule=[];r0=wust_reconstruct(obs,initial,cfg0);
assert(isequal(r0.c_mps,initial)&&isempty(r0.records)&&r0.completed_updates==0);
short=cfg;short.schedule=1;short.planned_schedule_length=3;rs=wust_reconstruct(obs,initial,short);
assert(strcmp(rs.completion.reason,'caller_truncated_schedule'));
bad=obs;bad.mask(1,:,1)=false;bad.mask(1,1,1)=true;
reject(@()wust_reconstruct(bad,initial,cfg),'WUST:NumericalFailure');
bad=obs;bad.Y(1,1,1)=NaN;
reject(@()wust_reconstruct(bad,initial,cfg),'WUST:InvalidInput');
bad.mask(1,1,1)=false;finite=wust_reconstruct(bad,initial,cfg);assert(all(isfinite(finite.c_mps(:))));
naninit=initial;naninit(1)=NaN;reject(@()wust_reconstruct(obs,naninit,cfg),'WUST:InvalidInput');
badcfg=cfg;badcfg.update_rtol=1e-3;reject(@()wust_reconstruct(obs,initial,badcfg),'WUST:InvalidInput');
badcfg=cfg;badcfg.update_mask(1,1)=true;reject(@()wust_reconstruct(obs,initial,badcfg),'WUST:InvalidInput');
badcfg=cfg;badcfg.schedule=3;reject(@()wust_reconstruct(obs,initial,badcfg),'WUST:InvalidInput');
% Finite values can still overflow the objective; they must not yield success.
bad=obs;bad.Y(:)=realmax/2;reject(@()wust_reconstruct(bad,initial,cfg),'WUST:NumericalFailure');
fprintf('Completion/zero/truncation and seven invalid/numerical failure cases passed\n');
end

function reject(call,identifier)
failed=false;try,call();catch err,failed=strcmp(err.identifier,identifier);end
assert(failed,'Expected %s',identifier);
end
