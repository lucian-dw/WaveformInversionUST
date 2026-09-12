function result=wfi_reconstruct(obs,initial,cfg)
% Frequency-continuation FWI. Block-LU is the linear solver, not the algorithm.
% Upstream-compatible clipped PR/FR NCG + linearized step in slowness.
% Fork fixes: exact mass-stencil derivative, explicit contracts, costs and mask.
required={'backend','schedule','bounds_mps','max_update_mps','step_damping', ...
    'source_batch_size','pml_strength','pml_m','stencil_bounds','wavenumber', ...
    'filter_cutoff','filter_order','update_mask'};
for j=1:numel(required),assert(isfield(cfg,required{j}),['Missing ' required{j}]);end
wfi_setup(cfg.backend);
assert(strcmp(obs.schema,'wfi.measurements.v1') && obs.fourier_sign==-1,'Observation schema/sign mismatch');
assert(all(isfinite(cfg.schedule(:))) && all(cfg.schedule(:)==round(cfg.schedule(:))) ...
    && all(cfg.schedule(:)>=1 & cfg.schedule(:)<=numel(obs.frequencies_hz)),'Invalid frequency-index schedule');
assert(numel(cfg.bounds_mps)==2 && cfg.bounds_mps(1)>0 && diff(cfg.bounds_mps)>0,'Invalid bounds');
assert(cfg.step_damping>0 && cfg.max_update_mps>0,'Invalid update controls');
c=double(initial);mask=logical(cfg.update_mask);assert(isequal(size(c),size(mask)),'Update mask mismatch');
assert(all(c(:)>=cfg.bounds_mps(1)&c(:)<=cfg.bounds_mps(2)),'Initial field outside bounds');
allTimer=tic;n=numel(cfg.schedule);history=zeros([size(c),n+1],'single');history(:,:,1)=c;
records=struct([]);previous=zeros(size(c));direction=previous;last=0;
for step=1:n
    fi=cfg.schedule(step);stepTimer=tic;
    [loss,g,state]=wfi_oracle(c,obs,fi,cfg);
    rawg=g;
    if cfg.filter_cutoff>0
        g=ringingRemovalFilt(obs.x_m,obs.y_m,g,mean(initial(:)), ...
            obs.frequencies_hz(fi),cfg.filter_cutoff,cfg.filter_order);
    end
    g(~mask)=0;beta=0;den=sum(previous(:).^2);
    if fi==last && den>0
        beta=min(max(sum(g(:).*(g(:)-previous(:)))/den,0),sum(g(:).^2)/den);
    end
    direction=beta*direction-g;direction(~mask)=0;
    if sum(rawg(:).*direction(:))>=0,direction=-g;beta=0;end
    previous=g;last=fi;denom=0;offset=0;linearTimer=tic;
    for b=1:numel(state.virtual)
        v=state.virtual{b};a=state.scales{b};nb=numel(a);ids=offset+(1:nb);offset=offset+nb;
        source=state.solver.massApply(v.*direction,false);
        u=state.solver.solve(source,false);u=reshape(u,[],nb);
        derivative=-u(obs.rx_index,:).*a.*obs.mask(ids,:,fi).';
        val=sum(abs(derivative).^2,'all');if isa(val,'gpuArray'),val=gather(val);end
        denom=denom+double(val);
    end
    if wfiUseGPU,wait(gpuDevice);end
    linearSeconds=toc(linearTimer);alpha=0;
    if denom>0,alpha=max(0,-sum(rawg(:).*direction(:))/denom);end
    candidate=1./(1./c+cfg.step_damping*alpha*double(direction));
    candidate(~isfinite(candidate)|candidate<=0)=c(~isfinite(candidate)|candidate<=0);
    delta=max(-cfg.max_update_mps,min(cfg.max_update_mps,candidate-c));
    candidate=max(cfg.bounds_mps(1),min(cfg.bounds_mps(2),c+delta));candidate(~mask)=c(~mask);c=candidate;
    history(:,:,step+1)=single(c);
    record=struct('step',step,'frequency_hz',obs.frequencies_hz(fi), ...
        'loss_before_update',loss,'relative_residual_before_update',sqrt(2*loss/max(state.data_norm,eps)), ...
        'alpha',alpha,'beta',beta,'wall_seconds',toc(stepTimer), ...
        'assembly_factor_seconds',state.factor_seconds,'forward_seconds',state.forward_seconds, ...
        'adjoint_seconds',state.adjoint_seconds,'linearized_seconds',linearSeconds, ...
        'fresh_factorizations',1,'forward_rhs',state.forward_rhs, ...
        'adjoint_rhs',state.adjoint_rhs,'linearized_rhs',numel(obs.tx_index));
    if isempty(records),records=record;else,records(step)=record;end %#ok<AGROW>
    fprintf('FWI %d/%d f=%.3f MHz residual(pre)=%.6g wall=%.3fs\n', ...
        step,n,obs.frequencies_hz(fi)/1e6,records(step).relative_residual_before_update,records(step).wall_seconds);
end
result=struct('schema','wfi.reconstruction.v1','c_mps',c,'history_mps',history, ...
    'records',records,'config',cfg,'wall_seconds',toc(allTimer), ...
    'selection','final (no target labels or best-GT selection)', ...
    'gradient_contract','exact discrete nine-point mass stencil');
end
