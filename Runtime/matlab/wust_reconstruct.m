function result=wust_reconstruct(obs,initial,cfg)
% Frequency-continuation FWI. Block-LU is the linear solver, not the algorithm.
% Upstream-compatible clipped PR/FR NCG + linearized step in slowness.
% Fork fixes: exact mass-stencil derivative, explicit contracts, costs and mask.
[obs,cfg]=prepare_reconstruction(obs,initial,cfg);
wust_setup(cfg.backend);
c=double(initial);mask=logical(cfg.update_mask);assert(isequal(size(c),size(mask)),'Update mask mismatch');
assert(all(c(:)>=cfg.bounds_mps(1)&c(:)<=cfg.bounds_mps(2)),'Initial field outside bounds');
allTimer=tic;n=numel(cfg.schedule);
if cfg.save_history
    history=zeros([size(c),n+1],'single');history(:,:,1)=c;
end
records=struct([]);previous=zeros(size(c));direction=previous;last=0;
for step=1:n
    fi=cfg.schedule(step);stepTimer=tic;
    [loss,g,state]=wust_oracle(c,obs,fi,cfg);
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
    if wustUseGPU,wait(gpuDevice);end
    linearSeconds=toc(linearTimer);alpha=0;
    assert(isfinite(denom)&&denom>=0&&all(isfinite(direction(:))), ...
        'WUST:NumericalFailure','Nonfinite direction/curvature');
    assert(denom>0||all(direction(:)==0), 'WUST:NumericalFailure','Zero curvature with nonzero update direction');
    if denom>0,alpha=max(0,-sum(rawg(:).*direction(:))/denom);end
    candidate=1./(1./c+cfg.step_damping*alpha*double(direction));
    assert(all(isfinite(candidate(:))&candidate(:)>0), 'WUST:NumericalFailure', ...
        'Invalid candidate; refusing to silently replace it with the old model');
    oldSlowness=1./c;
    delta=max(-cfg.max_update_mps,min(cfg.max_update_mps,candidate-c));
    candidate=max(cfg.bounds_mps(1),min(cfg.bounds_mps(2),c+delta));candidate(~mask)=c(~mask);c=candidate;
    if cfg.save_history,history(:,:,step+1)=single(c);end
    record=struct('step',step,'frequency_hz',obs.frequencies_hz(fi), ...
        'loss_before_update',loss,'relative_residual_before_update',sqrt(2*loss/max(state.data_norm,eps)), ...
        'alpha',alpha,'beta',beta,'wall_seconds',toc(stepTimer), ...
        'assembly_factor_seconds',state.factor_seconds,'forward_seconds',state.forward_seconds, ...
        'adjoint_seconds',state.adjoint_seconds,'linearized_seconds',linearSeconds, ...
        'fresh_factorizations',1,'forward_rhs',state.forward_rhs, ...
        'adjoint_rhs',state.adjoint_rhs,'linearized_rhs',numel(obs.tx_index));
    record.relative_slowness_update=norm(1./c(mask)-oldSlowness(mask))/max(norm(oldSlowness(mask)),realmin);
    record.stage_id=fi;record.loss_model_step=step-1;
    if isempty(records),records=record;else,records(step)=record;end %#ok<AGROW>
    fprintf('FWI %d/%d f=%.3f MHz residual(pre)=%.6g wall=%.3fs\n', ...
        step,n,obs.frequencies_hz(fi)/1e6,records(step).relative_residual_before_update,records(step).wall_seconds);
end
reason='schedule_complete';category='completion';
if n==0,reason='zero_updates';category='budget';
elseif n<cfg.planned_schedule_length,reason='caller_truncated_schedule';category='budget';end
result=struct('schema','wust.reconstruction','schema_version',2,'c_mps',c, ...
    'records',records,'config',cfg,'wall_seconds',toc(allTimer), ...
    'selection','final (no target labels or best-GT selection)', ...
    'gradient_contract','exact discrete nine-point mass stencil');
if cfg.save_history,result.history_mps=history;end
result.executed_schedule=cfg.schedule;result.completed_updates=n;
result.planned_schedule_length=cfg.planned_schedule_length;
result.executed_frequencies_hz=obs.frequencies_hz(cfg.schedule);
result.canonical_frequencies_hz=obs.frequencies_hz;
result.observation_preparation=obs.preparation;
result.completion=struct('category',category,'reason',reason,'converged',false);
result.final_data_residual=[];
result.update_semantics=struct('optimization_variable','slowness', ...
    'update_variable','full_slowness','update_units','s/m','update_norm','l2', ...
    'update_norm_scope','declared_update_mask', ...
    'update_normalization','norm(s_new-s_old)/norm(s_old)', ...
    'iteration_unit','frequency_schedule_update','update_rtol_enabled',false);
end

function [obs,cfg] = prepare_reconstruction(obs,initial,cfg)
obs=wust_prepare_observations(obs);
required={'backend','bounds_mps','max_update_mps','step_damping', ...
    'source_batch_size','pml_strength','pml_m','stencil_bounds','wavenumber', ...
    'filter_cutoff','filter_order','update_mask'};
allowed=[required,{'dispersion','frequency_schedule_hz','max_updates','save_history'}];
assert(all(ismember(required,fieldnames(cfg))) && all(ismember(fieldnames(cfg),allowed)), ...
    'WUST:InvalidInput','Missing or unsupported reconstruction config fields');
for name={'pml_m','pml_strength','max_update_mps','step_damping','source_batch_size','filter_order'}
    v=cfg.(name{1});assert(isnumeric(v)&&isreal(v)&&isscalar(v)&&isfinite(v)&&v>0, ...
        'WUST:InvalidInput','Expected positive finite %s',name{1});
end
assert(cfg.source_batch_size==round(cfg.source_batch_size) && cfg.filter_order==round(cfg.filter_order) && ...
    isscalar(cfg.filter_cutoff)&&isfinite(cfg.filter_cutoff)&&cfg.filter_cutoff>=0, ...
    'WUST:InvalidInput','Invalid batch/filter controls');
for name={'bounds_mps','stencil_bounds'}
    b=cfg.(name{1});assert(numel(b)==2&&all(isfinite(b))&&b(1)>0&&b(2)>b(1), ...
        'WUST:InvalidInput','Invalid speed bounds');
end
assert(ismember(string(cfg.backend),["cpu","gpu"])&& ...
    ismember(string(cfg.wavenumber),["continuum","kwave-ldr9"]), ...
    'WUST:InvalidInput','Unsupported backend/wavenumber');
if strcmp(cfg.wavenumber,'kwave-ldr9')
    assert(isfield(cfg,'dispersion'), 'WUST:InvalidInput','LDR9 requires explicit simulation dispersion facts');
end
assert(isequal(size(initial),[numel(obs.y_m),numel(obs.x_m)]) && isreal(initial) && ...
    all(isfinite(initial(:))) && all(initial(:)>=cfg.bounds_mps(1)&initial(:)<=cfg.bounds_mps(2)), ...
    'WUST:InvalidInput','Initial map must match grid and speed bounds; no resize or scalar expansion');
assert(isequal(size(cfg.update_mask),size(initial)) && ...
    all(cfg.update_mask(:)==0|cfg.update_mask(:)==1), 'WUST:InvalidInput','Invalid update mask');
cfg.update_mask=logical(cfg.update_mask);
[X,Y]=meshgrid(obs.x_m,obs.y_m);tol=16*eps(max(abs([obs.x_m(:);obs.y_m(:)]))+1);
interior=X>min(obs.x_m)+cfg.pml_m-tol & X<max(obs.x_m)-cfg.pml_m+tol & ...
    Y>min(obs.y_m)+cfg.pml_m-tol & Y<max(obs.y_m)-cfg.pml_m+tol;
interior([1,end],:)=false;interior(:,[1,end])=false;
assert(all(~cfg.update_mask(:)|interior(:)), 'WUST:InvalidInput','Update mask enters PML/Dirichlet boundary');
if ~isfield(cfg,'frequency_schedule_hz')
    full_hz=obs.frequencies_hz;
else
    full_hz=cfg.frequency_schedule_hz(:);
end
assert(isnumeric(full_hz)&&isreal(full_hz)&&(isempty(full_hz)||isvector(full_hz))&&all(isfinite(full_hz)&full_hz>0), ...
    'WUST:InvalidInput','frequency_schedule_hz must contain finite positive Hz');
[matched,schedule]=ismember(full_hz,obs.frequencies_hz);
assert(all(matched),'WUST:InvalidInput','Schedule frequencies must exactly match observations');
cfg.planned_schedule_length=numel(schedule);
if isfield(cfg,'max_updates') && ~isempty(cfg.max_updates)
    n=cfg.max_updates;
    assert(isnumeric(n)&&isreal(n)&&isscalar(n)&&isfinite(n)&&n>=0&&n==round(n), ...
        'WUST:InvalidInput','max_updates must be a nonnegative integer');
    schedule=schedule(1:min(n,numel(schedule)));
end
cfg.schedule=schedule(:).';
if ~isfield(cfg,'save_history'),cfg.save_history=false;end
assert(islogical(cfg.save_history)&&isscalar(cfg.save_history), ...
    'WUST:InvalidInput','save_history must be boolean');
end
