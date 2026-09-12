function wust_run(requestPath)
% Stable batch entry: JSON paths/options + MAT arrays, no benchlab imports.
req=jsondecode(fileread(requestPath));assert(strcmp(req.schema,'wfi.request.v1'),'Unknown request schema');
assert(~exist(req.output_mat,'file'),'Refusing to overwrite an existing result');
input=load(req.input_mat);timer=tic;
switch req.operation
    case 'simulate'
        result=wust_simulate(input.model,req.config);
    case 'prepare'
        % Larger masks/grids live in the MAT file; JSON remains a small adapter message.
        result=wust_prepare(input.rf,input.prepare_config);
    case 'reconstruct'
        cfg=req.config;cfg.update_mask=input.update_mask;
        result=wust_reconstruct(input.obs,input.initial_mps,cfg);
    otherwise,error('Unsupported operation');
end
result.runtime_wall_seconds=toc(timer);
result.matlab_version=version;
result.runtime_version=wust_version;
parent=fileparts(req.output_mat);if isempty(parent),parent=pwd;end
assert(exist(parent,'dir')==7,'Output parent must already exist');
tmp=[tempname(parent) '.mat'];save(tmp,'result','-v7.3');movefile(tmp,req.output_mat);
end
