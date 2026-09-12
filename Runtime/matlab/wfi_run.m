function wfi_run(requestPath)
% Stable batch entry: JSON paths/options + MAT arrays, no benchlab imports.
req=jsondecode(fileread(requestPath));assert(strcmp(req.schema,'wfi.request.v1'),'Unknown request schema');
assert(~exist(req.output_mat,'file'),'Refusing to overwrite an existing result');
input=load(req.input_mat);timer=tic;
switch req.operation
    case 'simulate'
        result=wfi_simulate(input.model,req.config);
    case 'prepare'
        % Larger masks/grids live in the MAT file; JSON remains a small adapter message.
        result=wfi_prepare(input.rf,input.prepare_config);
    case 'reconstruct'
        cfg=req.config;cfg.update_mask=input.update_mask;
        result=wfi_reconstruct(input.obs,input.initial_mps,cfg);
    otherwise,error('Unsupported operation');
end
result.runtime_wall_seconds=toc(timer);
result.matlab_version=version;
result.runtime_version='0.1.0';
parent=fileparts(req.output_mat);if isempty(parent),parent=pwd;end
assert(exist(parent,'dir')==7,'Output parent must already exist');
tmp=[tempname(parent) '.mat'];save(tmp,'result','-v7.3');movefile(tmp,req.output_mat);
end
