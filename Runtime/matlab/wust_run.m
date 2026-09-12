function wust_run(requestPath)
% Private batch dispatch after Python envelope/admission validation.
req=jsondecode(fileread(requestPath));wust_assert_schema(req,'wust.request');
assert(~isfile(req.output_manifest),'WUST:InvalidInput','Output exists');
timer=tic;
try
    switch req.operation
        case 'ingest_frequency'
            input=wust_read_artifact(req.input_manifest,'wust.frequency_input');
            result=wust_ingest_frequency(input);
        case 'prepare'
            rf=wust_read_artifact(req.input_manifest,'wust.rf');
            cfg=req.config;geo=wust_geometry(cfg.grid,rf.tx_xy_m,rf.rx_xy_m);
            cfg=rmfield(cfg,'grid');cfg.x_m=geo.x_m;cfg.y_m=geo.y_m;
            if strcmp(rf.data_units,'Pa'),cfg.data_units='Pa*s';else,cfg.data_units=rf.data_units;end
            cfg.measurement_provenance=rf.measurement_provenance;
            result=wust_prepare(rf,cfg);
        case 'reconstruct'
            obs=wust_read_artifact(req.input_manifest,'wust.measurements');
            initial=wust_read_artifact(req.initial_manifest,'wust.initial_model');
            cfg=req.config;cfg.schedule=req.schedule;cfg.update_mask=initial.update_mask;
            if isfield(req,'planned_schedule_length'),cfg.planned_schedule_length=req.planned_schedule_length;end
            result=wust_reconstruct(obs,initial.initial_mps,cfg);
            result.observation_preparation=obs.preparation;
        case 'simulate'
            model=wust_read_artifact(req.input_manifest,'wust.simulation_input');
            geo=wust_geometry(model.grid,model.tx_xy_m,model.rx_xy_m);
            model.x_m=geo.x_m;model.y_m=geo.y_m;model.tx_index=geo.tx_index;model.rx_index=geo.rx_index;
            result=wust_simulate(model,req.config);
        otherwise,error('WUST:InvalidInput','Unsupported operation');
    end
    result.runtime_wall_seconds=toc(timer);
    result.environment=struct('matlab_version',version,'platform',computer,'runtime_version',wust_version);
    if strcmp(req.operation,'reconstruct')
        result.environment.backend=req.config.backend;
        if strcmp(req.config.backend,'gpu')
            result.environment.precision_profile='gpu_complex64';g=gpuDevice;
            result.environment.gpu_device=g.Name;
            for name={'ComputeCapability','DriverVersion','ToolkitVersion'}
                if isprop(g,name{1}),result.environment.(name{1})=g.(name{1});end
            end
            result.environment.mex=struct('decompBlockLU',which('decompBlockLU'),'applyBlockLU',which('applyBlockLU'));
        else,result.environment.precision_profile='cpu_float64';end
    end
    wust_write_artifact(req.output_manifest,result);
catch err
    failure=struct('schema','wust.execution_failure','schema_version',1,'identifier',err.identifier,'message',err.message);
    fid=fopen(req.output_manifest,'w');
    if fid>=0,fprintf(fid,'%s',jsonencode(failure));fclose(fid);end
    rethrow(err);
end
end
