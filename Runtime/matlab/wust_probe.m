function facts = wust_probe(output)
facts=struct('matlab_available',true,'matlab_version',version,'platform',computer, ...
    'cpu_available',false,'gpu_available',false,'failure_reason','', 'mex',{{}});
try
    wust_setup('cpu');
    x=(-3:3)*.001;c=1500*ones(7);
    solver=HelmholtzSolver(x,x,c,zeros(7),1e5,-1,10,.001,[1400,1700]);
    q=zeros(7);q(4,4)=1;u=solver.solve(q,false);
    assert(all(isfinite(u(:))));facts.cpu_available=true;
catch err,facts.cpu_failure_reason=err.message;end
facts.gpu_device_available=false;
try
    facts.gpu_device_available=gpuDeviceCount>0;
    if facts.gpu_device_available
        g=gpuDevice;facts.gpu_device=g.Name;
        for field={'ComputeCapability','DriverVersion','ToolkitVersion'}
            if isprop(g,field{1}),facts.(field{1})=g.(field{1});end
        end
        wust_setup('gpu');
        solver=HelmholtzSolver(x,x,c,zeros(7),1e5,-1,10,.001,[1400,1700]);
        u=solver.solve(q,false);assert(all(isfinite(gather(u(:)))));
        facts.gpu_available=true;
    end
catch err,facts.gpu_failure_reason=err.message;end
for name={'decompBlockLU','applyBlockLU'}
    p=which(name{1});facts.mex{end+1}=struct('name',name{1},'path',p,'available',exist(name{1},'file')==3);
end
facts.simulation=struct('kwave_toolbox_available',exist('kWaveGrid','class')==8, ...
    'native_binary_available',false,'native_note','Tooling binary must be explicitly configured; not probed');
if nargin>0
    fid=fopen(output,'w');close=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(facts));
end
end
