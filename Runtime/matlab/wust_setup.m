function wust_setup(backend)
% Add only the maintained solver. Historical examples run in another process.
if nargin<1, backend='cpu'; end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'solver'),'-begin');
solver = fullfile(root, 'solver');
names = {'HelmholtzSolver', 'ringingRemovalFilt', 'stencilOptParams', ...
    'assembleBlockTridiagonalsGPU', 'kwaveLdr9WavenumberSquared', 'wustUseGPU'};
for i = 1:numel(names)
    assert(strcmp(which(names{i}), fullfile(solver, [names{i} '.m'])), ...
        'WUST:PathConflict', 'Unexpected solver resolution: %s', names{i});
end
if ~ismember(string(backend),["cpu","gpu"]),error('backend must be cpu or gpu');end
setenv('WUST_BACKEND',char(backend));
if strcmp(backend,'gpu')
    assert(gpuDeviceCount>0,'No GPU is available');
    assert(exist('decompBlockLU','file')==3 && exist('applyBlockLU','file')==3, ...
        'Run wust_build_mex first; GPU requests never fall back silently.');
    for name = {'decompBlockLU', 'applyBlockLU'}
        assert(strcmp(which(name{1}), fullfile(solver, [name{1} '.' mexext])), ...
            'WUST:PathConflict', 'GPU MEX must come from the maintained solver: %s', name{1});
    end
end
end
