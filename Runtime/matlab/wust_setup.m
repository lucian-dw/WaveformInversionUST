function wust_setup(backend)
    % Install maintained paths and check the explicitly requested deployment.
    if nargin < 1
        backend = 'cpu';
    end
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'matlab'), fullfile(root, 'solver'), '-begin');
    assert(ismember(string(backend), ["cpu", "gpu"]), 'backend must be cpu or gpu');
    if strcmp(backend, 'gpu')
        assert(gpuDeviceCount > 0, 'No GPU is available');
        assert(exist('decompBlockLU', 'file') == 3 && exist('applyBlockLU', 'file') == 3, ...
               'Run wust_build_mex first; GPU requests never fall back silently.');
    end
end
