function wfi_setup(backend)
% Isolated runtime: no genpath, no project/DPS imports; original scripts remain unchanged.
if nargin<1, backend='cpu'; end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'solver'),'-begin');
if ~ismember(string(backend),["cpu","gpu"]),error('backend must be cpu or gpu');end
setenv('WFI_BACKEND',char(backend));
if strcmp(backend,'gpu')
    assert(gpuDeviceCount>0,'No GPU is available');
    assert(exist('decompBlockLU','file')==3 && exist('applyBlockLU','file')==3, ...
        'Run wfi_build_mex first; GPU requests never fall back silently.');
end
end
