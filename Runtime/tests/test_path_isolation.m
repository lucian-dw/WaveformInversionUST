function test_path_isolation
% Reference paths must not select historical solver code in production.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
oldPath = path; oldDir = pwd;
cleanup = onCleanup(@() restore(oldPath, oldDir));
addpath(fullfile(root, 'Runtime', 'matlab'));
historical = fullfile(root, 'reference', 'upstream-9fb31657', 'Functions');
addpath(historical, '-begin');
wust_setup('cpu');
assert(strcmp(which('HelmholtzSolver'), ...
    fullfile(root, 'Runtime', 'solver', 'HelmholtzSolver.m')));
assert(strcmp(wust_version, strtrim(fileread(fullfile(root, 'Runtime', 'VERSION')))));
cd(historical);
rejected = false;
try
    wust_setup('cpu');
catch err
    rejected = strcmp(err.identifier, 'WUST:PathConflict');
end
assert(rejected, 'Historical current-directory shadowing must fail closed');
fprintf('Maintained path selection and historical shadow rejection passed\n');
end

function restore(oldPath, oldDir)
cd(oldDir);
path(oldPath);
end
