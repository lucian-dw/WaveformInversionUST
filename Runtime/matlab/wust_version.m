function value = wust_version
% One version source shared with the Python launcher.
root = fileparts(fileparts(mfilename('fullpath')));
value = strtrim(fileread(fullfile(root, 'VERSION')));
assert(~isempty(regexp(value, '^\d+\.\d+\.\d+([+-][A-Za-z0-9.-]+)?$', 'once')), ...
    'Invalid Runtime/VERSION');
end
