function value = wust_version
    % One version source shared with the Python launcher.
    root = fileparts(fileparts(mfilename('fullpath')));
    value = strtrim(fileread(fullfile(root, 'VERSION')));
end
