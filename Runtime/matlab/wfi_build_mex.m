function wfi_build_mex
% Fork change: checked CUDA calls; compiler compatibility without admin installs.
root=fileparts(fileparts(mfilename('fullpath')));
old=pwd; cleanup=onCleanup(@()cd(old)); cd(fullfile(root,'solver'));
mexcuda('-R2018a','-lcusolver','-lcublas','-output','decompBlockLU','decompBlockLU.cu');
mexcuda('-R2018a','-lcublas','-output','applyBlockLU','applyBlockLU.cu');
end
