function yes=wfiUseGPU
% Explicit backend instead of availability-dependent numerical behavior.
yes=strcmp(getenv('WFI_BACKEND'),'gpu');
end
