function yes=wustUseGPU
% Explicit backend instead of availability-dependent numerical behavior.
yes=strcmp(getenv('WUST_BACKEND'),'gpu');
end
