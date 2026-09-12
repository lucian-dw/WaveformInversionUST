function [qEffective,dqDm,metadata] = kwaveLdr9WavenumberSquared( ...
    soundSpeed,frequencyHz,spacingM,b,d,e,timeStepS,referenceSpeedMps,modelReferenceSpeedMps)
%KWAVELDR9WAVENUMBERSQUARED Match the local k-Wave time dispersion with a
% fixed-coefficient nine-point Helmholtz stencil.  The angular objective is
% a frozen 8-point Gauss-Legendre average over [0,pi/4].

if nargin<9 || isempty(modelReferenceSpeedMps)
    modelReferenceSpeedMps=mean(double(soundSpeed(:)));
end
c=double(soundSpeed);f=double(frequencyHz);h=double(spacingM);
dt=double(timeStepS);cr=double(referenceSpeedMps);c0=double(modelReferenceSpeedMps);
if any(c(:)<=0)||f<=0||h<=0||dt<=0||cr<=0||c0<=0
    error('All physical inputs must be positive.');
end
omega=2*pi*f;
temporalSine=sin(omega*dt/2);
z=(cr./c).*temporalSine;
if any(abs(z(:))>=1)
    error('k-Wave local dispersion target is evanescent: max |z|=%g.',max(abs(z(:))));
end
K=(2*h/(cr*dt)).*asin(z);

nodes=[-0.9602898564975363 -0.7966664774136267 ...
    -0.5255324099163290 -0.1834346424956498 ...
     0.1834346424956498  0.5255324099163290 ...
     0.7966664774136267  0.9602898564975363];
weights=0.5*[0.1012285362903763 0.2223810344533745 ...
    0.3137066458778873 0.3626837833783620 ...
    0.3626837833783620 0.3137066458778873 ...
    0.2223810344533745 0.1012285362903763];
theta=(nodes+1)*(pi/8);
N=zeros(size(K));D=zeros(size(K));Np=zeros(size(K));Dp=zeros(size(K));
for j=1:numel(theta)
    ct=cos(theta(j));st=sin(theta(j));
    x=K.*ct;y=K.*st;
    cx=cos(x);cy=cos(y);sx=sin(x);sy=sin(y);
    lambda=-4*b+2*(2*b-1).*(cx+cy)+4*(1-b).*cx.*cy;
    mass=1-d-e+(d/2).*(cx+cy)+e.*cx.*cy;
    lambdaPrime=-2*(2*b-1).*(sx.*ct+sy.*st) ...
        -4*(1-b).*(sx.*ct.*cy+cx.*sy.*st);
    massPrime=-(d/2).*(sx.*ct+sy.*st) ...
        -e.*(sx.*ct.*cy+cx.*sy.*st);
    w=weights(j);
    N=N+w.*lambda.*mass;
    D=D+w.*mass.*mass;
    Np=Np+w.*(lambdaPrime.*mass+lambda.*massPrime);
    Dp=Dp+w.*(2*mass.*massPrime);
end
Q=-N./D;
Qp=-(Np.*D-N.*Dp)./(D.*D);
qEffective=Q./(h*h);
m=(c0./c).^2;
a=(cr/c0)*temporalSine;
dKdm=(h*a)./(cr*dt.*sqrt(m).*sqrt(1-z.*z));
dqDm=(Qp./(h*h)).*dKdm;
if any(~isfinite(qEffective(:))|qEffective(:)<=0)|any(~isfinite(dqDm(:)))
    error('LDR-9 produced invalid wavenumber or derivative.');
end
metadata=struct('mode','kwave-aware-ldr9','quadrature','gauss-legendre-8', ...
    'time_step_s',dt,'reference_speed_mps',cr, ...
    'model_reference_speed_mps',c0,'q_min',min(qEffective(:)), ...
    'q_max',max(qEffective(:)),'dqdm_min',min(dqDm(:)), ...
    'dqdm_max',max(dqDm(:)),'target_K_min',min(K(:)), ...
    'target_K_max',max(K(:)));
end
