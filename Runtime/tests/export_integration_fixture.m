function export_integration_fixture(inputPath,outputPath)
% A coherent asymmetric pressure fixture, not arbitrary sentinel pressure.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));wust_setup('cpu');
input=wust_read_artifact(inputPath,'wust.frequency_input');
geo=wust_geometry(input.grid,input.tx_xy_m,input.rx_xy_m);
[X,Y]=meshgrid(geo.x_m,geo.y_m);c=1500+10*exp(-((X-.003).^2+(Y+.002).^2)/2e-5);
nt=numel(geo.tx_index);src=zeros([size(c),nt]);
for i=1:nt,src(geo.tx_index(i)+(i-1)*numel(c))=1;end
p=complex(zeros(nt,numel(geo.rx_index),numel(input.frequencies_hz)));
for fi=1:numel(input.frequencies_hz)
    solver=HelmholtzSolver(geo.x_m,geo.y_m,c,zeros(size(c)),input.frequencies_hz(fi),-1,10,.003,[1400,1700]);
    u=solver.solve(src,false);u=reshape(u,[],nt);p(:,:,fi)=u(geo.rx_index,:).';
end
input.pressure=p;input.mask=permute(input.mask,[2,3,1]);
input.pressure_axes='tx,rx,frequency';input.mask_axes='tx,rx,frequency';input.fourier_sign=-1;
input.data_units='unknown';input.spectrum_normalization='unknown';
wust_write_artifact(outputPath,input);
end
