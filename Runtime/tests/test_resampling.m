function test_resampling
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))),'matlab'));
fs=10.584e6;t=(0:4095)/fs;low=sin(2*pi*1.25e6*t+.31);high=sin(2*pi*3.5e6*t-.27);
[out,tt]=downsampleKWaveChannelData([low;high],t,2,'polyphase');
ref=[low(1:2:end).',high(1:2:end).'];
tone=@(a,f)abs(sum(double(a).*exp(-2i*pi*f*tt(:))))/numel(tt);
pass=tone(out(:,1),1.25e6)/tone(ref(:,1),1.25e6);
stop=tone(out(:,2),fs/2-3.5e6)/tone(ref(:,2),fs/2-3.5e6);
assert(abs(pass-1)<.02&&stop<1e-3);fprintf('FIR pass ratio %.6g, alias %.2f dB\n',pass,20*log10(stop));
end
