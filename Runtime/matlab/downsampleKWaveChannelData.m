function [downsampled, savedTime, metadata] = downsampleKWaveChannelData( ...
    orderedSensorByTime, originalTime, downsampleFactor, method, varargin)
%DOWNSAMPLEKWAVECHANNELDATA Downsample k-Wave channels with an auditable policy.
% orderedSensorByTime is [receiver, time].  The returned data are
% [saved_time, receiver], matching the archived full_dataset convention.

parser = inputParser;
parser.addParameter('filterHalfLength', 24, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 1);
parser.addParameter('kaiserBeta', 8.6, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 0);
parser.parse(varargin{:}); opts = parser.Results;

orderedSensorByTime = single(orderedSensorByTime);
originalTime = double(originalTime(:).');
downsampleFactor = round(double(downsampleFactor));
method = lower(char(string(method)));
if downsampleFactor < 1
    error('downsampleFactor must be a positive integer.');
end
if size(orderedSensorByTime, 2) ~= numel(originalTime)
    error('Channel time dimension does not match originalTime.');
end

switch method
    case {'stride', 'direct'}
        downsampled = orderedSensorByTime(:, 1:downsampleFactor:end).';
        method = 'stride';
    case {'polyphase', 'fir-polyphase', 'filtered'}
        % resample applies a zero-phase-compensated Kaiser-windowed FIR
        % anti-alias filter before decimation.  A longer-than-default
        % half-length is used so the policy is explicit and reproducible.
        % R2021b's Signal Processing Toolbox only accepts double input in
        % resample.  Keep the archived channel-data contract as single while
        % doing the short anti-alias filtering calculation in double.
        downsampled = single(resample(double(orderedSensorByTime.'), ...
            1, downsampleFactor, round(double(opts.filterHalfLength)), ...
            double(opts.kaiserBeta)));
        method = 'fir-polyphase';
    otherwise
        error('Unknown resampling method: %s.', method);
end

dt = median(diff(originalTime));
savedTime = originalTime(1) + (0:size(downsampled, 1)-1) * ...
    (downsampleFactor * dt);
metadata = struct('method', method, ...
    'downsample_factor', downsampleFactor, ...
    'filter_half_length', round(double(opts.filterHalfLength)), ...
    'kaiser_beta', double(opts.kaiserBeta), ...
    'original_sample_rate_hz', 1 / dt, ...
    'saved_sample_rate_hz', 1 / (downsampleFactor * dt), ...
    'nyquist_hz', 1 / (2 * downsampleFactor * dt));
end
