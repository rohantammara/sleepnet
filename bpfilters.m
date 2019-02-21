clc;
clear;

%%% Prepare data %%%
load('psg_db/slp01am.mat');
data = val(3, 1:997500);
load('psg_db/slp01bm.mat');
data = [data val(3, 1:997500)];
load('psg_db/slp02am.mat');
data = [data val(3, 1:37500) val(3, 52500:224999) val(3, 232500:674999) val(3, 690000:704999) val(3, 712500:839999) val(3, 847500:884999) val(892500:997499)];
load('psg_db/slp02bm.mat');
data = [data val(3, 1:697500) val(3, 705000:997499)];
load('psg_db/slp03m.mat');
data = [data val(3, 1:997500)];
load('psg_db/slp04m.mat');
data = [data val(3, 1:997500)];
load('psg_db/slp14m.mat');
data = [data val(3, 45000:539999) val(3, 547500:997499)];
load('psg_db/slp16m.mat');
data = [data val(3, 195000:997499)];
load('psg_db/slp32m.mat');
data = [data val(3, 1:997500)];
load('psg_db/slp37m.mat');
data = [data val(3, 15000:997499)];
load('psg_db/slp41m.mat');
data = [data val(3, 1:997500)];
load('psg_db/slp45m.mat');
data = [data val(3, 1:997500)];
load('psg_db/slp48m.mat');
data = [data val(3, 1:465000)];
load('psg_db/slp01bm.mat');
data = [data val(3, 472500:997499)];
load('psg_db/slp59m.mat');
data = [data val(3, 165000:997499)];
load('psg_db/slp60m.mat');
data = [data val(3, 1:997500)];
load('psg_db/slp61m.mat');
data = [data val(3, 150000:997499)];
load('psg_db/slp66m.mat');
data = [data val(3, 1:997500)];
load('psg_db/slp67xm.mat');
data = [data val(3, 1:997500)];
clear val;
m = 2307;
data = reshape(data, [m, 7500]);
%%% Create filters and apply%%%
% Sampling frequency
fs = 250;
% Order of Butterworth Bandpass Filter 
order = 2;
% Delta
[b,a] = butter(order, [0.5 4]/(fs/2));
delta = zeros(m, 7500);
for i = 1:m
    delta(i, :) = filtfilt(b,a,data(i, :));
end
    % Theta
[b,a] = butter(order, [4 8]/(fs/2));
theta = zeros(m, 7500);
for i = 1:m
    theta(i, :) = filtfilt(b,a,data(i, :));
end
% Alpha
[b,a] = butter(order, [8 14]/(fs/2));
alpha = zeros(m, 7500);
for i = 1:m
    alpha(i, :) = filtfilt(b,a,data(i, :));
end
% Beta
[b,a] = butter(order, [14 30]/(fs/2));
beta = zeros(m, 7500);
for i = 1:m
    beta(i, :) = filtfilt(b,a,data(i, :));
end
    % Gamma
[b,a] = butter(order, [30 75]/(fs/2));
gamma = zeros(m, 7500);
for i = 1:m
    gamma(i, :) = filtfilt(b,a,data(i, :));
end
clear a b;
%{
data = reshape(data, [1, m*7500]);
delta = reshape(delta, [1, m*7500]);
theta = reshape(theta, [1, m*7500]);
alpha = reshape(alpha, [1, m*7500]);
beta = reshape(beta, [1, m*7500]);
gamma = reshape(gamma, [1, m*7500]);
%}

%%% Plot %%%
k = 3;
figure(1);
subplot(3,2,1);
plot(data(k, :), 'r');
title('EEG Signal (Main)')
subplot(3,2,2);
plot(delta(k, :));
title('Delta (0.5 - 4 Hz)')
subplot(3,2,3);
plot(theta(k, :));
title('Theta (4 - 8 Hz)')
subplot(3,2,4);
plot(alpha(k, :));
title('Alpha (8 - 14 Hz)')
subplot(3,2,5);
plot(beta(k, :));
title('Beta (14 - 30 Hz)')
subplot(3,2,6);
plot(gamma(k, :));
title('Gamma (30 - 75 Hz)');