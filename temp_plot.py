from numpy import *
from matplotlib.pyplot import *
from scipy.stats import binned_statistic

figure()

subplot(311)
g = floor_divide(sim['spk'][1], 50)
tt = sim['spk'][0] - trunc(sim['spk'][0])
stim = sim['spk'][:, logical_and(tt < 0.02, g == 0)]
is_resp = logical_and(tt > 0.02, tt < 0.04)
resp1 = sim['spk'][:, logical_and(is_resp, g == 1)]
resp2 = sim['spk'][:, logical_and(is_resp, g == 2)]
plot(stim[0], stim[1], '.k')
plot(resp1[0], resp1[1], '.g')
plot(resp2[0], resp2[1], '.b')

subplot(312)
plot(sim['t'], 100 * sim['timer'], c='k', label='timer')
plot(sim['t'], sim['resp1'], c='g', label='resp1')
plot(sim['t'], sim['resp2'], c='b', label='resp2')
ylim((0, 10))

subplot(313)
plot(sim['t'], cumsum(sim['da']), c='g', label='da')

show()

figure()
#r1_indices = logical_and(sim['syn'][1] >= 50, sim['syn'][1] < 100)
#r2_indices = logical_and(sim['syn'][1] >= 100, sim['syn'][1] < 150)
pcolor(sim['syn'][0], sim['syn'][1], marker='s', c=(1 - sim['syn'][2]), edgecolor='none')
#plot([0,1],[0,1])
#gray()
xlim((0, 800))
ylim((0, 1000))

show()
