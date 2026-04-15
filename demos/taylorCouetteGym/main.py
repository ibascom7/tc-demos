from taylor_couette_mixing.envs.helpers import Helpers

omega = 5  # rad/sec
time_step = 1 # seconds

helpers = Helpers("taylor_couette_mixing/cases/tc_mixing_case")
results = helpers.do_simulation(omega, time_step)

print(results[0])
print(results[0].keys())