import numpy as np
# How the weather changes from day to day
m_transition = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float)

# What are the odds of bossman carrying the umbrella if it is raining and vice versa
observations_true = np.array([[0.9, 0], [0, 0.2]], dtype=np.float)
observations_false = np.array([[0.1, 0], [0, 0.8]], dtype=np.float)
m_obs = [observations_true, observations_false]
"""
states = []
for each timeslot t:
    msg, state = calc_msg(msg, t) # Compute message according to (15.11), m_1:t
    states.add(state)
"""

def normalize(vector):
    return np.array(vector/sum(vector))


def calc_msg(prev_msg, day):

    # Calculate Ot+1*max(P(xt+1|xt)*m)
    obs_matrix = observations_false
    if observations[day+1]:
        obs_matrix = observations_true
    # See what value of Rain that maximises the probability
    values = []
    for i in range(2):
        values.append(max(m_transition[i]*(prev_msg)))
    #print(values)
    # Multiply with observation matrix
    prev_msg = obs_matrix.dot(values)
    print(prev_msg)
    state = prev_msg[0] > prev_msg[1]
    return state, prev_msg

observations = [None, True, True, False, True, True]
def viterbi(start_prob):
    print(start_prob)
    cur_msg = start_prob
    states = []
    t = len(observations)
    for i in range(t-1):
        state, cur_msg = calc_msg(cur_msg, i)
        states.append(state)
    print("Most likely occured events are: ", states)

viterbi([0.818, 0.182])
