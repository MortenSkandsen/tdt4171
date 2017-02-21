import numpy as np

# HIDDEN MARKOV MODEL


# Transition matrix, 2x2
# How the weather changes from day to day
m_transition = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float)
# Observation matrix, diagonal only, 2x2
# What are the odds of bossman carrying the umbrella if it is raining and vice versa
observations_true = np.array([[0.9, 0], [0, 0.2]], dtype=np.float)
observations_false = np.array([[0.1, 0], [0, 0.8]], dtype=np.float)
# Previous observations (e_t), this will be a list of bools
# True for item i if it rained on day i
observations = [None, True, True, False, True, True]

def normalize(vector):
    return np.array(vector/sum(vector))

def predict(prob_vect):
    # In: Probability vector with <p(r0 = true), p(r0 = false)>
    # Sum over both entries in the vector, multiply with P(Rt+1 | r0) (Transition)
    return m_transition.dot(prob_vect)

def update(prediction, day):
    # In: prediction vector
    # Updates the vector by calculating P(e_t+1| x_t+1)
    saw_umbrella = observations[day]
    obs_matrix = observations_false
    if saw_umbrella:
        obs_matrix = observations_true
    return normalize(obs_matrix.dot(prediction))

def normalize(vector):
    return np.array(vector/sum(vector))

def forward(cur_vec, day):
    cur_vec = predict(cur_vec)
    cur_vec = update(cur_vec, day)
    return cur_vec

def backward(cur_vec, day):
    # Returns the probability vector of P(e_{k+1:t}|X_k)
    # t is given by the length of observations
    # End day is the time slot k
    # The observation matrix is dependent on the evidence in the previous day
    obs_matrix = observations_false
    if observations[day-1]:
        obs_matrix = observations_true
    next_backward_msg = ( m_transition.dot(obs_matrix)).dot(cur_vec)
    return next_backward_msg

def forward_backward():
    # Smoothing returns the backwards probability of P(X_k | e_{1:t})
    t = len(observations)
    forward_vec = [None]*t
    smoothed_vec = [None]*t
    # Initial values for forward and backwards messages
    forward_vec[0] = [0.5, 0.5]
    back_msg = np.array([1,1])
    # Loop over all timeslots, first forwards, then backwards
    for i in range(1,t):
        forward_vec[i] = forward(forward_vec[i-1], i)
        print(forward_vec[i])
    for i in range(t-1,0,-1):
        smoothed_vec[i] = normalize(np.multiply(forward_vec[i], back_msg))
        back_msg = backward(back_msg, i)
    return smoothed_vec


forward_backward()
