import numpy as np

# HIDDEN MARKOV MODEL


# Transition matrix, 2x2
m_transition = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float)
# Observation matrix, diagonal only, 2x2
observations_true = np.array([[0.9, 0], [0, 0.2]], dtype=np.float)
observations_false = np.array([[0.1, 0], [0, 0.8]], dtype=np.float)
# Previous observations (e_t), this will be a list of bools
# True for item i if it rained on day i
observations = [None, True, True, False, True, True]

def predict(prob_vect = np.array([0.5, 0.5])):
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

def forward(f0):
    day = 1
    # Start with a vector of equal probability
    cur_vec = np.array([0.5, 0.5])
    # For all days with evidences
    for i in range(len(observations)-1):
        cur_vec = predict(cur_vec)
        cur_vec = update(cur_vec, day)
        day += 1

def forward-backward():

    forward_vec = 0
    smoothed_vec = 0



print(cur_vec)
