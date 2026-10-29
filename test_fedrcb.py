import numpy as np

from fedrcb.linear_algebra import (
    add_local_moment,
    client_state_from_reduced_statistics,
    initialise_local_moments,
    local_reduced_statistics,
)


def test_personalized_reconstruction_matches_joint_ridge():
    clients, shared, private = 4, 2, 1
    ridge = 0.7
    rng = np.random.default_rng(12)
    moments = initialise_local_moments(clients, shared, private)
    joint_dim = shared + clients * private
    joint_gram = ridge * np.eye(joint_dim)
    joint_response = np.zeros(joint_dim)
    for client in range(clients):
        for _ in range(12):
            xg = rng.normal(size=shared)
            xl = rng.normal(size=private)
            reward = float(rng.normal())
            add_local_moment(*moments, client=client, x_global=xg, x_local=xl, reward=reward)
            feature = np.zeros(joint_dim)
            feature[:shared] = xg
            feature[shared + client * private:shared + (client + 1) * private] = xl
            joint_gram += np.outer(feature, feature)
            joint_response += feature * reward
    reduced_gram, reduced_response = local_reduced_statistics(*moments, ridge)
    joint_inverse = np.linalg.inv(joint_gram)
    joint_estimate = joint_inverse @ joint_response
    for client in range(clients):
        state = client_state_from_reduced_statistics(
            reduced_gram.sum(axis=0), reduced_response.sum(axis=0),
            local_cross_design=moments[1][client], local_design=moments[2][client],
            local_response=moments[4][client], regularization=ridge,
        )
        indices = list(range(shared)) + [shared + client]
        assert np.allclose(state.theta, joint_estimate[indices], atol=1e-12)
        assert np.allclose(state.inverse_design, joint_inverse[np.ix_(indices, indices)], atol=1e-12)
