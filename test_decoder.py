"""Small exact checks for decoding, directed routing, and explicit route breaks."""
import itertools
import unittest

import numpy as np
from scipy.special import logsumexp

from decoder import RoadNetwork, chosen_arcs, route_sections, transition_geometry, viterbi


class DecoderTests(unittest.TestCase):
    def test_viterbi_matches_exhaustive_sequence_search(self):
        rng = np.random.default_rng(8)
        scores = rng.uniform(.05, .95, (4, 3))
        roads = rng.uniform(0, 100, (3, 3, 3))
        displacement = np.array([5., 20., 35.])
        geometry = dict(distances=roads, displacement=displacement, eligible=np.ones(3, bool))
        ids = np.tile(np.arange(3), (4, 1))
        states, breaks = viterbi(scores, ids, geometry, beta_m=15, strength=.3)
        emission = np.log(scores / scores.sum(1, keepdims=True))
        transition = -np.abs(roads-displacement[:, None, None])/15
        transition -= logsumexp(transition, axis=2, keepdims=True)

        def joint(path):
            return (sum(emission[row, state] for row, state in enumerate(path))
                    + .3*sum(transition[row, path[row], path[row+1]] for row in range(3)))

        expected = max(itertools.product(range(3), repeat=4), key=joint)
        np.testing.assert_array_equal(states, expected)
        self.assertEqual(breaks, [])

    def test_missing_candidates_do_not_invent_a_road(self):
        ids = np.array([[0, 1], [-1, -1], [0, 1]])
        geometry = dict(distances=np.zeros((2, 2, 2)), displacement=np.zeros(2), eligible=np.ones(2, bool))
        states, breaks = viterbi(np.array([[.8, .2], [0, 0], [.1, .9]]), ids, geometry, 10, 1)
        np.testing.assert_array_equal(chosen_arcs(ids, states), [0, -1, 1])
        self.assertEqual(breaks, [dict(observation=1, reason="missing_candidates")])

    def test_zero_probabilities_use_a_finite_numerical_floor(self):
        ids = np.array([[0, 1], [0, 1]])
        geometry = dict(distances=np.array([[[0., np.inf], [np.inf, 0.]]]),
                        displacement=np.zeros(1), eligible=np.ones(1, bool))
        states, breaks = viterbi(np.array([[1., 0.], [0., 1.]]), ids, geometry, 10, 1)
        self.assertEqual(states[0], states[1])
        self.assertEqual(breaks, [])

    def test_gap_and_disconnection_restart_the_sequence(self):
        ids = np.tile(np.arange(2), (3, 1))
        geometry = dict(distances=np.full((2, 2, 2), np.inf),
                        displacement=np.zeros(2), eligible=np.array([False, True]))
        states, breaks = viterbi(np.array([[.9, .1], [.1, .9], [.8, .2]]), ids, geometry, 10, 1)
        np.testing.assert_array_equal(states, [0, 1, 0])
        self.assertEqual([item["reason"] for item in breaks], ["time_gap", "unreachable_candidates"])

    def test_connected_alternative_overrides_independent_winners(self):
        ids = np.array([[0, 1], [0, 1]])
        geometry = dict(distances=np.array([[[np.inf, 1.], [1., np.inf]]]),
                        displacement=np.ones(1), eligible=np.ones(1, bool))
        states, breaks = viterbi(np.array([[.9, .1], [.9, .1]]), ids, geometry, 10, 1)
        self.assertNotEqual(states[0], states[1])
        self.assertEqual(breaks, [])

    def test_directed_paths_and_inserted_intermediate_roads(self):
        # Four junctions about 11 m apart. The middle road is unobserved.
        nodes = np.array([[0., 0.], [.0001, 0.], [.0002, 0.], [.0003, 0.]])
        arcs = np.array([[0, 1], [1, 2], [2, 3]])
        track = np.array([[.00005, 0., 0.], [.00025, 0., 1.]])
        ids = np.array([[0], [2]])
        network = RoadNetwork(nodes, arcs, track[0, :2])
        self.assertIsNone(network.connecting_arcs(3, 0, 200))
        self.assertEqual(network.connecting_arcs(1, 2, 200), [1])
        self.assertIsNone(network.connecting_arcs(1, 2, 5))
        geometry = transition_geometry(network, track, ids)
        self.assertAlmostEqual(geometry["distances"][0, 0, 0], 22.239016, places=5)
        states, breaks = viterbi(np.ones((2, 1)), ids, geometry, 10, 1)
        sections = route_sections(network, track, ids, geometry, states, breaks)
        self.assertEqual(sections[0]["arc_ids"], [0, 1, 2])

    def test_long_gap_produces_separate_route_sections(self):
        nodes = np.array([[0., 0.], [.0001, 0.]])
        arcs = np.array([[0, 1]])
        track = np.array([[.00001, 0., 0.], [.00009, 0., 100.]])
        ids = np.zeros((2, 1), dtype=int)
        network = RoadNetwork(nodes, arcs, track[0, :2])
        geometry = transition_geometry(network, track, ids)
        states, breaks = viterbi(np.ones((2, 1)), ids, geometry, 10, 1)
        self.assertEqual(len(route_sections(network, track, ids, geometry, states, breaks)), 2)

    def test_backwards_motion_requires_a_directed_path_beyond_jitter(self):
        nodes = np.array([[0., 0.], [.001, 0.]])
        arcs = np.array([[0, 1]])
        track = np.array([[.0009, 0., 0.], [.0001, 0., 1.]])
        ids = np.zeros((2, 1), dtype=int)
        network = RoadNetwork(nodes, arcs, track[0, :2])
        geometry = transition_geometry(network, track, ids)
        self.assertFalse(np.isfinite(geometry["distances"]).any())
        _, breaks = viterbi(np.ones((2, 1)), ids, geometry, 10, 1)
        self.assertEqual(breaks[0]["reason"], "unreachable_candidates")


if __name__ == "__main__":
    unittest.main()
