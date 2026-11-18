import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from api import agreement
from refine import refined_route
from roads import ROOT, load_trip


class DemoChecks(unittest.TestCase):
    def test_no_reference_is_unscored(self):
        with TemporaryDirectory() as folder:
            self.assertIsNone(agreement(Path(folder), None, []))

    def test_reference_agreement(self):
        with TemporaryDirectory() as folder:
            trip = Path(folder)
            (trip / f"{trip.name}.route").write_text("0\n")
            nodes = np.array([[1., 2.], [1.1, 2.]])
            track = np.array([[1.,2.,0.], [1.1,2.,1.]])
            data = nodes, np.array([[0,1]]), track
            self.assertEqual(agreement(trip, data, [0,1]), .5)

    def test_prediction_without_reference(self):
        import joblib
        from api import BASE
        from threadpoolctl import threadpool_limits

        trip = ROOT / "student_test/00000091"
        if not trip.is_dir():
            self.skipTest("Set PATHSNAP_DATA to run dataset integration check")
        data = load_trip(trip)
        model = joblib.load(BASE / "results/refinement/model.joblib")
        with patch("refine.load_trip", side_effect=AssertionError), threadpool_limits(limits=2):
            result = refined_route(Path("nonexistent"),model["classifier"],model["configuration"],trip_data=data)
        self.assertEqual(len(result["decoded_arc_ids"]), len(data[2]))
