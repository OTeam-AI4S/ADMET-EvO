from scripts.build_v26_goal_scorecard import _digest


def test_v26_digest_is_order_stable():
    left = {"protocol": "v", "tasks": ["A"], "finalists": {"A": {"x": 1, "y": 2}}}
    right = {"finalists": {"A": {"y": 2, "x": 1}}, "tasks": ["A"], "protocol": "v"}
    assert _digest(left) == _digest(right)
