# Prospective transport-grid materiality audit

- Protocol SHA-256: `500a7dd3d312f56348c4fe2f49d1c9d3e21deb19594997436021cc5b81db0e79`
- Production source SHA-256: `d72e0502cd609aa7fc25d3a2b76ab217889381d3ed49e6ed57f9f3593fb9fa6a`
- Sentinel endpoints: 5
- Validation splits audited: 25
- Data action changes training membership: 25/25 seed-specific splits
- Endpoints with at least one material data action: 5/5
- Model fits: 0
- Test loaded: false

This is a preflight audit only. It verifies that the data intervention is not
a universal no-op before authorizing the 150-fit complete grid. It does not
estimate action effects and cannot make a cross-task memory claim.
