# QUANTA_X2 asset

Load `quanta_x2.usd` for the complete mobile robot with two seven-joint arms and
integrated grippers. The single USD includes four waist joints, head joints,
drive wheels, four cameras, an IMU link, and an RTX 2D lidar.

Geometry, grippers, and the lidar definition are stored directly in the USD,
without external USD references. `OmniPBR.mdl` is the built-in Isaac Sim material
module; the asset does not require a network asset server. The USD uses Git LFS.

The SDK end-effector bodies are `left_gripper_base_link` and
`right_gripper_base_link`, coincident with `left_flange` and `right_flange`;
the base is `base_link`. Active gripper joints are `left_gripper` and
`right_gripper`, with targets in radians from 0 to 1.89. Four additional finger
joints follow the gripper motors through mimic constraints authored in the USD.

See [SDK instructions](../../docs/quanta_x2_sdk.md) for launching and ROS topics.
