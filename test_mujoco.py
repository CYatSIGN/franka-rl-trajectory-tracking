import mujoco

# Create a simple simulation
xml = """
<mujoco>
    <worldbody>
        <body name="test_body" pos="0 0 1">
            <freejoint/>
            <geom type="sphere"
                  size="0.1"
                  mass="1"/>
        </body>

        <geom type="plane"
              size="2 2 0.1"/>
    </worldbody>
</mujoco>
"""

# Load model
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("Starting MuJoCo simulation...")

# Simulate 1000 steps
for step in range(1000):
    mujoco.mj_step(model, data)

print("Simulation completed successfully!")
print("Simulation time:", data.time)
print("Final sphere height:", data.qpos[2])