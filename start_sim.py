# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
    创建石头并破碎
    

    # Usage
    isaaclab.sh/bat -p ./start_sim.py
    or
    python ./start_sim.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher




# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
import usdTools

import carb.input
import omni.appwindow
import numpy as np
from isaaclab.sim import PhysxCfg

appwindow = omni.appwindow.get_default_app_window()
keyboard = appwindow.get_keyboard()
get_closest=False

impact=False
reset=False
def on_input(e):
    global impact
    global get_closest
    global reset
    if e.type == carb.input.KeyboardEventType.KEY_PRESS:
        if e.input == carb.input.KeyboardInput.T:
            get_closest=True
        if e.input == carb.input.KeyboardInput.B:

            impact=True
        if e.input == carb.input.KeyboardInput.R:
            reset=True
    return True


input = carb.input.acquire_input_interface()

def design_scene():
    # 创建基础场景
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground,translation=(0, 0, -3.0))
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    sim_utils.create_prim("/World/Objects", "Xform")

    # 生成石头
    # rock=usdTools.generate_prebroken_rock(num_points=50, scale=0.5, num_cells=10, root_path="/World/Objects/base_rock_0", base_translation=(0, 0, 0), seed=None)
    rock=usdTools.load_rock_from_file(file_name="rock_data.pkl",rock_name="base_rock_0",root_path="/World/Objects/base_rock_0",base_translation=(0, 0, 0))
    print(len(rock.rock_objects))
    return rock


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.02,physx= PhysxCfg(enable_external_forces_every_iteration=True))
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    rock=design_scene()
    keyboard_sub_id = input.subscribe_to_keyboard_events(keyboard, on_input)
    
    sim.reset()
    
    sim_dt = sim.get_physics_dt()
    total_time = 0.0
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    global impact
    global get_closest
    global reset
    # Simulate physics
    while simulation_app.is_running():
        
        # perform step
        sim.step()

            
        total_time += sim_dt
        usdTools.update_rocks_state(rock, sim_dt)

        if get_closest:
            closest_id, rock, distance = usdTools.get_closest_rock_and_id(rock, impact_point=[0, 0, 10])
            print(f"Closest rock ID: {closest_id}, Distance: {distance}")
            get_closest=False
            rock=usdTools.apply_impact(rock, impact_idx=closest_id, impact_dir=[0, -1, 0])
        
        if impact:

            print("Applying impact...")
            idx = np.argmax(rock.centers[:, 2])
            rock=usdTools.apply_impact(rock, impact_idx=idx, impact_dir=[0, -1, 0])
            impact=False
            
        if reset:
            rock=usdTools.reset_rock(rock,"base_rock_0")
            reset= False

        


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()