# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

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
import prebreakv2
import usdTools

import carb.windowing
import carb.input
import omni.appwindow
import numpy as np

from pxr import UsdPhysics
appwindow = omni.appwindow.get_default_app_window()
keyboard = appwindow.get_keyboard()
get_closest=False

impact=False
def on_input(e):
    global impact
    global get_closest
    if e.type == carb.input.KeyboardEventType.KEY_PRESS:
        if e.input == carb.input.KeyboardInput.T:
            get_closest=True
        if e.input == carb.input.KeyboardInput.B:

            impact=True
    return True


input = carb.input.acquire_input_interface()

# import impact_model

def design_scene(sim):
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground,translation=(0, 0, -3.0))

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    # create a new xform prim for all objects to be spawned under
    sim_utils.create_prim("/World/Objects", "Xform")
    # rock = prebreakv2.generate_random_rock(num_points=40,scale=0.5)
    # import test
    # test.spawn_trimesh_as_rigid_body("/World/Objects/base_rock_1", rock, mass=0.3)
    
    # rock=usdTools.spawn_rock(sim, rock_usd_path="rock.usd", root_path="/World/Objects/rock_0", base_translation=(0, 0, 10))

    # # Voronoi破碎
    # fragments = prebreakv2.voronoi_fracture(rock, num_cells=25)
    
    # masses = prebreakv2.compute_mass(fragments)
    
    # adj,centers,ids=prebreakv2.build_connectivity(fragments)
    # # 直接加载进场景
    # root, mesh_prims = usdTools.load_meshes_to_isaaclab(fragments, masses, root_path="/World/Objects/base_rock_0", base_translation=(0, 0, 10))
    
    
    # impact=impact_model.apply_impact(centers,adj,idx,[0, -1, 0])
    # groups_subgraphs, groups_centers, groups_ids = impact_model.group_by_subgraphs(impact, centers, ids)
    
    # usdTools.update_break_meshes(groups_ids, root)
    
    rock=usdTools.generate_prebroken_rock(num_points=50, scale=0.5, num_cells=10, root_path="/World/Objects/base_rock_0", base_translation=(0, 0, 0), seed=None)
    
    
    
    return rock

# from isaaclab.sim.views import XformPrimView
from isaaclab.sim import PhysxCfg
# from isaacsim.core.prims import RigidPrim

import time
def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.02,physx= PhysxCfg(enable_external_forces_every_iteration=True,))
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    rock=design_scene(sim)
    # print(rock.centers, rock.ids, rock.adj)
    keyboard_sub_id = input.subscribe_to_keyboard_events(keyboard, on_input)
    # sleep_time = 0.01
    
    # time.sleep(1)  # 等待场景加载完成
    # Play the simulator
    stage=omni.usd.get_context().get_stage()
    stage.Flatten()
    
    sim.reset()
    
    sim_dt = sim.get_physics_dt()
    total_time = 0.0
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    global impact
    global get_closest
    isimpacted=False
    # Simulate physics
    ds=0
    while simulation_app.is_running():
        # perform step
        
        
        
        sim.step()
        # if isimpacted:
        #     for rock in rocks:
        #         rock.rock_obj._initialize_callback(None)
        #         print(rock.rock_obj.is_initialized)
                
        #     isimpacted=False
            
        total_time += sim_dt
        usdTools.update_rocks_state(rock, sim_dt)
        # print(rock.rock_objects[0].data.root_state_w)
        # for idx, meshview in enumerate(rock.meshviews):
        #     positions, orientations = rock.meshviews[idx].get_world_poses()
        # print(positions)
        # if not isimpacted:
        # print(rock.rock_obj.is_initialized)
        #     rock.rock_obj.update(sim_dt)
        if get_closest:
            closest_id, rock, distance = usdTools.get_closest_rock_and_id(rock, impact_point=[0, 0, 10])
            print(f"Closest rock ID: {closest_id}, Distance: {distance}")
            get_closest=False
            rock=usdTools.apply_impact(rock, impact_idx=closest_id, impact_dir=[0, -1, 0])
        
        if impact:
            sim.forward()
            
            # positions, orientations = rock.meshviews[2].get_world_poses()
            # print(positions)
            print("Applying impact...")
            # rock.rock_obj.update(total_time)
            
            
            
            idx = np.argmax(rock.centers[:, 2])
            # sim_utils.delete_prim("/World/Objects/rock_0")
            
            rock=usdTools.apply_impact(rock, impact_idx=idx, impact_dir=[0, -1, 0])
            # print(rock)
            impact=False
            isimpacted=True
        
            # prim = stage.GetPrimAtPath("/World/Objects/base_rock_0_group_0")

            # print(UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath()).IsValid())

            
        # if isimpacted:
            

        
        # sim.render()
        


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()