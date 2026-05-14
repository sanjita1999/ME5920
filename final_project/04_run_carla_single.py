import carla
import os
import csv
import json
import math
import time
import argparse
import random


def kmh_to_ms(kmh):
    return kmh / 3.6


def get_speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)


def yaw_diff(yaw1, yaw0):
    d = yaw1 - yaw0
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def set_weather(world, name):
    presets = {
        "ClearNoon": carla.WeatherParameters.ClearNoon,
        "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
        "ClearSunset": carla.WeatherParameters.ClearSunset,
    }
    world.set_weather(presets.get(name, carla.WeatherParameters.ClearNoon))


def run_case(param_path, traj_out, outcome_out, duration=12.0):
    with open(param_path, "r") as f:
        p = json.load(f)

    random.seed(int(str(p["case_id"])[-4:]))

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world = client.load_world(p.get("town", "Town04"))
    time.sleep(2)

    set_weather(world, p.get("weather", "ClearNoon"))

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter(p.get("vehicle_model", "vehicle.tesla.model3"))[0]

    spawn_points = world.get_map().get_spawn_points()
    spawn = spawn_points[100]

    vehicle = world.spawn_actor(vehicle_bp, spawn)
    actors = [vehicle]

    os.makedirs(os.path.dirname(traj_out), exist_ok=True)
    os.makedirs(os.path.dirname(outcome_out), exist_ok=True)

    start_yaw = vehicle.get_transform().rotation.yaw
    max_yaw_change = 0.0
    max_lateral_offset = 0.0
    max_speed = 0.0

    lane_departure = False
    roadside_contact = False

    side = p["lane_departure_side"]
    side_sign = 1 if side == "right" else -1

    trigger = p["steer_trigger_time"]
    steer_duration = p["steer_duration"]

    wobble_amp = random.uniform(0.04, 0.16)
    wobble_freq = random.uniform(4.0, 8.0)
    brake_pulse_time = trigger + random.uniform(0.4, 1.3)
    handbrake_time = trigger + random.uniform(0.7, 1.8)
    handbrake_strength = random.random() < 0.45

    spin_mode = p.get("spin_mode", "medium")

    if spin_mode == "mild":
        handbrake_strength = False
        wobble_amp *= 0.7

    elif spin_mode == "medium":
        pass

    elif spin_mode == "aggressive":
        handbrake_strength = True
        wobble_amp *= 1.5
        p["brake_factor"] = min(1.0, p["brake_factor"] + 0.25)


    with open(traj_out, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time",
            "x", "y", "z",
            "speed_kmh",
            "yaw",
            "yaw_change",
            "lateral_offset",
            "lane_departure",
            "roadside_contact",
            "control_throttle",
            "control_steer",
            "control_brake",
            "control_hand_brake",
            "phase"
        ])

        t = 0.0

        while t <= duration:
            world.tick()

            throttle = 0.55
            steer = 0.0
            brake = 0.0
            hand_brake = False
            phase = "normal"

            if t < trigger:
                throttle = 0.65
                steer = 0.0
                brake = 0.0
                phase = "approach"

            elif trigger <= t < trigger + steer_duration:
                dt = t - trigger

                wobble = wobble_amp * math.sin(wobble_freq * dt)

                throttle = 0.35
                steer = side_sign * (p["steer_magnitude"] + wobble)
                brake = p["brake_factor"] * random.uniform(0.2, 0.5)
                phase = "departure_steer"

            elif trigger + steer_duration <= t < trigger + steer_duration + 1.6:
                dt = t - (trigger + steer_duration)

                wobble = wobble_amp * math.sin(wobble_freq * dt + 1.5)

                throttle = random.uniform(0.05, 0.22)
                steer = side_sign * (p["counter_steer_magnitude"] + wobble)
                brake = p["brake_factor"] * random.uniform(0.8, 1.2)
                phase = "counter_steer_spin"

                if handbrake_strength and abs(t - handbrake_time) < 0.20:
                    hand_brake = True

            else:
                throttle = random.uniform(0.0, 0.08)
                steer = side_sign * random.uniform(-0.08, 0.08)
                brake = min(1.0, p["brake_factor"] + random.uniform(0.2, 0.5))
                phase = "settle"

            if abs(t - brake_pulse_time) < 0.25:
                brake = min(1.0, brake + random.uniform(0.2, 0.4))

            steer = max(-1.0, min(1.0, steer))
            brake = max(0.0, min(1.0, brake))
            throttle = max(0.0, min(1.0, throttle))

            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(steer),
                    brake=float(brake),
                    hand_brake=hand_brake
                )
            )

            tf = vehicle.get_transform()
            loc = tf.location
            yaw = tf.rotation.yaw
            yd = abs(yaw_diff(yaw, start_yaw))

            speed = get_speed_kmh(vehicle)
            max_speed = max(max_speed, speed)
            max_yaw_change = max(max_yaw_change, yd)

            lateral_offset = abs(loc.y - spawn.location.y)
            max_lateral_offset = max(max_lateral_offset, lateral_offset)

            if lateral_offset > 3.0:
                lane_departure = True

            if lateral_offset > 5.5:
                roadside_contact = True

            writer.writerow([
                round(t, 2),
                loc.x,
                loc.y,
                loc.z,
                speed,
                yaw,
                yd,
                lateral_offset,
                lane_departure,
                roadside_contact,
                throttle,
                steer,
                brake,
                hand_brake,
                phase
            ])

            t += settings.fixed_delta_seconds

    outcome = {
        "case_id": p["case_id"],
        "lane_departure_success": lane_departure,
        "roadside_contact_success": roadside_contact,
        "max_yaw_change": round(max_yaw_change, 2),
        "max_lateral_offset": round(max_lateral_offset, 2),
        "max_speed_kmh": round(max_speed, 2),
        "target_yaw_change": p["target_yaw_change"],
        "rollover_label": p["rollover_label"],
        "wobble_amp": round(wobble_amp, 3),
        "wobble_freq": round(wobble_freq, 3),
        "handbrake_used": handbrake_strength
    }

    with open(outcome_out, "w") as f:
        json.dump(outcome, f, indent=4)

    for a in actors:
        a.destroy()

    settings.synchronous_mode = False
    world.apply_settings(settings)

    print("Finished:", p["case_id"])
    print(outcome)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--traj_out", required=True)
    parser.add_argument("--outcome_out", required=True)
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()

    run_case(args.params, args.traj_out, args.outcome_out, args.duration)