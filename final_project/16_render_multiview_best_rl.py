import carla
import os
import json
import math
import time
import argparse
import random
import subprocess
from queue import Queue, Empty

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RL_PARAM_DIR = os.path.join(BASE_DIR, "outputs", "llm_predictions_rl")
MULTIVIEW_DIR = os.path.join(BASE_DIR, "outputs", "multiview_frames")
VIDEO_DIR = os.path.join(BASE_DIR, "outputs", "multiview_videos")

os.makedirs(MULTIVIEW_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


def set_weather(world, name):
    world.set_weather(carla.WeatherParameters.ClearSunset)


def get_camera_bp(bp_lib, width=1280, height=720, fov=85):
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    return cam_bp


def save_image_callback(queue):
    def _callback(image):
        queue.put(image)
    return _callback


def make_camera(world, bp, transform, attach_to=None):
    if attach_to is not None:
        return world.spawn_actor(bp, transform, attach_to=attach_to)
    return world.spawn_actor(bp, transform)


def spawn_background_traffic(world, bp_lib, spawn, actors, count=20):
    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicle_bps = [
        bp for bp in vehicle_bps
        if "bike" not in bp.id.lower()
        and "motorcycle" not in bp.id.lower()
        and "carlamotors" not in bp.id.lower()
    ]

    traffic_offsets = [
    (-35, -3.5, 0.35), (-60, -7.0, 0.42), (-85, -10.5, 0.46),
    (-115, -3.5, 0.38), (-145, -7.0, 0.42),

    (35, -3.5, 0.32), (65, -7.0, 0.40), (95, -10.5, 0.44),
    (125, -3.5, 0.36), (155, -7.0, 0.40),

    (-40, 3.5, 0.34), (-75, 7.0, 0.42), (-110, 10.5, 0.45),
    (-140, 3.5, 0.38),

    (40, 3.5, 0.32), (75, 7.0, 0.42), (110, 10.5, 0.45),
    (145, 3.5, 0.38),
    ]

    for dx, dy, throttle in traffic_offsets[:count]:
        bp = random.choice(vehicle_bps)

        tf = carla.Transform(
            carla.Location(
                x=spawn.location.x + dx,
                y=spawn.location.y + dy,
                z=spawn.location.z + 0.3
            ),
            spawn.rotation
        )

        try:
            v = world.try_spawn_actor(bp, tf)

            if v is not None:
                actors.append(v)
                v.set_autopilot(False)

                if dy > 0:
                    v.set_transform(
                        carla.Transform(
                            tf.location,
                            carla.Rotation(
                                pitch=spawn.rotation.pitch,
                                yaw=spawn.rotation.yaw + 180,
                                roll=spawn.rotation.roll
                            )
                        )
                    )

                v._background_throttle = throttle

        except Exception:
            pass


def apply_background_traffic_controls(actors, ego_vehicle):
    for a in actors:
        if a.id == ego_vehicle.id:
            continue

        if "vehicle" in a.type_id:
            throttle = getattr(a, "_background_throttle", 0.50)
            a.apply_control(
                carla.VehicleControl(
                    throttle=float(throttle),
                    steer=0.0,
                    brake=0.0
                )
            )


def create_cameras(world, bp_lib, vehicle, spawn):
    cams = {}

    # 1. Top view: clean trajectory view
    top_bp = get_camera_bp(bp_lib, width=1280, height=720, fov=88)
    top_tf = carla.Transform(
        carla.Location(x=0, y=0, z=52),
        carla.Rotation(pitch=-90, yaw=0, roll=0)
    )
    cams["top"] = make_camera(world, top_bp, top_tf, attach_to=vehicle)

    # 2. Ego view: dashcam-like
    ego_bp = get_camera_bp(bp_lib, width=1280, height=720, fov=95)
    ego_tf = carla.Transform(
        carla.Location(x=1.7, y=0.0, z=1.45),
        carla.Rotation(pitch=-3, yaw=0, roll=0)
    )
    cams["ego"] = make_camera(world, ego_bp, ego_tf, attach_to=vehicle)

    # 3. Front CCTV: roadside surveillance angle
    cctv_front_bp = get_camera_bp(bp_lib, width=1280, height=720, fov=78)
    cctv_front_tf = carla.Transform(
        carla.Location(
            x=spawn.location.x + 32,
            y=spawn.location.y - 9,
            z=7
        ),
        carla.Rotation(pitch=-10, yaw=150, roll=0)
    )
    cams["cctv_front"] = make_camera(world, cctv_front_bp, cctv_front_tf)

    # 4. Opposite CCTV: less zoomed out than before, still captures crash region
    cctv_opp_bp = get_camera_bp(bp_lib, width=1280, height=720, fov=76)
    cctv_opp_tf = carla.Transform(
        carla.Location(
            x=spawn.location.x - 30,
            y=spawn.location.y + 20,
            z=10
        ),
        carla.Rotation(pitch=-13, yaw=-24, roll=0)
    )
    cams["cctv_opposite"] = make_camera(world, cctv_opp_bp, cctv_opp_tf)

    return cams


def build_video_from_frames(case_id, view_name, fps=20):
    frame_dir = os.path.join(MULTIVIEW_DIR, case_id, view_name)
    out_video = os.path.join(VIDEO_DIR, f"{case_id}_{view_name}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "%06d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        out_video
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_video


def run_case(param_path, duration=12.0):
    with open(param_path, "r") as f:
        p = json.load(f)

    case_id = str(p["case_id"])
    random.seed(int(case_id[-4:]))

    case_frame_dir = os.path.join(MULTIVIEW_DIR, case_id)
    os.makedirs(case_frame_dir, exist_ok=True)

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)

    world = client.load_world(p.get("town", "Town04"))
    time.sleep(2)

    set_weather(world, p.get("weather", "ClearSunset"))

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

    spawn_background_traffic(world, bp_lib, spawn, actors, count=10)

    cameras = create_cameras(world, bp_lib, vehicle, spawn)
    actors.extend(cameras.values())

    queues = {}

    for name, cam in cameras.items():
        view_dir = os.path.join(case_frame_dir, name)
        os.makedirs(view_dir, exist_ok=True)

        q = Queue()
        cam.listen(save_image_callback(q))
        queues[name] = q

    side = p.get("lane_departure_side", "right")
    side_sign = 1 if side == "right" else -1

    trigger = float(p["steer_trigger_time"])
    steer_duration = float(p["steer_duration"])

    wobble_amp = random.uniform(0.04, 0.16)
    wobble_freq = random.uniform(4.0, 8.0)

    spin_mode = p.get("spin_mode", "medium")
    handbrake_strength = random.random() < 0.45

    if spin_mode == "mild":
        handbrake_strength = False
        wobble_amp *= 0.7

    elif spin_mode == "aggressive":
        handbrake_strength = True
        wobble_amp *= 1.5
        p["brake_factor"] = min(1.0, float(p["brake_factor"]) + 0.25)

    brake_pulse_time = trigger + random.uniform(0.4, 1.3)
    handbrake_time = trigger + random.uniform(0.7, 1.8)

    frame_id = 0
    t = 0.0

    while t <= duration:
        world.tick()

        apply_background_traffic_controls(actors, vehicle)

        throttle = 0.55
        steer = 0.0
        brake = 0.0
        hand_brake = False

        if t < trigger:
            throttle = 0.65

        elif trigger <= t < trigger + steer_duration:
            dt = t - trigger
            wobble = wobble_amp * math.sin(wobble_freq * dt)

            throttle = 0.35
            steer = side_sign * (float(p["steer_magnitude"]) + wobble)
            brake = float(p["brake_factor"]) * random.uniform(0.2, 0.5)

        elif trigger + steer_duration <= t < trigger + steer_duration + 1.6:
            dt = t - (trigger + steer_duration)
            wobble = wobble_amp * math.sin(wobble_freq * dt + 1.5)

            throttle = random.uniform(0.05, 0.22)
            steer = side_sign * (float(p["counter_steer_magnitude"]) + wobble)
            brake = float(p["brake_factor"]) * random.uniform(0.8, 1.2)

            if handbrake_strength and abs(t - handbrake_time) < 0.20:
                hand_brake = True

        else:
            throttle = random.uniform(0.0, 0.08)
            steer = side_sign * random.uniform(-0.08, 0.08)
            brake = min(1.0, float(p["brake_factor"]) + random.uniform(0.2, 0.5))

        if abs(t - brake_pulse_time) < 0.25:
            brake = min(1.0, brake + random.uniform(0.2, 0.4))

        vehicle.apply_control(
            carla.VehicleControl(
                throttle=max(0.0, min(1.0, float(throttle))),
                steer=max(-1.0, min(1.0, float(steer))),
                brake=max(0.0, min(1.0, float(brake))),
                hand_brake=hand_brake
            )
        )

        for name, q in queues.items():
            try:
                image = q.get(timeout=2.0)
                out_path = os.path.join(case_frame_dir, name, f"{frame_id:06d}.png")
                image.save_to_disk(out_path)
            except Empty:
                print(f"[WARN] Missing frame for {name} at {frame_id}")

        frame_id += 1
        t += settings.fixed_delta_seconds

    for cam in cameras.values():
        cam.stop()

    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass

    settings.synchronous_mode = False
    world.apply_settings(settings)

    print(f"Rendered cinematic frames for {case_id}")

    for view in ["top", "ego", "cctv_front", "cctv_opposite"]:
        video = build_video_from_frames(case_id, view, fps=20)
        print(view, "->", video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--param_path", default=None)
    parser.add_argument("--duration", type=float, default=12.0)

    args = parser.parse_args()

    if args.param_path is None:
        param_path = os.path.join(RL_PARAM_DIR, f"{args.case_id}_rl.json")
    else:
        param_path = args.param_path

    run_case(param_path, duration=args.duration)