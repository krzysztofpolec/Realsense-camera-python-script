import cv2
import pyrealsense2 as rs
import numpy as np
import datetime

# Configuration for RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the RealSense pipeline
pipeline.start(config)

# Define the codec and create VideoWriter objects
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
color_filename = f"color_video_{timestamp}.avi"
depth_filename = f"depth_video_{timestamp}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
color_out = None
depth_out = None
is_recording = False

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        if is_recording:
            # Write the frames into their respective files
            color_out.write(color_image)
            depth_out.write(depth_colormap)  # Writing the visual representation of the depth frame

        # Display the resulting frames
        cv2.imshow('RealSense Color Stream', color_image)
        cv2.imshow('RealSense Depth Stream', depth_colormap)

        key = cv2.waitKey(1)
        if key == ord('q'):  # Quit or start/stop recording
            if is_recording:
                is_recording = False
                color_out.release()
                depth_out.release()
                print("Recording stopped.")
                break  # or just continue if you want to toggle recording on/off
            else:
                is_recording = True
                color_out = cv2.VideoWriter(color_filename, fourcc, 30.0, (640, 480))
                depth_out = cv2.VideoWriter(depth_filename, fourcc, 30.0, (640, 480))
                print("Started recording.")

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()