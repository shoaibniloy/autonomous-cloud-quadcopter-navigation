#!/usr/bin/env python3
import cv2
import zmq
import time
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_zmq_publisher(port="5556"):
    """Initialize ZMQ PUB socket."""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(1)  # Allow subscribers to connect
    logger.info(f"ZMQ publisher bound to tcp://*:{port}")
    return context, socket

def main():
    parser = argparse.ArgumentParser(description="Camera feed server")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--resolution", nargs=2, type=int, default=[240, 180], help="Camera resolution (width height)")
    parser.add_argument("--port", type=str, default="5556", help="ZMQ port for broadcasting frames")
    args = parser.parse_args()

    # Initialize ZMQ
    context, socket = setup_zmq_publisher(port=args.port)

    # Initialize camera
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        logger.error(f"Could not open camera device {args.device}")
        raise RuntimeError(f"Could not open camera device {args.device}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.resolution[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    logger.info(f"Camera initialized with resolution {args.resolution[0]}x{args.resolution[1]}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue

            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_data = buffer.tobytes()

            # Send frame via ZMQ
            socket.send(frame_data)
            logger.debug("Sent frame via ZMQ")

            time.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        logger.info("Camera server interrupted by user")
    finally:
        cap.release()
        socket.close()
        context.term()
        logger.info("Camera server stopped")

if __name__ == "__main__":
    main()