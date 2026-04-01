from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import cv2
from furhat_realtime_api import FurhatClient


def create_tracker() -> Optional[object]:
	# OpenCV tracker API changed across versions; try both legacy and modern locations.
	try:
		if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
			return cv2.legacy.TrackerMOSSE_create()
	except Exception:
		pass
	try:
		if hasattr(cv2, "TrackerMOSSE_create"):
			return cv2.TrackerMOSSE_create()
	except Exception:
		pass
	return None


class FurhatFaceFollower:
	def __init__(
		self,
		furhat: FurhatClient,
		max_yaw: float = 0.35,
		max_pitch: float = 0.20,
		smooth_alpha: float = 0.35,
		send_interval_s: float = 0.12,
		track_pitch: bool = False,
		invert_x: bool = True,
		center_gaze: bool = False,
	) -> None:
		self.furhat = furhat
		self.max_yaw = max_yaw
		self.max_pitch = max_pitch
		self.smooth_alpha = smooth_alpha
		self.send_interval_s = send_interval_s
		self.track_pitch = track_pitch
		self.invert_x = invert_x
		self.center_gaze = center_gaze

		self._smoothed_yaw = 0.0
		self._smoothed_pitch = 0.0
		self._last_send_ts = 0.0
		self._attend_method: Optional[str] = None
		self._last_seen_ts = 0.0
		self._last_yaw = 0.0
		self._last_pitch = 0.0
		self._gaze_method: Optional[str] = None

	def _send_center_gaze(self) -> None:
		if self._gaze_method == "request_gaze" and hasattr(self.furhat, "request_gaze"):
			try:
				self.furhat.request_gaze(x=0.0, y=0.0, z=1.0)
				return
			except Exception:
				self._gaze_method = None

		if self._gaze_method == "request_look_at" and hasattr(self.furhat, "request_look_at"):
			try:
				self.furhat.request_look_at(x=0.0, y=0.0, z=1.0)
				return
			except Exception:
				self._gaze_method = None

		if hasattr(self.furhat, "request_gaze"):
			try:
				self.furhat.request_gaze(x=0.0, y=0.0, z=1.0)
				self._gaze_method = "request_gaze"
				return
			except Exception:
				pass

		if hasattr(self.furhat, "request_look_at"):
			try:
				self.furhat.request_look_at(x=0.0, y=0.0, z=1.0)
				self._gaze_method = "request_look_at"
				return
			except Exception:
				pass

	def _normalize_to_angles(self, x_center: float, y_center: float, width: int, height: int) -> tuple[float, float]:
		x_norm = ((x_center / width) - 0.5) * 2.0
		y_norm = ((y_center / height) - 0.5) * 2.0

		# Keep values small because attend-location expects near-field coordinates, not large angles.
		if self.invert_x:
			x_norm = -x_norm
		yaw = max(-1.0, min(1.0, x_norm)) * self.max_yaw
		pitch = 0.0
		if self.track_pitch:
			pitch = max(-1.0, min(1.0, y_norm)) * self.max_pitch
		return yaw, pitch

	def _smooth(self, yaw: float, pitch: float) -> tuple[float, float]:
		a = self.smooth_alpha
		self._smoothed_yaw = a * yaw + (1.0 - a) * self._smoothed_yaw
		self._smoothed_pitch = a * pitch + (1.0 - a) * self._smoothed_pitch
		return self._smoothed_yaw, self._smoothed_pitch

	def _try_attend_location(self, yaw: float, pitch: float) -> bool:
		if not hasattr(self.furhat, "request_attend_location"):
			return False

		fn = getattr(self.furhat, "request_attend_location")
		attempts = [
			lambda: fn(x=yaw, y=-pitch, z=1.0),
			lambda: fn({"x": yaw, "y": -pitch, "z": 1.0}),
			lambda: fn(location={"x": yaw, "y": -pitch, "z": 1.0}),
		]
		for attempt in attempts:
			try:
				attempt()
				self._attend_method = "request_attend_location"
				return True
			except TypeError:
				continue
			except Exception:
				return False
		return False

	def _try_attend(self, yaw: float, pitch: float) -> bool:
		if not hasattr(self.furhat, "request_attend"):
			return False

		fn = getattr(self.furhat, "request_attend")
		attempts = [
			lambda: fn(location={"x": yaw, "y": -pitch, "z": 1.0}),
			lambda: fn(target={"x": yaw, "y": -pitch, "z": 1.0}),
			lambda: fn({"x": yaw, "y": -pitch, "z": 1.0}),
		]
		for attempt in attempts:
			try:
				attempt()
				self._attend_method = "request_attend"
				return True
			except TypeError:
				continue
			except Exception:
				return False
		return False

	def send_target(self, yaw: float, pitch: float) -> bool:
		sent = False
		if self._attend_method == "request_attend_location":
			sent = self._try_attend_location(yaw, pitch)
			if sent and self.center_gaze:
				self._send_center_gaze()
			return sent
		if self._attend_method == "request_attend":
			sent = self._try_attend(yaw, pitch)
			if sent and self.center_gaze:
				self._send_center_gaze()
			return sent

		if self._try_attend_location(yaw, pitch):
			if self.center_gaze:
				self._send_center_gaze()
			return True
		if self._try_attend(yaw, pitch):
			if self.center_gaze:
				self._send_center_gaze()
			return True
		return False

	def maybe_send(self, yaw: float, pitch: float) -> bool:
		now = time.time()
		if now - self._last_send_ts < self.send_interval_s:
			return False
		self._last_send_ts = now
		self._last_yaw = yaw
		self._last_pitch = pitch
		self._last_seen_ts = now
		return self.send_target(yaw, pitch)

	def maybe_resend_last_target(self, hold_timeout_s: float) -> bool:
		now = time.time()
		if self._last_seen_ts <= 0 or (now - self._last_seen_ts) > hold_timeout_s:
			return False
		if now - self._last_send_ts < self.send_interval_s:
			return False
		self._last_send_ts = now
		return self.send_target(self._last_yaw, self._last_pitch)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--host", type=str, default="127.0.0.1", help="Furhat robot IP address")
	parser.add_argument("--auth_key", type=str, default="admin", help="Authentication key for Realtime API")
	parser.add_argument("--camera", type=int, default=0, help="Camera index")
	parser.add_argument("--yaw_gain", type=float, default=0.35, help="Max horizontal gaze offset (recommended 0.15-0.45)")
	parser.add_argument("--pitch_gain", type=float, default=0.20, help="Max vertical gaze offset (used only when pitch tracking is enabled)")
	parser.add_argument("--eye_line_ratio", type=float, default=0.38, help="Eye line inside face box (0 top .. 1 bottom)")
	parser.add_argument("--invert_x", action="store_true", help="Invert horizontal mapping if movement feels mirrored")
	parser.add_argument("--center_gaze", action="store_true", help="Keep eyes centered while head still tracks")
	args = parser.parse_args()

	furhat = FurhatClient(host=args.host, auth_key=args.auth_key)
	furhat.set_logging_level(logging.INFO)

	try:
		furhat.connect()
	except Exception:
		print(f"Failed to connect to Furhat on {args.host}.")
		return

	# Light acknowledgement so you know the loop started.
	try:
		furhat.request_speak_text("I will follow your face. Press Q to stop.")
	except Exception:
		pass

	cam = cv2.VideoCapture(args.camera)
	if not cam.isOpened():
		print("Failed to open webcam.")
		return

	detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	follower = FurhatFaceFollower(
		furhat,
		max_yaw=max(0.05, min(0.8, args.yaw_gain)),
		max_pitch=max(0.05, min(0.6, args.pitch_gain)),
		track_pitch=True,
		invert_x=not args.invert_x,
		center_gaze=args.center_gaze,
	)
	deadzone_norm = 0.08
	y_deadzone_norm = 0.05
	lost_face_hold_s = 1.8
	detect_every_n_frames = 6
	frame_idx = 0
	tracker = None
	tracked_bbox = None

	print("Running face follow. Press 'q' in the video window to quit.")
	print("Press 'm' to toggle mirror/invert X at runtime.")
	print(f"Center gaze mode: {follower.center_gaze}")
	print("Tip: If tracking does not move Furhat, your SDK may expose a different attend API.")

	while True:
		frame_idx += 1
		ok, frame = cam.read()
		if not ok:
			continue

		h, w = frame.shape[:2]

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)

		face_box = None
		tracker_ok = False
		if tracker is not None:
			tracker_ok, bbox = tracker.update(frame)
			if tracker_ok:
				tx, ty, tw, th = bbox
				face_box = (int(tx), int(ty), int(tw), int(th))
				tracked_bbox = face_box

		# Refresh from detector periodically or when tracking is lost.
		if (not tracker_ok) or (frame_idx % detect_every_n_frames == 0):
			faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
			if len(faces) > 0:
				face_box = max(faces, key=lambda b: b[2] * b[3])
				x, y, fw, fh = face_box
				new_tracker = create_tracker()
				if new_tracker is not None:
					try:
						new_tracker.init(frame, (x, y, fw, fh))
						tracker = new_tracker
						tracked_bbox = face_box
					except Exception:
						tracker = None

		if face_box is not None:
			x, y, fw, fh = face_box
			cx = x + fw / 2.0
			eye_line_ratio = max(0.2, min(0.6, args.eye_line_ratio))
			cy = y + fh * eye_line_ratio

			x_norm = ((cx / w) - 0.5) * 2.0
			y_norm = ((cy / h) - 0.5) * 2.0
			if abs(x_norm) < deadzone_norm:
				x_norm = 0.0
			if abs(y_norm) < y_deadzone_norm:
				y_norm = 0.0
			cx = ((x_norm / 2.0) + 0.5) * w
			cy = ((y_norm / 2.0) + 0.5) * h

			raw_yaw, raw_pitch = follower._normalize_to_angles(cx, cy, w, h)
			# Keep only gaze/head orientation updates. No facial expression mirroring.
			yaw, pitch = follower._smooth(raw_yaw, raw_pitch)
			sent = follower.maybe_send(yaw, pitch)

			cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
			cv2.putText(
				frame,
				f"x={yaw:.3f} y={pitch:.3f} sent={sent} invert_x={follower.invert_x}",
				(10, 28),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(50, 220, 50),
				2,
			)
		else:
			sent = follower.maybe_resend_last_target(hold_timeout_s=lost_face_hold_s)
			cv2.putText(
				frame,
				f"No face detected (hold={sent}) invert_x={follower.invert_x}",
				(10, 28),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 200, 255),
				2,
			)

		cv2.imshow("Furhat Face Follow", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("m"):
			follower.invert_x = not follower.invert_x
			print(f"invert_x set to {follower.invert_x}")
		if key == ord("q"):
			break

	cam.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
