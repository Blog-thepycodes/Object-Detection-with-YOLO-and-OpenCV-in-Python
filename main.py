import torch
import cv2
import pandas as pd
from datetime import datetime
import os
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import threading
import signal
import sys




# Load YOLOv5s model for real-time video detection
model_realtime = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s for real-time detection
# Load YOLOv5x model for better accuracy in file processing
model_file = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # YOLOv5x for video and photo files




# Initialize variables
confidence_threshold = 0.3  # Confidence threshold for detections
frames_dir = 'detected_frames'
os.makedirs(frames_dir, exist_ok=True)




# Log file setup
log_file = 'detection_log.csv'
log_columns = ['Timestamp', 'Object', 'Confidence', 'Frame']
log_data = []




cap = None




def release_camera():
  """Release the camera resource."""
  global cap
  if cap is not None:
      cap.release()
      cv2.destroyAllWindows()
      cap = None




def signal_handler(sig, frame):
  """Handle termination signals to ensure resources are released."""
  release_camera()
  sys.exit(0)




signal.signal(signal.SIGINT, signal_handler)




def process_frame(frame, frame_width, frame_height, frame_count, model):
  """Process a single frame for object detection."""
  results = model(frame)
  labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
  n = len(labels)
  detected = False
  for i in range(n):
      row = cords[i]
      if row[4] >= confidence_threshold:
          x1, y1, x2, y2 = int(row[0] * frame_width), int(row[1] * frame_height), int(row[2] * frame_width), int(row[3] * frame_height)
          bgr = (0, 255, 0)
          cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
          text = f"{model.names[int(labels[i])]} {row[4]:.2f}"
          cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr, 2)
          timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
          log_data.append([timestamp, model.names[int(labels[i])], row[4], frame_count])
          detected = True
  return frame, detected




def process_video(video_path, progress_bar):
  """Process a video file for object detection."""
  cap = cv2.VideoCapture(video_path)
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  output_video_path = 'output.avi'
  out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))




  frame_count = 0
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      frame, _ = process_frame(frame, frame_width, frame_height, frame_count, model_file)
      out.write(frame)
      frame_count += 1
      progress = (frame_count / total_frames) * 100
      progress_bar['value'] = progress
      progress_bar.update_idletasks()




  cap.release()
  out.release()
  messagebox.showinfo("Info", f"Processed video saved to {output_video_path}")
  progress_bar['value'] = 0




def process_photo(photo_path):
  """Process a photo file for object detection."""
  frame = cv2.imread(photo_path)
  if frame is None:
      messagebox.showerror("Error", "Could not open or find the image.")
      return
  frame_height, frame_width, _ = frame.shape
  frame, detected = process_frame(frame, frame_width, frame_height, 0, model_file)
  max_display_size = 800
  scale = min(max_display_size / frame_width, max_display_size / frame_height)
  display_frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
  cv2.imshow('Processed Photo - The Pycodes', display_frame)
  cv2.waitKey(0)
  cv2.destroyAllWindows()




  if detected:
      frame_path = os.path.join(frames_dir, os.path.basename(photo_path))
      cv2.imwrite(frame_path, frame)
      messagebox.showinfo("Info", f"Processed photo saved at {frame_path}")
  else:
      messagebox.showinfo("Info", "No objects detected in the photo.")




def start_realtime_detection():
  """Start real-time video detection."""
  def run():
      global cap
      cap = cv2.VideoCapture(0)
      if not cap.isOpened():
          messagebox.showerror("Error", "Could not open video.")
          return
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      frame_width = int(cap.get(3))
      frame_height = int(cap.get(4))
      output_video_path = 'output.avi'
      out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
      frame_count = 0




      while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
              break
          frame, detected = process_frame(frame, frame_width, frame_height, frame_count, model_realtime)
          out.write(frame)
          cv2.imshow('YOLOv5 Object Detection - The Pycodes', frame)
          key = cv2.waitKey(1) & 0xFF
          if key == ord('q'):
              break
          elif key == ord('s'):
              frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
              cv2.imwrite(frame_path, frame)
              print(f"Frame {frame_count} saved at {frame_path}")
          frame_count += 1




      release_camera()
      out.release()
      messagebox.showinfo("Info", f"Annotated video saved to {output_video_path}")




  threading.Thread(target=run).start()




def start_video_processing():
  """Start video file processing."""
  video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
  if video_path:
      progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
      progress_bar.pack(pady=10)
      threading.Thread(target=process_video, args=(video_path, progress_bar)).start()




# Tkinter GUI setup
root = Tk()
root.title("YOLOv5 Object Detection - The Pycodes")
root.geometry("400x300")




btn_realtime = Button(root, text="Real-time Video Detection", command=start_realtime_detection)
btn_realtime.pack(pady=10)




btn_video = Button(root, text="Process Video File", command=start_video_processing)
btn_video.pack(pady=10)




btn_photo = Button(root, text="Process Photo File", command=lambda: threading.Thread(target=process_photo, args=(filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]),)).start())
btn_photo.pack(pady=10)




btn_exit = Button(root, text="Exit", command=root.quit)
btn_exit.pack(pady=10)




root.mainloop()




log_df = pd.DataFrame(log_data, columns=log_columns)
log_df.to_csv(log_file, index=False)
print(f"Detection log saved to {log_file}")
print(f"Detected frames saved to {frames_dir}")
