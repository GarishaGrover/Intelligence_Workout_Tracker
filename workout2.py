import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import json
import os
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import threading

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, width=200, height=50, bg_color="#4CAF50", hover_color="#45a049", text_color="white", font=("Arial", 12, "bold")):
        super().__init__(parent, width=width, height=height, highlightthickness=0)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = font
        self.text = text
        
        # Draw rounded rectangle
        self.rect = self.create_round_rect(0, 0, width, height, radius=25, fill=bg_color)
        self.text_id = self.create_text(width/2, height/2, text=text, fill=text_color, font=font)
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        
    def create_round_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                 x2-radius, y1,
                 x2, y1,
                 x2, y1+radius,
                 x2, y2-radius,
                 x2, y2,
                 x2-radius, y2,
                 x1+radius, y2,
                 x1, y2,
                 x1, y2-radius,
                 x1, y1+radius,
                 x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)
    
    def on_enter(self, event):
        self.itemconfig(self.rect, fill=self.hover_color)
        
    def on_leave(self, event):
        self.itemconfig(self.rect, fill=self.bg_color)
        
    def on_click(self, event):
        self.command()

class UserManager:
    def __init__(self):
        self.users_file = "users.json"
        self.users = self.load_users()
    
    def load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=4)
    
    def register_user(self, user_id, name, age, weight, height, fitness_level):
        if user_id in self.users:
            return False
        
        self.users[user_id] = {
            "name": name,
            "age": age,
            "weight": weight,
            "height": height,
            "fitness_level": fitness_level,
            "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "workout_history": []
        }
        self.save_users()
        return True
    
    def get_user(self, user_id):
        return self.users.get(user_id)
    
    def update_user_workout(self, user_id, workout_data):
        if user_id in self.users:
            self.users[user_id]["workout_history"].append(workout_data)
            self.save_users()

class ExerciseAnalyzer:
    def __init__(self, user_manager):
        self.user_manager = user_manager
        self.current_user = None
        self.current_exercise = None
        self.workout_start_time = None
        
        # Initialize models only when needed
        self.pose = None
        self.face_detection = None
        self.face_mesh = None
        
        # Exercise variables
        self.available_exercises = {
            "pushup": {"name": "Push-ups", "type": "strength", "muscles": ["chest", "triceps", "shoulders"]},
            "bicep": {"name": "Bicep Curls", "type": "strength", "muscles": ["biceps"]},
            "tricep": {"name": "Tricep Extensions", "type": "strength", "muscles": ["triceps"]},
            "squat": {"name": "Squats", "type": "legs", "muscles": ["quads", "glutes", "hamstrings"]},
            "shoulder_press": {"name": "Shoulder Press", "type": "strength", "muscles": ["shoulders", "triceps"]},
            "plank": {"name": "Plank", "type": "core", "muscles": ["core", "shoulders"]},
            "head": {"name": "Head Movement", "type": "rehab", "muscles": ["neck"]}
        }
        
        # Performance optimization
        self.last_processing_time = time.time()
        self.processing_interval = 0.033  # ~30 FPS
        
        # Exercise state
        self.reset_exercise_state()
        
        # CSV logging
        self.csv_file = "workout_records.csv"
        self.ensure_csv_header()
    
    def ensure_csv_header(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'User_ID', 'User_Name', 'Exercise', 'Repetitions', 
                    'Avg_Time_per_Rep', 'Duration', 'Calories_Burned', 'Feedback'
                ])
    
    def save_workout_data(self, duration):
        if self.current_user and self.current_exercise:
            user_data = self.user_manager.get_user(self.current_user)
            if not user_data:
                return
            
            # For bicep curls, sum both arms
            total_reps = self.counter
            if self.current_exercise == "bicep":
                total_reps = self.left_counter + self.right_counter
            
            avg_time = sum(self.rep_times) / len(self.rep_times) if self.rep_times else 0
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Estimate calories burned (very rough estimation)
            calories = self.estimate_calories_burned(user_data, duration)
            
            record = [
                timestamp, self.current_user, user_data["name"],
                self.available_exercises[self.current_exercise]["name"],
                total_reps, f"{avg_time:.2f}s", f"{duration:.2f}s",
                f"{calories:.2f}", self.feedback
            ]
            
            # Save to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(record)
            
            # Update user history
            workout_data = {
                "date": timestamp,
                "exercise": self.current_exercise,
                "reps": total_reps,
                "duration": duration,
                "calories": calories
            }
            self.user_manager.update_user_workout(self.current_user, workout_data)
    
    def estimate_calories_burned(self, user_data, duration):
        # Very rough estimation based on MET values
        met_values = {
            "pushup": 8, "bicep": 3, "tricep": 3, "squat": 5, 
            "shoulder_press": 3, "plank": 3, "head": 1
        }
        
        met = met_values.get(self.current_exercise, 3)
        weight_kg = float(user_data["weight"])
        
        # Calories = MET * weight(kg) * time(hours)
        calories = met * weight_kg * (duration / 3600)
        return calories
    
    def initialize_models(self):
        """Initialize models only when needed to save resources"""
        if not self.pose and self.current_exercise != "head":
            self.pose = mp_pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                model_complexity=1  # Lower complexity for better performance
            )
        
        if not self.face_detection and self.current_exercise == "head":
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.7
            )
        
        if not self.face_mesh and self.current_exercise == "head":
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
    
    def release_models(self):
        """Release models when not needed"""
        if self.pose:
            self.pose.close()
            self.pose = None
        
        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None
        
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
    
    def reset_exercise_state(self):
        self.counter = 0
        self.left_counter = 0
        self.right_counter = 0
        self.left_stage = None
        self.right_stage = None
        self.stage = None
        self.rep_start_time = None
        self.rep_times = []
        self.feedback = "Get ready to start"
        self.angle_buffer = deque(maxlen=5)
        self.head_position_history = deque(maxlen=10)
        self.head_movement_count = 0
        self.head_movement_direction = None
    
    def set_current_user(self, user_id):
        self.current_user = user_id
        user_data = self.user_manager.get_user(user_id)
        if user_data:
            print(f"User set to: {user_data['name']}")
    
    def set_current_exercise(self, exercise):
        # Release previous models
        self.release_models()
        
        if exercise in self.available_exercises:
            self.current_exercise = exercise
            self.reset_exercise_state()
            self.initialize_models()  # Initialize only needed models
            print(f"Exercise set to: {self.available_exercises[exercise]['name']}")
            return True
        return False
    
    def start_workout(self):
        self.workout_start_time = time.time()
        self.reset_exercise_state()
        print("Workout started!")
    
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def analyze_pushup(self, landmarks):
        try:
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = self.calculate_angle(hip, shoulder, elbow)
            torso_angle = self.calculate_angle(shoulder, hip, knee)
            
            self.angle_buffer.append(elbow_angle)
            smoothed_angle = sum(self.angle_buffer) / len(self.angle_buffer)
            
            # Push-up logic
            if smoothed_angle < 90:
                if self.stage == "up" or self.stage is None:
                    self.counter += 1
                    if self.rep_start_time:
                        rep_time = time.time() - self.rep_start_time
                        self.rep_times.append(rep_time)
                    self.rep_start_time = time.time()
                self.stage = "down"
                self.feedback = "Good form: Lower position"
            elif smoothed_angle > 160:
                self.stage = "up"
                self.feedback = "Good form: Push up!"
            else:
                self.feedback = f"Adjust form: Elbow angle {int(smoothed_angle)}°"
            
            if torso_angle > 190 or torso_angle < 170:
                self.feedback = "Keep your body straight!"
            
            return elbow_angle, shoulder_angle, torso_angle
        except Exception as e:
            return 0, 0, 0
    
    def analyze_bicep_curl(self, landmarks):
        try:
            # Left arm
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            l_elbow_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            
            # Right arm
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            r_elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            
            # Left arm logic
            if l_elbow_angle > 160:
                self.left_stage = "down"
            elif l_elbow_angle < 50:
                if self.left_stage == "down" or self.left_stage is None:
                    self.left_counter += 1
                    if self.rep_start_time:
                        rep_time = time.time() - self.rep_start_time
                        self.rep_times.append(rep_time)
                    self.rep_start_time = time.time()
                self.left_stage = "up"
            
            # Right arm logic
            if r_elbow_angle > 160:
                self.right_stage = "down"
            elif r_elbow_angle < 50:
                if self.right_stage == "down" or self.right_stage is None:
                    self.right_counter += 1
                    if self.rep_start_time:
                        rep_time = time.time() - self.rep_start_time
                        self.rep_times.append(rep_time)
                    self.rep_start_time = time.time()
                self.right_stage = "up"
            
            # Form feedback
            feedback_messages = []
            
            # Check for elbow movement
            l_shoulder_to_elbow = np.linalg.norm(np.array(l_shoulder) - np.array(l_elbow))
            r_shoulder_to_elbow = np.linalg.norm(np.array(r_shoulder) - np.array(r_elbow))
            
            if l_shoulder_to_elbow > 0.15 or r_shoulder_to_elbow > 0.15:
                feedback_messages.append("Keep your elbows stationary!")
            
            # Check for full range of motion
            if l_elbow_angle > 120 and self.left_stage == "down":
                feedback_messages.append("Left arm: Extend fully at the bottom")
            if r_elbow_angle > 120 and self.right_stage == "down":
                feedback_messages.append("Right arm: Extend fully at the bottom")
            
            # Check for proper contraction
            if l_elbow_angle > 70 and self.left_stage == "up":
                feedback_messages.append("Left arm: Squeeze bicep at the top")
            if r_elbow_angle > 70 and self.right_stage == "up":
                feedback_messages.append("Right arm: Squeeze bicep at the top")
            
            # Set feedback message
            if feedback_messages:
                self.feedback = " | ".join(feedback_messages)
            else:
                self.feedback = f"Left: {self.left_counter} | Right: {self.right_counter}"
            
            return l_elbow_angle, r_elbow_angle, 0
        except Exception as e:
            return 0, 0, 0
    
    def analyze_squat(self, landmarks):
        try:
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            knee_angle = self.calculate_angle(hip, knee, ankle)
            torso_angle = self.calculate_angle(shoulder, hip, knee)
            
            self.angle_buffer.append(knee_angle)
            smoothed_angle = sum(self.angle_buffer) / len(self.angle_buffer)
            
            # Squat logic
            if smoothed_angle > 160:
                self.stage = "up"
            elif smoothed_angle < 90:
                if self.stage == "up" or self.stage is None:
                    self.counter += 1
                    if self.rep_start_time:
                        rep_time = time.time() - self.rep_start_time
                        self.rep_times.append(rep_time)
                    self.rep_start_time = time.time()
                self.stage = "down"
            
            # Form feedback
            feedback_messages = []
            
            # Check torso position
            if torso_angle < 160:
                feedback_messages.append("Keep your chest up and back straight!")
            
            # Check knee alignment
            knee_x, ankle_x = knee[0], ankle[0]
            if abs(knee_x - ankle_x) > 0.05:
                feedback_messages.append("Keep your knees aligned with your ankles!")
            
            # Check depth
            if smoothed_angle > 100 and self.stage == "down":
                feedback_messages.append("Go deeper! Aim for at least 90° knee bend")
            
            # Check hip position
            if hip[1] < knee[1] and self.stage == "down":
                feedback_messages.append("Hips should go below knee level for full squat")
            
            # Set feedback message
            if feedback_messages:
                self.feedback = " | ".join(feedback_messages)
            else:
                self.feedback = f"Good form! Reps: {self.counter}"
            
            return knee_angle, torso_angle, 0
        except Exception as e:
            return 0, 0, 0
    
    def analyze_shoulder_press(self, landmarks):
        try:
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            
            self.angle_buffer.append(elbow_angle)
            smoothed_angle = sum(self.angle_buffer) / len(self.angle_buffer)
            
            # Shoulder press logic
            if smoothed_angle > 160:
                self.stage = "up"
            elif smoothed_angle < 90:
                if self.stage == "up" or self.stage is None:
                    self.counter += 1
                    if self.rep_start_time:
                        rep_time = time.time() - self.rep_start_time
                        self.rep_times.append(rep_time)
                    self.rep_start_time = time.time()
                self.stage = "down"
            
            # Form feedback
            feedback_messages = []
            
            # Check for full extension
            if smoothed_angle < 170 and self.stage == "up":
                feedback_messages.append("Extend arms fully at the top!")
            
            # Check elbow position
            if elbow[0] < shoulder[0] - 0.05:  # Elbows too far back
                feedback_messages.append("Bring elbows slightly forward")
            
            # Check wrist position
            if wrist[0] < elbow[0]:  # Wrist behind elbow
                feedback_messages.append("Keep wrists straight under elbows")
            
            # Set feedback message
            if feedback_messages:
                self.feedback = " | ".join(feedback_messages)
            else:
                self.feedback = f"Good form! Reps: {self.counter}"
            
            return elbow_angle, 0, 0
        except Exception as e:
            return 0, 0, 0
    
    def analyze_plank(self, landmarks):
        try:
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate body alignment angles
            torso_angle = self.calculate_angle(shoulder, hip, knee)
            hip_angle = self.calculate_angle(shoulder, hip, ankle)
            
            # Plank is a timed exercise, not reps
            current_time = time.time()
            if self.rep_start_time is None:
                self.rep_start_time = current_time
                self.counter = 0
            
            elapsed = current_time - self.rep_start_time
            self.counter = int(elapsed)  # Count seconds
            
            # Form feedback
            feedback_messages = []
            
            if torso_angle < 170:
                feedback_messages.append("Hips too high! Lower them to form a straight line")
            elif torso_angle > 190:
                feedback_messages.append("Hips sagging! Engage your core to lift them")
            
            # Check shoulder position
            if shoulder[1] > hip[1] + 0.05:  # Shoulders higher than hips
                feedback_messages.append("Push through your shoulders to create a straight line")
            
            # Set feedback message
            if feedback_messages:
                self.feedback = " | ".join(feedback_messages)
            else:
                self.feedback = f"Good form! Hold for {self.counter}s"
            
            return torso_angle, hip_angle, 0
        except Exception as e:
            return 0, 0, 0
    
    def process_frame(self, image):
        current_time = time.time()
        if current_time - self.last_processing_time < self.processing_interval:
            return image, self.feedback, self.counter
        
        self.last_processing_time = current_time
        
        if not self.pose and self.current_exercise != "head":
            return image, "Models not initialized", 0
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process for pose detection
        pose_results = self.pose.process(image_rgb) if self.pose else None
        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        angle1, angle2, angle3 = 0, 0, 0
        
        # Exercise analysis based on current selection
        if pose_results and pose_results.pose_landmarks and self.current_exercise != "head":
            mp_drawing.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            try:
                if self.current_exercise == "pushup":
                    angle1, angle2, angle3 = self.analyze_pushup(pose_results.pose_landmarks.landmark)
                elif self.current_exercise == "bicep":
                    angle1, angle2, angle3 = self.analyze_bicep_curl(pose_results.pose_landmarks.landmark)
                elif self.current_exercise == "squat":
                    angle1, angle2, angle3 = self.analyze_squat(pose_results.pose_landmarks.landmark)
                elif self.current_exercise == "shoulder_press":
                    angle1, angle2, angle3 = self.analyze_shoulder_press(pose_results.pose_landmarks.landmark)
                elif self.current_exercise == "plank":
                    angle1, angle2, angle3 = self.analyze_plank(pose_results.pose_landmarks.landmark)
            except Exception as e:
                pass
        
        # Display exercise information
        user_text = f"User: {self.current_user}" if self.current_user else "No user selected"
        exercise_text = f"Exercise: {self.available_exercises[self.current_exercise]['name']}" if self.current_exercise else "No exercise selected"
        
        cv2.putText(image, user_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, exercise_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display reps based on exercise type
        if self.current_exercise == "bicep":
            reps_text = f"Left: {self.left_counter} | Right: {self.right_counter}"
        else:
            reps_text = f"Reps/Time: {self.counter}"
            
        cv2.putText(image, reps_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display feedback with word wrapping
        y_offset = 120
        max_line_width = 60
        words = self.feedback.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_line_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for i, line in enumerate(lines):
            cv2.putText(image, line, (10, y_offset + i*20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Display angles if relevant
        if angle1 > 0:
            cv2.putText(image, f"Angle: {int(angle1)}", (10, y_offset + len(lines)*20 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image, self.feedback, self.counter

    def run_workout(self, duration_minutes):
        """Run the workout with webcam"""
        duration_seconds = duration_minutes * 60
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam. Please check your camera connection.")
            return
        
        # Start workout
        self.start_workout()
        workout_start_time = time.time()
        
        print("Workout started! Press 'q' to quit early.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame, feedback, counter = self.process_frame(frame)
            
            # Display timer
            elapsed = time.time() - workout_start_time
            if duration_seconds > 0:
                remaining = max(0, duration_seconds - elapsed)
                timer_text = f"Time: {int(elapsed)}s / Remaining: {int(remaining)}s"
                if remaining <= 0:
                    break
            else:
                timer_text = f"Time: {int(elapsed)}s"
            
            cv2.putText(processed_frame, timer_text, (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Advanced Gym Workout Analyzer', processed_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save workout data
        workout_duration_actual = time.time() - workout_start_time
        self.save_workout_data(workout_duration_actual)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show workout summary
        user_data = self.user_manager.get_user(self.current_user)
        if user_data:
            # For bicep curls, sum both arms
            total_reps = self.counter
            if self.current_exercise == "bicep":
                total_reps = self.left_counter + self.right_counter
                
            summary = f"""
            Workout Summary for {user_data['name']}:
            Exercise: {self.available_exercises[self.current_exercise]['name']}
            Duration: {workout_duration_actual:.2f} seconds
            Reps/Time: {total_reps}
            Estimated Calories: {self.estimate_calories_burned(user_data, workout_duration_actual):.2f}
            """
            messagebox.showinfo("Workout Complete", summary)

class GymApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Gym Trainer")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")
        
        # Center the window
        self.center_window()
        
        # Initialize user manager
        self.user_manager = UserManager()
        self.analyzer = None
        self.current_user = None
        
        self.setup_ui()
        
    def center_window(self):
        self.root.update_idletasks()
        width = 900
        height = 700
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def setup_ui(self):
        # Create main frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg="#2c3e50")
        header_frame.pack(pady=(0, 30))
        
        title_label = tk.Label(header_frame, text="🏋️ AI GYM TRAINER", 
                              font=("Arial", 28, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(header_frame, text="Smart Workout Analysis System", 
                                 font=("Arial", 14), fg="#ecf0f1", bg="#2c3e50")
        subtitle_label.pack()
        
        # Content frame
        content_frame = tk.Frame(main_frame, bg="#34495e", relief=tk.RAISED, bd=2)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Welcome message
        welcome_text = """Welcome to the AI Gym Trainer!

This advanced system uses computer vision to:
• Track your exercise form in real-time
• Count repetitions accurately
• Provide feedback on your technique
• Track your progress over time
• Personalize workouts based on your fitness level

Get started by selecting an option below:"""
        
        welcome_label = tk.Label(content_frame, text=welcome_text, font=("Arial", 12), 
                                fg="#ecf0f1", bg="#34495e", justify=tk.LEFT, wraplength=600)
        welcome_label.pack(pady=30, padx=20)
        
        # Button frame
        button_frame = tk.Frame(content_frame, bg="#34495e")
        button_frame.pack(pady=30)
        
        # Register button
        register_btn = ModernButton(button_frame, "📝 Register New User", 
                                   self.register_user, bg_color="#3498db", hover_color="#2980b9")
        register_btn.pack(pady=15)
        
        # Select user button
        select_btn = ModernButton(button_frame, "👤 Select Existing User", 
                                 self.select_user, bg_color="#e74c3c", hover_color="#c0392b")
        select_btn.pack(pady=15)
        
        # Quick start button (for demo)
        quick_btn = ModernButton(button_frame, "⚡ Quick Start Demo", 
                                self.quick_start, bg_color="#9b59b6", hover_color="#8e44ad")
        quick_btn.pack(pady=15)
        
        # Footer
        footer_frame = tk.Frame(main_frame, bg="#2c3e50")
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        footer_label = tk.Label(footer_frame, text="© 2023 AI Gym Trainer | Advanced Computer Vision Workout System", 
                               font=("Arial", 10), fg="#bdc3c7", bg="#2c3e50")
        footer_label.pack()
    
    def register_user(self):
        self.open_registration_window()
    
    def select_user(self):
        if not self.user_manager.users:
            messagebox.showinfo("No Users", "No users registered yet. Please register first.")
            return
        
        self.open_user_selection_window()
    
    def quick_start(self):
        # Create a temporary user for demo
        demo_id = "demo_" + datetime.now().strftime("%H%M%S")
        self.user_manager.register_user(demo_id, "Demo User", "25", "70", "175", "beginner")
        self.current_user = demo_id
        self.open_exercise_selection()
    
    def open_registration_window(self):
        reg_window = tk.Toplevel(self.root)
        reg_window.title("User Registration")
        reg_window.geometry("500x600")
        reg_window.configure(bg="#34495e")
        reg_window.grab_set()
        
        # Center the window
        reg_window.update_idletasks()
        width = reg_window.winfo_width()
        height = reg_window.winfo_height()
        x = (reg_window.winfo_screenwidth() // 2) - (width // 2)
        y = (reg_window.winfo_screenheight() // 2) - (height // 2)
        reg_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Registration form
        title_label = tk.Label(reg_window, text="👤 New User Registration", 
                              font=("Arial", 20, "bold"), fg="white", bg="#34495e")
        title_label.pack(pady=20)
        
        form_frame = tk.Frame(reg_window, bg="#34495e")
        form_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
        
        fields = [
            ("User ID:", "entry_user_id"),
            ("Full Name:", "entry_name"),
            ("Age:", "entry_age"),
            ("Weight (kg):", "entry_weight"),
            ("Height (cm):", "entry_height"),
            ("Fitness Level:", "combo_fitness")
        ]
        
        entries = {}
        
        for i, (label_text, field_name) in enumerate(fields):
            frame = tk.Frame(form_frame, bg="#34495e")
            frame.pack(fill=tk.X, pady=8)
            
            label = tk.Label(frame, text=label_text, font=("Arial", 12), 
                           fg="white", bg="#34495e", width=15, anchor=tk.W)
            label.pack(side=tk.LEFT)
            
            if field_name == "combo_fitness":
                combo = ttk.Combobox(frame, values=["Beginner", "Intermediate", "Advanced"], 
                                    state="readonly", font=("Arial", 12))
                combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
                combo.set("Beginner")
                entries[field_name] = combo
            else:
                entry = tk.Entry(frame, font=("Arial", 12))
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
                entries[field_name] = entry
        
        # Button frame
        btn_frame = tk.Frame(reg_window, bg="#34495e")
        btn_frame.pack(pady=20)
        
        def submit_registration():
            user_id = entries["entry_user_id"].get()
            name = entries["entry_name"].get()
            age = entries["entry_age"].get()
            weight = entries["entry_weight"].get()
            height = entries["entry_height"].get()
            fitness = entries["combo_fitness"].get()
            
            if not all([user_id, name, age, weight, height, fitness]):
                messagebox.showerror("Error", "Please fill all fields")
                return
            
            if self.user_manager.register_user(user_id, name, age, weight, height, fitness.lower()):
                messagebox.showinfo("Success", f"User {name} registered successfully!")
                self.current_user = user_id
                reg_window.destroy()
                self.open_exercise_selection()
            else:
                messagebox.showerror("Error", "User ID already exists")
        
        submit_btn = ModernButton(btn_frame, "✅ Complete Registration", submit_registration,
                                 bg_color="#27ae60", hover_color="#229954")
        submit_btn.pack(pady=10)
        
        cancel_btn = ModernButton(btn_frame, "❌ Cancel", reg_window.destroy,
                                 bg_color="#e74c3c", hover_color="#c0392b")
        cancel_btn.pack(pady=5)
    
    def open_user_selection_window(self):
        sel_window = tk.Toplevel(self.root)
        sel_window.title("Select User")
        sel_window.geometry("500x400")
        sel_window.configure(bg="#34495e")
        sel_window.grab_set()
        
        # Center the window
        sel_window.update_idletasks()
        width = sel_window.winfo_width()
        height = sel_window.winfo_height()
        x = (sel_window.winfo_screenwidth() // 2) - (width // 2)
        y = (sel_window.winfo_screenheight() // 2) - (height // 2)
        sel_window.geometry(f"{width}x{height}+{x}+{y}")
        
        title_label = tk.Label(sel_window, text="👥 Select User", 
                              font=("Arial", 20, "bold"), fg="white", bg="#34495e")
        title_label.pack(pady=20)
        
        # Create a frame for the listbox with scrollbar
        list_frame = tk.Frame(sel_window, bg="#34495e")
        list_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
        
        # Listbox with users
        listbox = tk.Listbox(list_frame, font=("Arial", 12), selectmode=tk.SINGLE)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox
        for user_id, user_data in self.user_manager.users.items():
            listbox.insert(tk.END, f"{user_id} - {user_data['name']} ({user_data['fitness_level']})")
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame
        btn_frame = tk.Frame(sel_window, bg="#34495e")
        btn_frame.pack(pady=20)
        
        def select_user():
            selection = listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select a user")
                return
            
            selected_index = selection[0]
            user_id = list(self.user_manager.users.keys())[selected_index]
            self.current_user = user_id
            sel_window.destroy()
            self.open_exercise_selection()
        
        select_btn = ModernButton(btn_frame, "✅ Select User", select_user,
                                 bg_color="#3498db", hover_color="#2980b9")
        select_btn.pack(pady=10)
        
        cancel_btn = ModernButton(btn_frame, "❌ Cancel", sel_window.destroy,
                                 bg_color="#e74c3c", hover_color="#c0392b")
        cancel_btn.pack(pady=5)
    
    def open_exercise_selection(self):
        ex_window = tk.Toplevel(self.root)
        ex_window.title("Select Exercise")
        ex_window.geometry("600x500")
        ex_window.configure(bg="#34495e")
        ex_window.grab_set()
        
        # Center the window
        ex_window.update_idletasks()
        width = ex_window.winfo_width()
        height = ex_window.winfo_height()
        x = (ex_window.winfo_screenwidth() // 2) - (width // 2)
        y = (ex_window.winfo_screenheight() // 2) - (height // 2)
        ex_window.geometry(f"{width}x{height}+{x}+{y}")
        
        title_label = tk.Label(ex_window, text="💪 Select Exercise", 
                              font=("Arial", 20, "bold"), fg="white", bg="#34495e")
        title_label.pack(pady=20)
        
        user_data = self.user_manager.get_user(self.current_user)
        user_info = f"User: {user_data['name']} | Level: {user_data['fitness_level'].capitalize()}"
        user_label = tk.Label(ex_window, text=user_info, font=("Arial", 12), 
                             fg="#ecf0f1", bg="#34495e")
        user_label.pack(pady=5)
        
        # Exercise buttons frame
        exercises_frame = tk.Frame(ex_window, bg="#34495e")
        exercises_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
        
        exercises = [
            ("🏋️ Push-ups", "pushup"),
            ("💪 Bicep Curls", "bicep"),
            ("🦵 Squats", "squat"),
            ("✊ Shoulder Press", "shoulder_press"),
            ("🧘 Plank", "plank"),
            ("🧠 Head Movement", "head")
        ]
        
        for ex_name, ex_key in exercises:
            btn = ModernButton(exercises_frame, ex_name, 
                              lambda k=ex_key: self.start_workout(k, ex_window),
                              width=300, height=45, bg_color="#9b59b6", hover_color="#8e44ad")
            btn.pack(pady=8)
        
        # Duration selection
        duration_frame = tk.Frame(ex_window, bg="#34495e")
        duration_frame.pack(pady=15)
        
        duration_label = tk.Label(duration_frame, text="Workout Duration (minutes):", 
                                 font=("Arial", 12), fg="white", bg="#34495e")
        duration_label.pack(side=tk.LEFT)
        
        self.duration_var = tk.StringVar(value="5")
        duration_spin = tk.Spinbox(duration_frame, from_=1, to=60, textvariable=self.duration_var,
                                  font=("Arial", 12), width=5)
        duration_spin.pack(side=tk.LEFT, padx=10)
    
    def start_workout(self, exercise_key, window):
        window.destroy()
        duration = int(self.duration_var.get()) if self.duration_var.get() else 5
        
        # Initialize analyzer and start workout
        self.analyzer = ExerciseAnalyzer(self.user_manager)
        self.analyzer.set_current_user(self.current_user)
        self.analyzer.set_current_exercise(exercise_key)
        
        # Start the workout in a new thread to keep UI responsive
        thread = threading.Thread(target=self.analyzer.run_workout, args=(duration,))
        thread.daemon = True
        thread.start()

def main():
    root = tk.Tk()
    app = GymApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()