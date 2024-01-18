from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
import threading
import time


class AccuracyPose:
    def __init__(self, animation, l2):
        self.animation = animation
        self.l2 = l2
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.stop_animation = False

    def calculate_pose_similarity(self, pose1, pose2):
        left_hip_index = 23
        right_hip_index = 24
        landmark_weights = np.ones(33)
        landmark_weights[[11, 12, 13, 14, 23, 24, 25, 26]] = 2
        pose1 = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose1.landmark])
        pose2 = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose2.landmark])

        torso_center1 = (pose1[left_hip_index] + pose1[right_hip_index]) / 2
        torso_center2 = (pose2[left_hip_index] + pose2[right_hip_index]) / 2
        normalized_pose1 = pose1 - torso_center1
        normalized_pose2 = pose2 - torso_center2
        distances = np.linalg.norm(normalized_pose1 - normalized_pose2, axis=1)
        weighted_mean_distance = np.average(distances, weights=landmark_weights)

        similarity = 1 - weighted_mean_distance
        return similarity

    def estimate_pose(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks

    def animationpose(self, cap):
        print('animationpose called')
        animation_video = cv2.VideoCapture(self.animation)
        q_pressed = False

        # Create a label to display the similarity score
        similarity_label = Label(f2, font=('Arial', 20), bg='lightblue', fg='black')
        similarity_label.place(relx=0.4, rely=0.8)

        # Create a list to store the similarity scores
        similarity_scores = []

        # Record the start time
        start_time = time.time()


        def on_key_press(event):
            nonlocal q_pressed
            if event.char == 'q':
                q_pressed = True

        root.bind('<KeyPress>', on_key_press)

        def update_frame():
            nonlocal q_pressed
            ret, frame = cap.read()
            if not ret:
                print('no frame from webcam')
                cap.release()
                cv2.destroyAllWindows()
                return
            current_pose = self.estimate_pose(frame)
            if current_pose is not None:
                ret_animation, frame_animation = animation_video.read()
                if not ret_animation:
                    print('no frame from animation video')
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                animation_pose = self.estimate_pose(frame_animation)
                if animation_pose is not None:
                    similarity = self.calculate_pose_similarity(current_pose, animation_pose)
                    similarity_percentage = round(similarity * 100, 2)
                    # print(f'{similarity_percentage}%')

                    # Update the similarity label text
                    similarity_label.config(text=f'Similarity: {similarity_percentage}%')

                    # Add the similarity score to the list
                    similarity_scores.append(similarity_percentage)

                # Draw pose landmarks and connections on the camera frame
                mp.solutions.drawing_utils.draw_landmarks(frame, current_pose, mp.solutions.pose.POSE_CONNECTIONS)

                # Check if the l2 label still exists before accessing its properties
                if l2.winfo_exists():
                    # Resize the image to fit the l2 label
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    img = img.resize((int(self.l2.winfo_width()), int(self.l2.winfo_height())), Image.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.l2.imgtk = imgtk
                    self.l2.configure(image=imgtk)

        while not q_pressed and not self.stop_animation:
            update_frame()
            time.sleep(0.01)
        stats_window = Toplevel(f2)
        stats_window.title('Statistics')

        # Create a new Text widget in the stats_window to display the statistics
        stats_text = Text(stats_window, font=('Arial', 10), bg='lightblue', fg='black')
        stats_text.pack(fill=BOTH, expand=True)

        # Calculate and display the full comparison results after the update_frame loop exits
        average_similarity = sum(similarity_scores) / len(similarity_scores)
        min_similarity = min(similarity_scores)
        max_similarity = max(similarity_scores)
        median_similarity = np.median(similarity_scores)
        std_similarity = np.std(similarity_scores)

        total_time = time.time() - start_time
        num_frames = len(similarity_scores)
        frame_rate = num_frames / total_time

        stats_text.insert(END, f'Average similarity: {average_similarity:.2f}%\n')
        stats_text.insert(END, f'Minimum similarity: {min_similarity:.2f}%\n')
        stats_text.insert(END, f'Maximum similarity: {max_similarity:.2f}%\n')
        stats_text.insert(END, f'Median similarity: {median_similarity:.2f}%\n')
        stats_text.insert(END, f'Standard deviation of similarity: {std_similarity:.2f}\n')

        stats_text.insert(END, f'Total running time: {total_time:.2f} seconds\n')
        stats_text.insert(END, f'Number of frames processed: {num_frames}\n')
        stats_text.insert(END, f'Average frame rate: {frame_rate:.2f} frames per second\n')
        if average_similarity >= 90:
            remark = 'Excellent! Your poses are very similar to the animation.'
        elif average_similarity >= 75:
            remark = 'Great job! Your poses are quite similar to the animation.'
        elif average_similarity >= 50:
            remark = 'Not bad! Keep practicing to improve your pose similarity.'
        else:
            remark = 'Keep practicing! Your poses could use some improvement.'

        stats_text.insert(END, f'\nRemark: {remark}\n')
        # Create a close button to close the stats_window and return to the f2 frame
        close_button = Button(stats_window, text='Close', command=stats_window.destroy)
        close_button.pack()

def ex1play():
    global accuracy
    cap = cv2.VideoCapture('e1.mkv')
    q_pressed = False

    def on_key_press(event):
        nonlocal q_pressed
        if event.char == 'q':
            q_pressed = True

    root.bind('<KeyPress>', on_key_press)

    def update_frame():
        global accuracy
        nonlocal q_pressed
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)
        img = img.resize((int(l1.winfo_width()), int(l1.winfo_height())), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        l1.imgtk = imgtk
        l1.configure(image=imgtk)
        if q_pressed or accuracy.stop_animation:
            f2.destroy()
            return
        l1.after(10, update_frame)

    update_frame()
def ex2play():
    global accuracy
    cap = cv2.VideoCapture('e2.mkv')
    q_pressed = False

    def on_key_press(event):
        nonlocal q_pressed
        if event.char == 'q':
            q_pressed = True

    root.bind('<KeyPress>', on_key_press)

    def update_frame():
        global accuracy
        nonlocal q_pressed
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)
        img = img.resize((int(l1.winfo_width()), int(l1.winfo_height())), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        l1.imgtk = imgtk
        l1.configure(image=imgtk)
        if q_pressed or accuracy.stop_animation:
            f2.destroy()
            return
        l1.after(10, update_frame)

    update_frame()
def ex3play():
    global accuracy
    cap = cv2.VideoCapture('e3.mkv')
    q_pressed = False

    def on_key_press(event):
        nonlocal q_pressed
        if event.char == 'q':
            q_pressed = True

    root.bind('<KeyPress>', on_key_press)

    def update_frame():
        global accuracy
        nonlocal q_pressed
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)
        img = img.resize((int(l1.winfo_width()), int(l1.winfo_height())), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        l1.imgtk = imgtk
        l1.configure(image=imgtk)
        if q_pressed or accuracy.stop_animation:
            f2.destroy()
            return
        l1.after(10, update_frame)

    update_frame()
def ex4play():
    global accuracy
    cap = cv2.VideoCapture('e4.mkv')
    q_pressed = False

    def on_key_press(event):
        nonlocal q_pressed
        if event.char == 'q':
            q_pressed = True

    root.bind('<KeyPress>', on_key_press)

    def update_frame():
        global accuracy
        nonlocal q_pressed
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)
        img = img.resize((int(l1.winfo_width()), int(l1.winfo_height())), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        l1.imgtk = imgtk
        l1.configure(image=imgtk)
        if q_pressed or accuracy.stop_animation:
            f2.destroy()
            return
        l1.after(10, update_frame)

    update_frame()
def ex5play():
    global accuracy
    cap = cv2.VideoCapture('e5.mkv')
    q_pressed = False

    def on_key_press(event):
        nonlocal q_pressed
        if event.char == 'q':
            q_pressed = True

    root.bind('<KeyPress>', on_key_press)

    def update_frame():
        global accuracy
        nonlocal q_pressed
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)
        img = img.resize((int(l1.winfo_width()), int(l1.winfo_height())), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        l1.imgtk = imgtk
        l1.configure(image=imgtk)
        if q_pressed or accuracy.stop_animation:
            f2.destroy()
            return
        l1.after(10, update_frame)

    update_frame()
def ex6play():
    global accuracy
    cap = cv2.VideoCapture('e6.mkv')
    q_pressed = False

    def on_key_press(event):
        nonlocal q_pressed
        if event.char == 'q':
            q_pressed = True

    root.bind('<KeyPress>', on_key_press)

    def update_frame():
        global accuracy
        nonlocal q_pressed
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)
        img = img.resize((int(l1.winfo_width()), int(l1.winfo_height())), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        l1.imgtk = imgtk
        l1.configure(image=imgtk)
        if q_pressed or accuracy.stop_animation:
            f2.destroy()
            return
        l1.after(10, update_frame)

    update_frame()
def ex1():
    global f2
    f2 = Frame(root, bg='lightblue')
    f2.place(relx=0.05, rely=0.15, relheight=0.65, relwidth=0.9)

    global l1
    l1 = Label(f2,text='exercise', bg='darkblue')
    l1.place(relx=0.025, rely=0.1, relheight=0.7, relwidth=0.45)

    global l2
    l2 = Label(f2,text='camera feed', bg='black', fg='white')
    l2.place(relx=0.525, rely=0.1, relheight=0.7, relwidth=0.45)

    global accuracy
    accuracy = AccuracyPose('e1.mkv', l2)

    def on_compare_click():
        print('on_compare_click called')

        cap = cv2.VideoCapture(0)

        countdown_label = Label(f2, font=('Arial', 50), bg='lightblue', fg='black')
        countdown_label.place(relx=0.48, rely=0.4)

        def countdown(count):
            if count > 0:
                countdown_label.config(text=str(count))
                f2.after(1000, countdown, count-1)
            else:
                countdown_label.destroy()
                ex1_thread = threading.Thread(target=ex1play)
                ex1_thread.start()
                animationpose_thread = threading.Thread(target=lambda: accuracy.animationpose(cap))
                animationpose_thread.start()

        countdown(5)

    def on_stop_click():
        global accuracy
        accuracy.stop_animation = True
        l1.destroy()
        l2.destroy()

    Button(f2, text='Compare', font=medium, bg='green', fg='white', command=on_compare_click).place(relx=0.4, rely=0.9, relheight=0.05, relwidth=0.2)
    Button(f2, text='Stop', font=medium, bg='red', fg='white', command=on_stop_click).place(relx=0.4, rely=0.95, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "q" to get Result', font=small).place(relx=0.05, rely=0.85, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "Compare" to Start', font=small).place(relx=0.05, rely=0.9, relheight=0.05, relwidth=0.2)

def ex2():
    global f2
    f2 = Frame(root, bg='lightblue')
    f2.place(relx=0.05, rely=0.15, relheight=0.65, relwidth=0.9)

    global l1
    l1 = Label(f2,text='exercise', bg='darkblue')
    l1.place(relx=0.025, rely=0.1, relheight=0.7, relwidth=0.45)

    global l2
    l2 = Label(f2,text='camera feed', bg='black', fg='white')
    l2.place(relx=0.525, rely=0.1, relheight=0.7, relwidth=0.45)

    global accuracy
    accuracy = AccuracyPose('e2.mkv', l2)

    def on_compare_click():
        print('on_compare_click called')

        cap = cv2.VideoCapture(0)

        countdown_label = Label(f2, font=('Arial', 50), bg='lightblue', fg='black')
        countdown_label.place(relx=0.48, rely=0.4)

        def countdown(count):
            if count > 0:
                countdown_label.config(text=str(count))
                f2.after(1000, countdown, count-1)
            else:
                countdown_label.destroy()
                ex2_thread = threading.Thread(target=ex2play)
                ex2_thread.start()
                animationpose_thread = threading.Thread(target=lambda: accuracy.animationpose(cap))
                animationpose_thread.start()

        countdown(5)

    def on_stop_click():
        global accuracy
        accuracy.stop_animation = True
        l1.destroy()
        l2.destroy()

    Button(f2, text='Compare', font=medium, bg='green', fg='white', command=on_compare_click).place(relx=0.4, rely=0.9, relheight=0.05, relwidth=0.2)
    Button(f2, text='Stop', font=medium, bg='red', fg='white', command=on_stop_click).place(relx=0.4, rely=0.95, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "q" to get Result', font=small).place(relx=0.05, rely=0.85, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "Compare" to Start', font=small).place(relx=0.05, rely=0.9, relheight=0.05, relwidth=0.2)
def ex3():
    global f2
    f2 = Frame(root, bg='lightblue')
    f2.place(relx=0.05, rely=0.15, relheight=0.65, relwidth=0.9)

    global l1
    l1 = Label(f2,text='exercise', bg='darkblue')
    l1.place(relx=0.025, rely=0.1, relheight=0.7, relwidth=0.45)

    global l2
    l2 = Label(f2,text='camera feed', bg='black', fg='white')
    l2.place(relx=0.525, rely=0.1, relheight=0.7, relwidth=0.45)

    global accuracy
    accuracy = AccuracyPose('e3.mkv', l2)

    def on_compare_click():
        print('on_compare_click called')

        cap = cv2.VideoCapture(0)

        countdown_label = Label(f2, font=('Arial', 50), bg='lightblue', fg='black')
        countdown_label.place(relx=0.48, rely=0.4)

        def countdown(count):
            if count > 0:
                countdown_label.config(text=str(count))
                f2.after(1000, countdown, count-1)
            else:
                countdown_label.destroy()
                ex3_thread = threading.Thread(target=ex3play)
                ex3_thread.start()
                animationpose_thread = threading.Thread(target=lambda: accuracy.animationpose(cap))
                animationpose_thread.start()

        countdown(5)

    def on_stop_click():
        global accuracy
        accuracy.stop_animation = True
        l1.destroy()
        l2.destroy()

    Button(f2, text='Compare', font=medium, bg='green', fg='white', command=on_compare_click).place(relx=0.4, rely=0.9, relheight=0.05, relwidth=0.2)
    Button(f2, text='Stop', font=medium, bg='red', fg='white', command=on_stop_click).place(relx=0.4, rely=0.95, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "q" to get Result', font=small).place(relx=0.05, rely=0.85, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "Compare" to Start', font=small).place(relx=0.05, rely=0.9, relheight=0.05, relwidth=0.2)
def ex4():
    global f2
    f2 = Frame(root, bg='lightblue')
    f2.place(relx=0.05, rely=0.15, relheight=0.65, relwidth=0.9)

    global l1
    l1 = Label(f2,text='exercise', bg='darkblue')
    l1.place(relx=0.025, rely=0.1, relheight=0.7, relwidth=0.45)

    global l2
    l2 = Label(f2,text='camera feed', bg='black', fg='white')
    l2.place(relx=0.525, rely=0.1, relheight=0.7, relwidth=0.45)

    global accuracy
    accuracy = AccuracyPose('e4.mkv', l2)

    def on_compare_click():
        print('on_compare_click called')

        cap = cv2.VideoCapture(0)

        countdown_label = Label(f2, font=('Arial', 50), bg='lightblue', fg='black')
        countdown_label.place(relx=0.48, rely=0.4)

        def countdown(count):
            if count > 0:
                countdown_label.config(text=str(count))
                f2.after(1000, countdown, count-1)
            else:
                countdown_label.destroy()
                ex4_thread = threading.Thread(target=ex4play)
                ex4_thread.start()
                animationpose_thread = threading.Thread(target=lambda: accuracy.animationpose(cap))
                animationpose_thread.start()

        countdown(5)

    def on_stop_click():
        global accuracy
        accuracy.stop_animation = True
        l1.destroy()
        l2.destroy()

    Button(f2, text='Compare', font=medium, bg='green', fg='white', command=on_compare_click).place(relx=0.4, rely=0.9, relheight=0.05, relwidth=0.2)
    Button(f2, text='Stop', font=medium, bg='red', fg='white', command=on_stop_click).place(relx=0.4, rely=0.95, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "q" to get Result', font=small).place(relx=0.05, rely=0.85, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "Compare" to Start', font=small).place(relx=0.05, rely=0.9, relheight=0.05, relwidth=0.2)
def ex5():
    global f2
    f2 = Frame(root, bg='lightblue')
    f2.place(relx=0.05, rely=0.15, relheight=0.65, relwidth=0.9)

    global l1
    l1 = Label(f2,text='exercise', bg='darkblue')
    l1.place(relx=0.025, rely=0.1, relheight=0.7, relwidth=0.45)

    global l2
    l2 = Label(f2,text='camera feed', bg='black', fg='white')
    l2.place(relx=0.525, rely=0.1, relheight=0.7, relwidth=0.45)

    global accuracy
    accuracy = AccuracyPose('e5.mkv', l2)

    def on_compare_click():
        print('on_compare_click called')

        cap = cv2.VideoCapture(0)

        countdown_label = Label(f2, font=('Arial', 50), bg='lightblue', fg='black')
        countdown_label.place(relx=0.48, rely=0.4)

        def countdown(count):
            if count > 0:
                countdown_label.config(text=str(count))
                f2.after(1000, countdown, count-1)
            else:
                countdown_label.destroy()
                ex5_thread = threading.Thread(target=ex5play)
                ex5_thread.start()
                animationpose_thread = threading.Thread(target=lambda: accuracy.animationpose(cap))
                animationpose_thread.start()

        countdown(5)

    def on_stop_click():
        global accuracy
        accuracy.stop_animation = True
        l1.destroy()
        l2.destroy()

    Button(f2, text='Compare', font=medium, bg='green', fg='white', command=on_compare_click).place(relx=0.4, rely=0.9, relheight=0.05, relwidth=0.2)
    Button(f2, text='Stop', font=medium, bg='red', fg='white', command=on_stop_click).place(relx=0.4, rely=0.95, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "q" to get Result', font=small).place(relx=0.05, rely=0.85, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "Compare" to Start', font=small).place(relx=0.05, rely=0.9, relheight=0.05, relwidth=0.2)
def ex6():
    global f2
    f2 = Frame(root, bg='lightblue')
    f2.place(relx=0.05, rely=0.15, relheight=0.65, relwidth=0.9)

    global l1
    l1 = Label(f2,text='exercise', bg='darkblue')
    l1.place(relx=0.025, rely=0.1, relheight=0.7, relwidth=0.45)

    global l2
    l2 = Label(f2,text='camera feed', bg='black', fg='white')
    l2.place(relx=0.525, rely=0.1, relheight=0.7, relwidth=0.45)

    global accuracy
    accuracy = AccuracyPose('e6.mkv', l2)

    def on_compare_click():
        print('on_compare_click called')

        cap = cv2.VideoCapture(0)

        countdown_label = Label(f2, font=('Arial', 50), bg='lightblue', fg='black')
        countdown_label.place(relx=0.48, rely=0.4)

        def countdown(count):
            if count > 0:
                countdown_label.config(text=str(count))
                f2.after(1000, countdown, count-1)
            else:
                countdown_label.destroy()
                ex6_thread = threading.Thread(target=ex6play)
                ex6_thread.start()
                animationpose_thread = threading.Thread(target=lambda: accuracy.animationpose(cap))
                animationpose_thread.start()

        countdown(5)

    def on_stop_click():
        global accuracy
        accuracy.stop_animation = True
        l1.destroy()
        l2.destroy()

    Button(f2, text='Compare', font=medium, bg='green', fg='white', command=on_compare_click).place(relx=0.4, rely=0.9, relheight=0.05, relwidth=0.2)
    Button(f2, text='Stop', font=medium, bg='red', fg='white', command=on_stop_click).place(relx=0.4, rely=0.95, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "q" to get Result', font=small).place(relx=0.05, rely=0.85, relheight=0.05, relwidth=0.2)
    Label(f2, text='Press "Compare" to Start', font=small).place(relx=0.05, rely=0.9, relheight=0.05, relwidth=0.2)

def start():
    global f1
    f1 = Frame(root, bg='grey6')
    f1.place(relx=0.05, rely=0.15, relheight=0.65, relwidth=0.9)
    Button(f1, text='Exercise 1', bg='darkblue', fg='white', font=large, command=ex1).place(relx=0.2, rely=0.1,relwidth=0.3,relheight=0.25)
    Button(f1, text='Exercise 2', bg='darkblue', fg='white', font=large, command=ex2).place(relx=0.6, rely=0.1,relwidth=0.3,relheight=0.25)
    Button(f1, text='Exercise 3', bg='darkblue', fg='white', font=large, command=ex3).place(relx=0.2, rely=0.4,relwidth=0.3,relheight=0.25)
    Button(f1, text='Exercise 4', bg='darkblue', fg='white', font=large, command=ex4).place(relx=0.6, rely=0.4,relwidth=0.3,relheight=0.25)
    Button(f1, text='Exercise 5', bg='darkblue', fg='white', font=large, command=ex5).place(relx=0.2, rely=0.7,relwidth=0.3,relheight=0.25)
    Button(f1, text='Exercise 6', bg='darkblue', fg='white', font=large, command=ex6).place(relx=0.6, rely=0.7,relwidth=0.3, relheight=0.25)
def quit():
    root.destroy()
small = ("Comic Sans Ms", 10, "bold")
large = ("Comic Sans MS", 24, "bold")
medium = ("Comic Sans MS", 16, "bold")
root = Tk()
w = root.winfo_screenwidth()
h = root.winfo_screenheight()
root.attributes('-fullscreen', True)
root.title("Exercizer")
# print(w, h)
# w = 1200
# h = 960
Label(root, text='Welcome to Exercizer', font=large).place(relx=0.2, rely=0, relwidth=0.6, relheight=0.2)
root.geometry("{0}x{1}+0+0".format(w, h))
Button(root, text='Press when ready to Continue', bg='green', font=medium, command=start).place(relx=0.2, rely=0.85,relwidth=0.3,relheight=0.1)
Button(root, text='Exit', bg='red', font=medium, command=quit).place(relx=0.6, rely=0.85, relwidth=0.3, relheight=0.1)
root.mainloop()


