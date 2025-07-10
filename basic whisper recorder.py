import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sounddevice as sd
import numpy as np
import whisper

# Load the Whisper model once at startup
model = whisper.load_model("base")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder and Transcriber")
        
        # Frame for device selection and wake word input
        top_frame = tk.Frame(root)
        top_frame.pack(pady=5)
        
        # Audio device selection dropdown
        device_frame = tk.Frame(top_frame)
        device_frame.pack(side=tk.TOP, pady=2)
        tk.Label(device_frame, text="Audio Input Device:").pack(side=tk.LEFT)
        self.devices = []
        all_devices = sd.query_devices()
        # Filter for devices with input capability
        for idx, device in enumerate(all_devices):
            if device['max_input_channels'] > 0:
                self.devices.append((idx, device['name']))
        if not self.devices:
            self.devices.append((None, "No available devices"))
        # Create dropdown options in "index: name" format
        self.device_var = tk.StringVar(root)
        default_device = f"{self.devices[0][0]}: {self.devices[0][1]}"
        self.device_var.set(default_device)
        options = [f"{d[0]}: {d[1]}" for d in self.devices]
        self.device_dropdown = tk.OptionMenu(device_frame, self.device_var, *options)
        self.device_dropdown.pack(side=tk.LEFT)
        
        # Wake word input
        wake_frame = tk.Frame(top_frame)
        wake_frame.pack(side=tk.TOP, pady=2)
        tk.Label(wake_frame, text="Wake Word:").pack(side=tk.LEFT)
        self.wake_word_entry = tk.Entry(wake_frame)
        self.wake_word_entry.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = tk.Label(root, text="Status: Idle", font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        # Audio level progress bar
        level_frame = tk.Frame(root)
        level_frame.pack(pady=5)
        tk.Label(level_frame, text="Audio Input Level:").pack(side=tk.LEFT)
        self.audio_level = ttk.Progressbar(level_frame, orient='horizontal', length=200, mode='determinate', maximum=100)
        self.audio_level.pack(side=tk.LEFT)
        
        # Start Recording button
        self.start_button = tk.Button(root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=5)
        
        # Scrolling text box for transcriptions
        self.transcription_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
        self.transcription_text.pack(pady=5)
        
        # Clear button to clear transcription history
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_text)
        self.clear_button.pack(pady=5)
        
        # Recording parameters
        self.duration = 5        # seconds to record
        self.sample_rate = 16000 # Whisper expects 16 kHz audio
        self.is_recording = False

    def clear_text(self):
        self.transcription_text.delete('1.0', tk.END)

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.start_button.config(state=tk.DISABLED)
            threading.Thread(target=self.record_and_transcribe, daemon=True).start()

    def update_status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=f"Status: {text}"))

    def update_audio_level(self, level):
        self.root.after(0, lambda: self.audio_level.config(value=level))

    def append_transcription(self, text):
        self.root.after(0, lambda: self.transcription_text.insert(tk.END, text + "\n"))

    def record_and_transcribe(self):
        self.update_status("Recording")
        recorded_frames = []
        
        # Callback to capture audio and update audio level
        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            rms = np.sqrt(np.mean(indata**2))
            level = min(int(rms * 1000), 100)
            self.update_audio_level(level)
            recorded_frames.append(indata.copy())
        
        # Retrieve selected device from dropdown (format "index: name")
        device_str = self.device_var.get()
        try:
            selected_device_index = int(device_str.split(":")[0])
        except ValueError:
            selected_device_index = None
        
        # Record audio from the selected device
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                callback=callback, device=selected_device_index)
        with stream:
            sd.sleep(int(self.duration * 1000))
        
        self.update_status("Transcribing")
        # Combine recorded frames and flatten to 1D array
        audio_data = np.concatenate(recorded_frames, axis=0)
        audio_data = np.squeeze(audio_data)
        
        # Transcribe using Whisper
        result = model.transcribe(audio_data)
        transcription = result["text"]
        self.append_transcription(transcription)
        
        # Check for wake word (case-insensitive)
        wake_word = self.wake_word_entry.get().strip().lower()
        if wake_word and wake_word in transcription.lower():
            print("wake word detected")
        
        self.update_status("Idle")
        self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        self.is_recording = False

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
