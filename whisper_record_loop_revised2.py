"""
# Audio Recorder and Transcriber
# Written by Justin Thomas and some LLM, MAR/2025
# Refactored for clarity and improved documentation on April 19, 2025

# Whisper audio transcription loop, you can trigger specific functions in response to the "trigger word".
# This is supposed to be the start of then using it for llm inference with LM Studio or Ollama
# Needs more management of the text window.

# If feeding into an LLM and asking for it to be improved, ask for additional functions to be called using multiple trigger words. Imagine calling "Save chat" or "Load #memory", or "Introduce new character"
# it would be best to load tools from a folder, not all of them defined in the main program. You could be pasting the json list of registered tools into the LLMs #context window

This application provides a GUI interface to:
1. Record audio from a selected input device
2. Transcribe the recording using OpenAI's Whisper model
3. Detect user-defined "trigger words" in the transcription

Features:
- Select audio input device from available options
- Define custom trigger words for triggering actions
- Adjust recording duration
- Enable/disable auto-restart of recording
- Visual indicators for audio level and recording progress
- Scrollable text display for transcription history

Future enhancement possibilities:
- Add more actions triggered by different trigger words
- Load tools from external modules
- Connect to LLMs via LM Studio or Ollama for processing transcribed text
- Improve text window management

Dependencies:
- tkinter: GUI framework
- sounddevice: Audio recording
- numpy: Array processing
- whisper: Audio transcription model
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import sounddevice as sd
import numpy as np
import whisper

# Load the Whisper model at startup
# Note: This will download the model on first run if not already available
print("Loading Whisper model (this may take a moment on first run)...")
model = whisper.load_model("base")
print("Whisper model loaded successfully!")

class AudioTranscriber:
    """
    Main application class that handles the GUI and audio processing functionality
    """
    def __init__(self, root):
        """Initialize the application with all UI components and settings"""
        self.root = root
        self.root.title("Audio Recorder and Transcriber")
        
        # Configuration variables
        self.sample_rate = 16000  # Whisper expects 16 kHz audio
        self.running = False      # Flag to control the recording loop
        
        self._create_ui_components()
    
    def _create_ui_components(self):
        """Create and arrange all UI components"""
        # === Top Configuration Frame ===
        config_frame = tk.Frame(self.root)
        config_frame.pack(pady=5)
        
        # Audio device selection
        self._setup_device_selection(config_frame)
        
        # Wake word input
        self._setup_wake_word_input(config_frame)
        
        # Recording duration input
        self._setup_duration_input(config_frame)
        
        # Auto-restart checkbox
        self._setup_auto_restart(config_frame)
        
        # === Instructions and Status ===
        # Instructions label
        instructions = (
            "Instructions:\n"
            "1. Select your audio input device from the dropdown above\n"
            "2. Click 'Start Recording' to test if your device is working (audio level should respond)\n"
            "3. Set your desired wake word and recording duration\n"
            "4. Enable auto-restart if you want continuous recording\n"
        )
        instruction_label = tk.Label(self.root, text=instructions, justify=tk.LEFT)
        instruction_label.pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Status: Idle", font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        # Audio level meter
        self._setup_audio_level_meter()
        
        # Recording progress bar
        self._setup_progress_bar()
        
        # === Control Buttons ===
        self._setup_control_buttons()
        
        # === Transcription Display ===
        # Scrolling text box for transcriptions
        self.transcription_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=10)
        self.transcription_text.pack(pady=5)
        
        # Clear button for transcription history
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_text)
        self.clear_button.pack(pady=5)
    
    def _setup_device_selection(self, parent_frame):
        """Set up the audio input device dropdown selector"""
        device_frame = tk.Frame(parent_frame)
        device_frame.pack(side=tk.TOP, pady=2)
        
        tk.Label(device_frame, text="Audio Input Device:").pack(side=tk.LEFT)
        
        # Get available audio input devices
        self.devices = []
        all_devices = sd.query_devices()
        for idx, device in enumerate(all_devices):
            if device['max_input_channels'] > 0:
                self.devices.append((idx, device['name']))
        
        if not self.devices:
            self.devices.append((None, "No available devices"))
        
        # Create dropdown menu
        self.device_var = tk.StringVar(self.root)
        default_device = f"{self.devices[0][0]}: {self.devices[0][1]}"
        self.device_var.set(default_device)
        options = [f"{d[0]}: {d[1]}" for d in self.devices]
        self.device_dropdown = tk.OptionMenu(device_frame, self.device_var, *options)
        self.device_dropdown.pack(side=tk.LEFT)
    
    def _setup_wake_word_input(self, parent_frame):
        """Set up the trigger word input fields"""
        # Frame for all trigger word inputs
        trigger_frame = tk.LabelFrame(parent_frame, text="Trigger Words", padx=5, pady=5)
        trigger_frame.pack(side=tk.TOP, pady=5, fill=tk.X)
        
        # Create entries for three trigger words
        self.trigger_words = []
        self.trigger_funcs = []
        
        for i in range(3):
            frame = tk.Frame(trigger_frame)
            frame.pack(side=tk.TOP, pady=2, fill=tk.X)
            
            tk.Label(frame, text=f"Trigger {i+1}:").pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=15)
            entry.pack(side=tk.LEFT, padx=(0, 5))
            
            # Default values for demonstration
            if i == 0:
                entry.insert(0, "hello assistant")
            elif i == 1:
                entry.insert(0, "save chat")
            elif i == 2:
                entry.insert(0, "new memory")
                
            tk.Label(frame, text="Function:").pack(side=tk.LEFT)
            func_entry = tk.Entry(frame, width=20)
            func_entry.pack(side=tk.LEFT, padx=(0, 5))
            
            # Default values for demonstration
            if i == 0:
                func_entry.insert(0, "greet_user")
            elif i == 1:
                func_entry.insert(0, "save_conversation")
            elif i == 2:
                func_entry.insert(0, "create_new_memory")
            
            self.trigger_words.append(entry)
            self.trigger_funcs.append(func_entry)
    
    def _setup_duration_input(self, parent_frame):
        """Set up the recording duration input field"""
        duration_frame = tk.Frame(parent_frame)
        duration_frame.pack(side=tk.TOP, pady=2)
        
        tk.Label(duration_frame, text="Recording Duration (seconds):").pack(side=tk.LEFT)
        self.duration_entry = tk.Entry(duration_frame, width=5)
        self.duration_entry.insert(0, "5")  # Default 5 seconds
        self.duration_entry.pack(side=tk.LEFT)
    
    def _setup_auto_restart(self, parent_frame):
        """Set up the auto-restart checkbox"""
        autorestart_frame = tk.Frame(parent_frame)
        autorestart_frame.pack(side=tk.TOP, pady=2)
        
        self.auto_restart_var = tk.BooleanVar(value=False)
        self.autorestart_checkbox = tk.Checkbutton(
            autorestart_frame, 
            text="Auto-Restart Recording", 
            variable=self.auto_restart_var
        )
        self.autorestart_checkbox.pack(side=tk.LEFT)
    
    def _setup_audio_level_meter(self):
        """Set up the audio input level meter"""
        level_frame = tk.Frame(self.root)
        level_frame.pack(pady=5)
        
        tk.Label(level_frame, text="Audio Input Level:").pack(side=tk.LEFT)
        self.audio_level = ttk.Progressbar(
            level_frame, 
            orient='horizontal', 
            length=200, 
            mode='determinate', 
            maximum=100
        )
        self.audio_level.pack(side=tk.LEFT)
    
    def _setup_progress_bar(self):
        """Set up the recording progress bar"""
        percent_frame = tk.Frame(self.root)
        percent_frame.pack(pady=5)
        
        tk.Label(percent_frame, text="Recording Progress:").pack(side=tk.LEFT)
        self.recording_progress = ttk.Progressbar(
            percent_frame, 
            orient='horizontal', 
            length=200, 
            mode='determinate', 
            maximum=100
        )
        self.recording_progress.pack(side=tk.LEFT)
    
    def _setup_control_buttons(self):
        """Set up the start and stop control buttons"""
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=5)
        
        self.start_button = tk.Button(
            buttons_frame, 
            text="Start Recording", 
            command=self.start_recording
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(
            buttons_frame, 
            text="Stop Recording", 
            command=self.stop_recording, 
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
    
    def clear_text(self):
        """Clear the transcription text area"""
        self.transcription_text.delete('1.0', tk.END)
    
    def start_recording(self):
        """Start the recording process in a separate thread"""
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            threading.Thread(target=self.recording_loop, daemon=True).start()
    
    def stop_recording(self):
        """Stop the recording process"""
        self.running = False
        self.stop_button.config(state=tk.DISABLED)
    
    def update_status(self, text):
        """Update the status label text"""
        self.root.after(0, lambda: self.status_label.config(text=f"Status: {text}"))
        # Also print to console for debugging
        print(f"Status: {text}")
    
    def update_audio_level(self, level):
        """Update the audio level meter"""
        self.root.after(0, lambda: self.audio_level.config(value=level))
    
    def update_recording_percentage(self, duration, start_time):
        """Update the recording progress bar based on elapsed time"""
        while True:
            elapsed = time.time() - start_time
            percent = min(100, (elapsed / duration) * 100)
            self.root.after(0, lambda: self.recording_progress.config(value=percent))
            if percent >= 100:
                break
            time.sleep(0.1)
        self.root.after(0, lambda: self.recording_progress.config(value=100))
    
    def transcribe_segment(self, audio_data):
        """
        Transcribe recorded audio and check for trigger words
        
        Args:
            audio_data: NumPy array containing audio samples
        """
        # Skip processing if audio data is empty
        if len(audio_data) == 0:
            self.update_status("No audio data to transcribe")
            return
            
        # Use Whisper to transcribe the recorded audio segment
        self.update_status("Processing transcription")
        result = model.transcribe(audio_data)
        transcription = result["text"]
        
        # Add transcription to text display
        self.root.after(0, lambda: self.transcription_text.insert(tk.END, transcription + "\n"))
        
        # Check for trigger words (case-insensitive)
        transcription_lower = transcription.lower()
        for i, entry in enumerate(self.trigger_words):
            trigger_word = entry.get().strip().lower()
            if trigger_word and trigger_word in transcription_lower:
                func_name = self.trigger_funcs[i].get().strip()
                self.update_status(f"Trigger word '{trigger_word}' detected!")
                print(f"Trigger word detected: '{trigger_word}' -> Function: '{func_name}'")
                
                # Execute the associated function if defined
                self._execute_trigger_function(trigger_word, func_name, transcription)
    
    def _execute_trigger_function(self, trigger_word, func_name, transcription):
        """
        Execute a function associated with a trigger word
        
        This is a placeholder that demonstrates how custom functions could be implemented.
        In a production system, these would be loaded from external modules or a registry.
        
        Args:
            trigger_word: The detected trigger word
            func_name: The name of the function to execute
            transcription: The full transcription text
        """
        # Example implementation of some basic functions
        if func_name == "greet_user":
            print("Executing: Greeting user")
            self.root.after(0, lambda: self.transcription_text.insert(tk.END, 
                            "Hello! I've detected you speaking to me.\n"))
            
        elif func_name == "save_conversation":
            print("Executing: Saving conversation")
            self.root.after(0, lambda: self.transcription_text.insert(tk.END, 
                            "Conversation would be saved here...\n"))
            
        elif func_name == "create_new_memory":
            print("Executing: Creating new memory")
            self.root.after(0, lambda: self.transcription_text.insert(tk.END, 
                            "Creating a new memory from this conversation...\n"))
        
        else:
            # For custom functions, you would implement a proper function registry
            # This is just a placeholder to show how it could work
            print(f"Custom function '{func_name}' would be executed here")
            self.root.after(0, lambda: self.transcription_text.insert(tk.END, 
                            f"Detected: '{trigger_word}' → Would execute: {func_name}()\n"))
    
    def recording_loop(self):
        """Main recording loop that captures audio and processes it"""
        while self.running:
            # Get the recording duration from the UI (default to 5 seconds if invalid)
            try:
                duration = float(self.duration_entry.get())
                if duration <= 0:
                    raise ValueError("Duration must be positive")
            except ValueError:
                duration = 5.0
                # Update the entry with the default value
                self.root.after(0, lambda: self.duration_entry.delete(0, tk.END))
                self.root.after(0, lambda: self.duration_entry.insert(0, str(duration)))
            
            self.update_status("Recording")
            recorded_frames = []
            start_time = time.time()
            
            # Start a thread to update the recording percentage progress bar
            progress_thread = threading.Thread(
                target=self.update_recording_percentage,
                args=(duration, start_time)
            )
            progress_thread.start()
            
            # Define callback function to capture audio data and update the audio level
            def callback(indata, frames, time_info, status):
                """Callback function for the sounddevice input stream"""
                if status:
                    print(f"Stream error: {status}")
                
                # Calculate audio level using RMS (root mean square)
                rms = np.sqrt(np.mean(indata**2))
                level = min(int(rms * 1000), 100)  # Scale and limit to 0-100
                self.update_audio_level(level)
                
                # Store the audio data
                recorded_frames.append(indata.copy())
            
            # Get selected device from dropdown (format "index: name")
            device_str = self.device_var.get()
            try:
                selected_device_index = int(device_str.split(":")[0])
            except ValueError:
                selected_device_index = None
            
            # Record audio for the specified duration
            try:
                # Print device information for debugging
                print(f"Recording from device: {device_str}")
                
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=callback,
                    device=selected_device_index
                )
                
                with stream:
                    sd.sleep(int(duration * 1000))  # Sleep for duration in milliseconds
                
                progress_thread.join()
                
                # Process the recorded audio
                self.update_status("Transcribing")
                if recorded_frames:
                    audio_data = np.concatenate(recorded_frames, axis=0)
                    audio_data = np.squeeze(audio_data)  # Convert from (n,1) to (n,)
                else:
                    audio_data = np.array([])
                
                # If auto-restart is enabled, launch transcription in a separate thread;
                # otherwise, process the transcription before starting a new recording.
                if self.auto_restart_var.get():
                    threading.Thread(
                        target=self.transcribe_segment,
                        args=(audio_data,),
                        daemon=True
                    ).start()
                else:
                    self.transcribe_segment(audio_data)
                
            except Exception as e:
                self.update_status(f"Error: {str(e)}")
                print(f"Recording error: {e}")
            
            # Set status to idle after processing
            self.update_status("Idle")
            
            # In non-auto mode, run only one cycle
            if not self.auto_restart_var.get():
                break
        
        # When the loop exits, reset the buttons
        self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
        self.running = False


def main():
    """Initialize and run the application"""
    root = tk.Tk()
    # Set a minimum window size
    root.minsize(500, 650)
    
    # Show startup message
    print("Starting Audio Transcriber Application")
    print("Please select your audio device and click 'Start Recording' to test")
    
    app = AudioTranscriber(root)
    root.mainloop()


if __name__ == "__main__":
    main()