"""
# Enhanced Voice Agent Dashboard
# Expanded from the original Whisper Audio Transcriber
# Uses Instructor library for structured API requests

This application provides a GUI interface to:
1. Record audio from a selected input device
2. Transcribe the recording using OpenAI's Whisper model
3. Process transcriptions with configurable personas
4. Execute tool functions like saving and summarizing conversations

Features:
- Select audio input device from available options
- Configure and switch between multiple personas
- Edit system prompts for each persona
- Save and summarize conversations using LLM processing
- Visual indicators for audio level and recording progress
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import time
import sounddevice as sd
import numpy as np
import whisper
import os
import json
import datetime
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI

# Load the Whisper model at startup
print("Loading Whisper model (this may take a moment on first run)...")
model = whisper.load_model("base")
print("Whisper model loaded successfully!")

# Define Pydantic models for structured output
class ConversationSummary(BaseModel):
    """Model for structured conversation summarization"""
    main_topics: list[str] = Field(..., description="List of main topics discussed in the conversation")
    key_points: list[str] = Field(..., description="List of key points mentioned in the conversation")
    summary: str = Field(..., description="A concise summary of the entire conversation")
    action_items: list[str] = Field(..., description="Action items or tasks mentioned in the conversation")
    
class Persona(BaseModel):
    """Model for persona configuration"""
    name: str
    system_prompt: str
    description: str
    tools: list[str] = []

class AudioTranscriberPlus:
    """
    Enhanced audio transcriber with persona management and tool integration
    """
    def __init__(self, root):
        """Initialize the application with all UI components and settings"""
        self.root = root
        self.root.title("Voice Agent Dashboard")
        
        # Configuration variables
        self.sample_rate = 16000  # Whisper expects 16 kHz audio
        self.running = False      # Flag to control the recording loop
        self.conversation_history = []  # Store transcription history
        
        # Set up personas
        self.setup_personas()
        
        # Initialize Instructor client
        self.setup_instructor_client()
        
        # Create the UI
        self._create_ui_components()
    
    def setup_personas(self):
        """Setup default personas"""
        self.personas = {
            "assistant": Persona(
                name="Assistant",
                system_prompt="You are a helpful assistant that responds to user queries.",
                description="General-purpose assistant for everyday tasks"
            ),
            "summarizer": Persona(
                name="Summarizer", 
                system_prompt="You analyze conversations and provide concise summaries highlighting the main points.",
                description="Specialized in summarizing content"
            ),
            "command": Persona(
                name="Command Parser",
                system_prompt="You convert natural language instructions into specific commands or actions.",
                description="Specialized in parsing commands from speech"
            )
        }
        
        self.current_persona = "assistant"
    
    def setup_instructor_client(self):
        """Initialize the Instructor-patched OpenAI client"""
        self.client = instructor.from_openai(
            OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio"),
            mode=instructor.Mode.TOOLS
        )
    
    def _create_ui_components(self):
        """Create and arrange all UI components"""
        # Create main frames
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.main_content = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_content.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left panel for controls
        self.left_panel = tk.Frame(self.main_content)
        self.main_content.add(self.left_panel)
        
        # Right panel for transcription and output
        self.right_panel = tk.Frame(self.main_content)
        self.main_content.add(self.right_panel)
        
        # Create components for each panel
        self._setup_config_panel(self.top_frame)
        self._setup_control_panel(self.left_panel)
        self._setup_transcription_panel(self.right_panel)
        
        # Status bar at bottom
        self.status_label = tk.Label(self.root, text="Status: Idle", font=("Arial", 12))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def _setup_config_panel(self, parent_frame):
        """Set up the top configuration panel"""
        # Audio device selection
        device_frame = tk.Frame(parent_frame)
        device_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(device_frame, text="Audio Input:").pack(side=tk.LEFT)
        
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
        
        # Recording duration input
        duration_frame = tk.Frame(parent_frame)
        duration_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(duration_frame, text="Duration (s):").pack(side=tk.LEFT)
        self.duration_entry = tk.Entry(duration_frame, width=5)
        self.duration_entry.insert(0, "5")  # Default 5 seconds
        self.duration_entry.pack(side=tk.LEFT)
        
        # Auto-restart checkbox
        self.auto_restart_var = tk.BooleanVar(value=False)
        self.autorestart_checkbox = tk.Checkbutton(
            parent_frame, 
            text="Auto-Restart", 
            variable=self.auto_restart_var
        )
        self.autorestart_checkbox.pack(side=tk.LEFT, padx=10)
    
    def _setup_control_panel(self, parent_frame):
        """Set up the left control panel with persona management and tools"""
        # Create a notebook with tabs
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Persona tab
        persona_frame = tk.Frame(notebook)
        notebook.add(persona_frame, text="Personas")
        
        # Tools tab
        tools_frame = tk.Frame(notebook)
        notebook.add(tools_frame, text="Tools")
        
        # Triggers tab
        triggers_frame = tk.Frame(notebook)
        notebook.add(triggers_frame, text="Triggers")
        
        # Set up each tab
        self._setup_persona_tab(persona_frame)
        self._setup_tools_tab(tools_frame)
        self._setup_triggers_tab(triggers_frame)
        
        # Control buttons
        buttons_frame = tk.Frame(parent_frame)
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
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
        
        # Audio level meter
        level_frame = tk.Frame(parent_frame)
        level_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        tk.Label(level_frame, text="Audio Level:").pack(side=tk.LEFT)
        self.audio_level = ttk.Progressbar(
            level_frame, 
            orient='horizontal', 
            length=200, 
            mode='determinate', 
            maximum=100
        )
        self.audio_level.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Recording progress bar
        percent_frame = tk.Frame(parent_frame)
        percent_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        tk.Label(percent_frame, text="Progress:").pack(side=tk.LEFT)
        self.recording_progress = ttk.Progressbar(
            percent_frame, 
            orient='horizontal', 
            length=200, 
            mode='determinate', 
            maximum=100
        )
        self.recording_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def _setup_persona_tab(self, parent_frame):
        """Set up the persona management tab"""
        # Persona selection
        selection_frame = tk.LabelFrame(parent_frame, text="Select Persona")
        selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.persona_var = tk.StringVar(value=self.current_persona)
        
        for persona_id, persona in self.personas.items():
            rb = tk.Radiobutton(
                selection_frame,
                text=f"{persona.name} - {persona.description}",
                variable=self.persona_var,
                value=persona_id,
                command=self.on_persona_change
            )
            rb.pack(anchor=tk.W, padx=5, pady=2)
        
        # System prompt editor
        prompt_frame = tk.LabelFrame(parent_frame, text="System Prompt")
        prompt_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.prompt_editor = scrolledtext.ScrolledText(
            prompt_frame, 
            wrap=tk.WORD,
            height=10
        )
        self.prompt_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Load the current persona's prompt
        self.prompt_editor.delete('1.0', tk.END)
        self.prompt_editor.insert('1.0', self.personas[self.current_persona].system_prompt)
        
        # Save button
        save_button = tk.Button(
            prompt_frame,
            text="Save Prompt",
            command=self.save_system_prompt
        )
        save_button.pack(pady=5)
    
    def _setup_tools_tab(self, parent_frame):
        """Set up the tools tab with buttons for available tools"""
        # Create a frame for each tool
        save_frame = tk.LabelFrame(parent_frame, text="Save Conversation")
        save_frame.pack(fill=tk.X, padx=5, pady=5)
        
        save_button = tk.Button(
            save_frame,
            text="Save Conversation to File",
            command=self.save_conversation
        )
        save_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Create format selection
        self.save_format_var = tk.StringVar(value="txt")
        format_frame = tk.Frame(save_frame)
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        tk.Radiobutton(format_frame, text="Text", variable=self.save_format_var, value="txt").pack(side=tk.LEFT)
        tk.Radiobutton(format_frame, text="JSON", variable=self.save_format_var, value="json").pack(side=tk.LEFT)
        
        # Summarize tool
        summarize_frame = tk.LabelFrame(parent_frame, text="Summarize Conversation")
        summarize_frame.pack(fill=tk.X, padx=5, pady=5)
        
        summarize_button = tk.Button(
            summarize_frame,
            text="Generate Summary",
            command=self.summarize_conversation
        )
        summarize_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Add clear conversation button
        clear_frame = tk.LabelFrame(parent_frame, text="Clear")
        clear_frame.pack(fill=tk.X, padx=5, pady=5)
        
        clear_button = tk.Button(
            clear_frame,
            text="Clear Conversation History",
            command=self.clear_conversation
        )
        clear_button.pack(fill=tk.X, padx=5, pady=5)
    
    def _setup_triggers_tab(self, parent_frame):
        """Set up the triggers tab with trigger word configuration"""
        # Create entries for three trigger words
        self.trigger_words = []
        self.trigger_funcs = []
        
        for i in range(3):
            frame = tk.LabelFrame(parent_frame, text=f"Trigger {i+1}")
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            word_frame = tk.Frame(frame)
            word_frame.pack(fill=tk.X, padx=5, pady=2)
            
            tk.Label(word_frame, text="Phrase:").pack(side=tk.LEFT)
            word_entry = tk.Entry(word_frame, width=20)
            word_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            
            # Default values
            if i == 0:
                word_entry.insert(0, "save conversation")
            elif i == 1:
                word_entry.insert(0, "summarize")
            elif i == 2:
                word_entry.insert(0, "switch persona")
                
            func_frame = tk.Frame(frame)
            func_frame.pack(fill=tk.X, padx=5, pady=2)
            
            tk.Label(func_frame, text="Function:").pack(side=tk.LEFT)
            func_entry = tk.Entry(func_frame, width=20)
            func_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            
            # Default values
            if i == 0:
                func_entry.insert(0, "save_conversation")
            elif i == 1:
                func_entry.insert(0, "summarize_conversation")
            elif i == 2:
                func_entry.insert(0, "switch_persona_dialog")
            
            self.trigger_words.append(word_entry)
            self.trigger_funcs.append(func_entry)
    
    def _setup_transcription_panel(self, parent_frame):
        """Set up the right panel for transcription and output display"""
        # Notebook with tabs for different views
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Transcription tab
        transcript_frame = tk.Frame(notebook)
        notebook.add(transcript_frame, text="Transcript")
        
        # Response tab
        response_frame = tk.Frame(notebook)
        notebook.add(response_frame, text="Response")
        
        # Summary tab
        summary_frame = tk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")
        
        # Set up each tab
        self.transcription_text = scrolledtext.ScrolledText(
            transcript_frame, 
            wrap=tk.WORD,
            width=50, 
            height=20
        )
        self.transcription_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.response_text = scrolledtext.ScrolledText(
            response_frame, 
            wrap=tk.WORD,
            width=50, 
            height=20
        )
        self.response_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.summary_text = scrolledtext.ScrolledText(
            summary_frame, 
            wrap=tk.WORD,
            width=50, 
            height=20
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def on_persona_change(self):
        """Handle persona change event"""
        # Save the current prompt before switching
        self.save_system_prompt()
        
        # Update the current persona
        self.current_persona = self.persona_var.get()
        
        # Load the new persona's prompt
        self.prompt_editor.delete('1.0', tk.END)
        self.prompt_editor.insert('1.0', self.personas[self.current_persona].system_prompt)
        
        self.update_status(f"Switched to {self.personas[self.current_persona].name} persona")
    
    def save_system_prompt(self):
        """Save the current system prompt"""
        # Get the current text from the editor
        current_text = self.prompt_editor.get('1.0', 'end-1c')
        
        # Update the persona's system prompt
        self.personas[self.current_persona].system_prompt = current_text
        
        self.update_status(f"System prompt for {self.personas[self.current_persona].name} saved")
    
    def clear_text(self, text_widget):
        """Clear the specified text widget"""
        text_widget.delete('1.0', tk.END)
    
    def clear_conversation(self):
        """Clear the conversation history and all text widgets"""
        self.conversation_history = []
        self.clear_text(self.transcription_text)
        self.clear_text(self.response_text)
        self.clear_text(self.summary_text)
        self.update_status("Conversation history cleared")
    
    def save_conversation(self):
        """Save the conversation to a file"""
        if not self.conversation_history:
            messagebox.showinfo("Info", "No conversation to save.")
            return
        
        # Get current timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"conversation_{timestamp}.{self.save_format_var.get()}"
        
        # Open file dialog
        filename = filedialog.asksaveasfilename(
            defaultextension=f".{self.save_format_var.get()}",
            filetypes=[
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ],
            initialfile=default_filename
        )
        
        if not filename:
            return  # User cancelled
        
        try:
            if self.save_format_var.get() == "txt":
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("# Conversation Transcript\n\n")
                    for i, entry in enumerate(self.conversation_history):
                        f.write(f"[{i+1}] {entry}\n\n")
            else:  # JSON format
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump({
                        "timestamp": timestamp,
                        "entries": self.conversation_history
                    }, f, indent=2)
                    
            self.update_status(f"Conversation saved to {filename}")
            
            # Also show in response
            self.response_text.insert(tk.END, f"\n--- Conversation saved to {filename} ---\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {str(e)}")
    
    def summarize_conversation(self):
        """Generate a summary of the conversation using LLM"""
        if not self.conversation_history:
            messagebox.showinfo("Info", "No conversation to summarize.")
            return
        
        self.update_status("Generating summary...")
        
        # Switch to summarizer persona
        previous_persona = self.current_persona
        self.current_persona = "summarizer"
        
        # Prepare conversation text
        conversation_text = "\n".join([f"Entry {i+1}: {entry}" for i, entry in enumerate(self.conversation_history)])
        
        # Start the summarization in a separate thread
        threading.Thread(
            target=self._run_summarization,
            args=(conversation_text, previous_persona),
            daemon=True
        ).start()
    
    def _run_summarization(self, conversation_text, previous_persona):
        """Run the summarization process in a background thread"""
        try:
            # Use Instructor to get structured output
            system_prompt = self.personas["summarizer"].system_prompt
            
            response = self.client.chat.completions.create(
                model="second-state/Mistral-Nemo-Instruct-2407-GGUF",
                response_model=ConversationSummary,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please summarize the following conversation:\n\n{conversation_text}"}
                ],
                temperature=0.7,
            )
            
            # Display the results
            self.root.after(0, lambda: self.display_summary(response))
            
            # Reset to previous persona
            self.root.after(0, lambda: self.set_persona(previous_persona))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Could not generate summary: {str(e)}"))
            self.root.after(0, lambda: self.set_persona(previous_persona))
            self.root.after(0, lambda: self.update_status("Summarization failed"))
    
    def set_persona(self, persona_id):
        """Set the active persona"""
        self.current_persona = persona_id
        self.persona_var.set(persona_id)
        self.prompt_editor.delete('1.0', tk.END)
        self.prompt_editor.insert('1.0', self.personas[self.current_persona].system_prompt)
    
    def display_summary(self, summary):
        """Display the generated summary"""
        self.clear_text(self.summary_text)
        
        # Format and display the summary
        self.summary_text.insert(tk.END, "# Conversation Summary\n\n")
        
        self.summary_text.insert(tk.END, "## Main Topics\n")
        for i, topic in enumerate(summary.main_topics):
            self.summary_text.insert(tk.END, f"{i+1}. {topic}\n")
        
        self.summary_text.insert(tk.END, "\n## Key Points\n")
        for i, point in enumerate(summary.key_points):
            self.summary_text.insert(tk.END, f"{i+1}. {point}\n")
        
        self.summary_text.insert(tk.END, "\n## Summary\n")
        self.summary_text.insert(tk.END, f"{summary.summary}\n")
        
        self.summary_text.insert(tk.END, "\n## Action Items\n")
        for i, item in enumerate(summary.action_items):
            self.summary_text.insert(tk.END, f"{i+1}. {item}\n")
        
        # Also show in response
        self.response_text.insert(tk.END, "\n--- Summary generated! See Summary tab. ---\n")
        
        self.update_status("Summary generated successfully")
    
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
    
    def add_to_transcript(self, text):
        """Add text to the transcript display and history"""
        # Add to UI
        self.root.after(0, lambda: self.transcription_text.insert(tk.END, text + "\n"))
        self.root.after(0, lambda: self.transcription_text.see(tk.END))
        
        # Add to history
        self.conversation_history.append(text)
    
    def switch_persona_dialog(self):
        """Show dialog to switch personas"""
        # Create a simple dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Switch Persona")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Select Persona:").pack(pady=10)
        
        # Create radio buttons for each persona
        persona_var = tk.StringVar(value=self.current_persona)
        
        for persona_id, persona in self.personas.items():
            rb = tk.Radiobutton(
                dialog,
                text=f"{persona.name}",
                variable=persona_var,
                value=persona_id
            )
            rb.pack(anchor=tk.W, padx=20, pady=5)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        tk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=tk.RIGHT, padx=10)
        
        tk.Button(
            button_frame,
            text="Switch",
            command=lambda: self.switch_persona_from_dialog(persona_var.get(), dialog)
        ).pack(side=tk.RIGHT, padx=10)
    
    def switch_persona_from_dialog(self, persona_id, dialog):
        """Switch to the selected persona and close the dialog"""
        self.set_persona(persona_id)
        self.update_status(f"Switched to {self.personas[persona_id].name} persona")
        dialog.destroy()
    
    def process_trigger_words(self, transcription):
        """Check for trigger words in the transcription"""
        transcription_lower = transcription.lower()
        triggered = False
        
        for i, entry in enumerate(self.trigger_words):
            trigger_word = entry.get().strip().lower()
            if trigger_word and trigger_word in transcription_lower:
                func_name = self.trigger_funcs[i].get().strip()
                self.update_status(f"Trigger word '{trigger_word}' detected!")
                
                # Execute the associated function
                if hasattr(self, func_name):
                    method = getattr(self, func_name)
                    method()
                    triggered = True
                else:
                    print(f"Function '{func_name}' not found")
        
        return triggered
    
    def process_with_llm(self, text):
        """Process text with the current persona's LLM configuration"""
        self.update_status("Processing with LLM...")
        
        try:
            # Get the current persona's system prompt
            system_prompt = self.personas[self.current_persona].system_prompt
            
            # Make the API request
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            response = client.chat.completions.create(
                model="second-state/Mistral-Nemo-Instruct-2407-GGUF",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
            )
            
            # Display the response
            response_text = response.choices[0].message.content
            self.root.after(0, lambda: self.response_text.insert(tk.END, f"\n--- {self.personas[self.current_persona].name} Response ---\n{response_text}\n"))
            self.root.after(0, lambda: self.response_text.see(tk.END))
            
            self.update_status("LLM processing complete")
            
        except Exception as e:
            error_msg = f"LLM processing error: {str(e)}"
            self.update_status(error_msg)
            self.root.after(0, lambda: self.response_text.insert(tk.END, f"\n--- Error ---\n{error_msg}\n"))
    
    def transcribe_segment(self, audio_data):
        """
        Transcribe recorded audio and process it
        
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
        
        # Add transcription to display and history
        self.add_to_transcript(transcription)
        
        # Check for trigger words
        triggered = self.process_trigger_words(transcription)
        
        # If no triggers activated, process with the current persona's LLM
        if not triggered and transcription.strip():
            threading.Thread(
                target=self.process_with_llm,
                args=(transcription,),
                daemon=True
            ).start()
    
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
    root.minsize(800, 600)
    
    # Show startup message
    print("Starting Enhanced Voice Agent Dashboard")
    print("Please select your audio device and click 'Start Recording' to test")
    
    app = AudioTranscriberPlus(root)
    root.mainloop()


if __name__ == "__main__":
    main()