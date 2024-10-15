import tkinter as tk
from tkinter import ttk
import pyaudio
import wave
import whisper
import os
import datetime
import threading
import queue
import numpy as np
from difflib import SequenceMatcher

class TranscriptionTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcription Tool")
        self.root.geometry("800x600")  # Increased window size

        self.is_recording = False
        self.frames_queue = queue.Queue()
        self.all_segments = []  # To store all segments
        self.transcriptions = []  # To store all transcriptions
        self.sample_rate = 48000  # Default sample rate
        self.channels = 2
        self.chunk = 1024
        self.segment_duration = 2  # Segment duration in seconds
        self.concatenation_segments = 5  # Number of segments to concatenate
        self.volume_threshold = 800  # Default volume threshold
        self.segments_count = 0
        self.closed_index = 0  # Initialize closed_index
        self.fixed_content = ""  # To store fixed content

        self.p = pyaudio.PyAudio()

        # Load the Whisper AI model once
        self.model = whisper.load_model("base", device="cuda")  # Use the 'large' model for higher accuracy

        self.init_ui()

        # Bind the close event to ensure proper shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        self.device_label = ttk.Label(top_frame, text="Select Audio Input Device:")
        self.device_label.pack(side=tk.LEFT, padx=5)

        self.device_list = ttk.Combobox(top_frame, width=50)
        self.device_list['values'] = [f"{i}: {self.p.get_device_info_by_index(i)['name']}" for i in range(self.p.get_device_count()) if self.p.get_device_info_by_index(i)['maxInputChannels'] > 0]
        self.device_list.pack(side=tk.LEFT, padx=5)

        self.sample_rate_label = ttk.Label(top_frame, text="Select Sample Rate:")
        self.sample_rate_label.pack(side=tk.LEFT, padx=5)

        self.sample_rate_list = ttk.Combobox(top_frame, values=[8000, 16000, 22050, 44100, 48000, 96000, 192000])
        self.sample_rate_list.current(4)  # Default to 48000
        self.sample_rate_list.pack(side=tk.LEFT, padx=5)

        self.language_label = ttk.Label(top_frame, text="Select Language:")
        self.language_label.pack(side=tk.LEFT, padx=5)

        # Language selection
        self.language_list = ttk.Combobox(top_frame, values=["en", "es", "fr", "de", "zh", "ja", "ko", "ru", "it", "pt", "ar", "hi"])
        self.language_list.current(0)  # Default to English
        self.language_list.pack(side=tk.LEFT, padx=5)

        self.volume_threshold_label = ttk.Label(top_frame, text="Volume Threshold:")
        self.volume_threshold_label.pack(side=tk.LEFT, padx=5)

        self.volume_threshold_entry = tk.Entry(top_frame)
        self.volume_threshold_entry.insert(0, str(self.volume_threshold))
        self.volume_threshold_entry.pack(side=tk.LEFT, padx=5)

        self.start_button = ttk.Button(self.root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.TOP, pady=5)

        self.pause_button = ttk.Button(self.root, text="Pause Recording", command=self.pause_recording, state='disabled')
        self.pause_button.pack(side=tk.TOP, pady=5)

        self.resume_button = ttk.Button(self.root, text="Resume Recording", command=self.resume_recording, state='disabled')
        self.resume_button.pack(side=tk.TOP, pady=5)

        self.stop_button = ttk.Button(self.root, text="Stop Recording", command=self.stop_recording, state='disabled')
        self.stop_button.pack(side=tk.TOP, pady=5)

        self.transcription_text = tk.Text(self.root, wrap=tk.WORD, font=("Arial", 12))
        self.transcription_text.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

    def start_recording(self):
        self.is_recording = True
        self.segments_count = 0
        self.closed_index = 0  # Reset closed_index for new recording session
        self.fixed_content = ""  # Reset fixed content
        self.all_segments = []  # Reset segments for new recording session
        self.transcriptions = []  # Reset transcriptions for new recording session
        self.start_button['state'] = 'disabled'
        self.pause_button['state'] = 'normal'
        self.stop_button['state'] = 'normal'

        self.sample_rate = int(self.sample_rate_list.get())
        self.volume_threshold = int(self.volume_threshold_entry.get())
        self.selected_language = self.language_list.get()
        device_index = int(self.device_list.get().split(":")[0])

        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  frames_per_buffer=self.chunk,
                                  input=True,
                                  input_device_index=device_index)

        self.recording_thread = threading.Thread(target=self.record)
        self.recording_thread.start()
        self.transcription_thread = threading.Thread(target=self.process_segments)
        self.transcription_thread.start()

    def record(self):
        try:
            while self.is_recording:
                segment_frames = []
                for _ in range(0, int(self.sample_rate / self.chunk * self.segment_duration)):
                    if not self.is_recording:
                        break
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    segment_frames.append(data)
                if segment_frames:
                    self.frames_queue.put((self.segments_count, segment_frames))
                    self.segments_count += 1
                    print(f"Recorded segment {self.segments_count - 1}")
        except Exception as e:
            print(f"Error in record: {e}")

    def analyze_volume(self, segment_frames):
        audio_data = np.frombuffer(b''.join(segment_frames), dtype=np.int16)
        volume = np.max(np.abs(audio_data))
        return volume

    def process_segments(self):
        while self.is_recording or not self.frames_queue.empty():
            if not self.frames_queue.empty():
                segment_id, segment_frames = self.frames_queue.get()

                # Analyze the volume of the segment
                volume = self.analyze_volume(segment_frames)
                segment_type, debug_message = self.determine_segment_type(volume)
                print(f"Segment {segment_id} - {debug_message}")

                self.all_segments.append((segment_id, segment_frames, segment_type))

                # Save the current segment as a separate file
                self.save_segment(segment_id, segment_frames)

                # Check for END and START sequence
                if debug_message == "E/S":
                    # Perform the additional steps
                    self.fixed_content = self.transcription_text.get(1.0, tk.END).strip()
                    self.transcription_text.insert(tk.END, ' \n')
                    self.closed_index = len(self.fixed_content) +1
                    self.transcriptions.clear()  # Reset transcriptions list
                    # No need to trigger the correction process anymore

                # Process the most recent segments (up to self.concatenation_segments)
                segments_to_process = self.get_recent_segments()
                concatenated_frames = [frame for _, frames, _ in segments_to_process for frame in frames]
                self.transcribe(concatenated_frames)

    def determine_segment_type(self, volume):
        if not self.all_segments:
            return "START", "FIRST"
        previous_type = self.all_segments[-1][2]
        if volume > self.volume_threshold:
            return "NORMAL", "NORMAL"
        else:
            if previous_type == "NORMAL":
                return "SILENT", "N/O"
            elif previous_type == "START":
                self.all_segments[-1] = (self.all_segments[-1][0], self.all_segments[-1][1], "SILENT")
                return "START", "O/S"
            elif previous_type == "SILENT":
                self.all_segments[-1] = (self.all_segments[-1][0], self.all_segments[-1][1], "END")
                return "START", "E/S"
            elif previous_type == "END":
                return "START", "ERR/S"
            else:
                return "START", "ERR/S"

    def get_recent_segments(self):
        if len(self.all_segments) > self.concatenation_segments:
            recent_segments = self.all_segments[-self.concatenation_segments:]
        else:
            recent_segments = self.all_segments
        # Exclude segments before the last START segment in the recent_segments list
        start_index = next((i for i in reversed(range(len(recent_segments))) if recent_segments[i][2] == "START"), None)
        if start_index is not None:
            return recent_segments[start_index:]
        return recent_segments

    def save_segment(self, segment_id, segment_frames):
        try:
            segment_filename = f"segment_{segment_id}.wav"
            file_path = os.path.join("assets", "recordings", segment_filename)

            wf = wave.open(file_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(segment_frames))
            wf.close()

        except Exception as e:
            print(f"Error in save_segment: {e}")

    def transcribe(self, concatenated_frames):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"temp_recording_{timestamp}.wav"
            temp_file_path = os.path.join("assets", "recordings", temp_filename)

            wf = wave.open(temp_file_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(concatenated_frames))
            wf.close()

            # Ensure audio data is correctly processed before transcription
            audio = whisper.load_audio(temp_file_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            options = whisper.DecodingOptions(language=self.selected_language, fp16=False)
            result = whisper.decode(self.model, mel, options)

            # Store the transcription
            if len(result.text.strip()) > 1 and result.text.strip() != "Thank you.":
                self.transcriptions.append(result.text.strip())
                self.merge_transcriptions()
                return result.text.strip()
            else:
                print(f"Ignored short or empty transcription")

            # Remove temporary file
            os.remove(temp_file_path)

        except Exception as e:
            print(f"Error in transcribe: {e}")

    def merge_transcriptions(self):
        try:
            if not self.transcriptions:
                return

            merged_text = self.transcriptions[0]
            for i in range(1, len(self.transcriptions)):
                merged_text = self.merge_two_texts(merged_text, self.transcriptions[i])

            # Update the transcription text box
            self.transcription_text.delete(f"1.{self.closed_index}", tk.END)
            self.transcription_text.insert(tk.END, merged_text)

            # Save the merged transcription
            with open(os.path.join("assets", "transcriptions", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), "w") as file:
                file.write(self.transcription_text.get(1.0, tk.END).strip())

        except Exception as e:
            print(f"Error in merge_transcriptions: {e}")

    def merge_two_texts(self, text1, text2):
        # Split texts into tokens
        tokens1 = text1.split()
        tokens2 = text2.split()

        # Create a SequenceMatcher object
        seq_matcher = SequenceMatcher(None, tokens1, tokens2)
        match = seq_matcher.find_longest_match(0, len(tokens1), 0, len(tokens2))

        if match.size > 0:
            start = tokens1[:match.a]
            overlap = tokens1[match.a: match.a + match.size]
            end = tokens2[match.b + match.size:]

            # Merge tokens with confidence
            merged_tokens = start + overlap + end
        else:
            merged_tokens = tokens1 + tokens2

        merged_text = ' '.join(merged_tokens)
        return merged_text

    def pause_recording(self):
        self.is_recording = False
        self.pause_button['state'] = 'disabled'
        self.resume_button['state'] = 'normal'

    def resume_recording(self):
        self.is_recording = True
        self.pause_button['state'] = 'normal'
        self.resume_button['state'] = 'disabled'
        self.recording_thread = threading.Thread(target=self.record)
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False

        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join()

        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.process_remaining_segments()

        self.start_button['state'] = 'normal'
        self.pause_button['state'] = 'disabled'
        self.resume_button['state'] = 'disabled'
        self.stop_button['state'] = 'disabled'

    def process_remaining_segments(self):
        remaining_thread = threading.Thread(target=self.process_remaining)
        remaining_thread.start()

    def process_remaining(self):
        all_segment_frames = []
        while not self.frames_queue.empty():
            _, segment_frames = self.frames_queue.get()
            all_segment_frames.extend(segment_frames)

        # Ensure the last segment is concatenated with previous segments
        if len(self.all_segments) > self.concatenation_segments:
            segments_to_process = self.all_segments[-self.concatenation_segments:]
        else:
            segments_to_process = self.all_segments

        concatenated_frames = [frame for _, frames, _ in segments_to_process for frame in frames]
        if all_segment_frames:
            concatenated_frames.extend(all_segment_frames)
        if concatenated_frames:
            self.transcribe(concatenated_frames)

    def on_closing(self):
        if self.is_recording:
            self.stop_recording()
        self.root.destroy()

    def __del__(self):
        self.p.terminate()

if __name__ == "__main__":
    if not os.path.exists("assets/recordings"):
        os.makedirs("assets/recordings")

    if not os.path.exists("assets/transcriptions"):
        os.makedirs("assets/transcriptions")
    
    root = tk.Tk()
    app = TranscriptionTool(root)
    root.mainloop()
