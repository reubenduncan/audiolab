import gradio as gr
from typing import List
from tts_engine import tts_engine
from audio_generators import music_generator, sfx_generator, audio_separator

def create_tts_inference_interface():
    """Create the TTS tab interface."""
    with gr.Column():
        gr.Markdown("## Text-to-Speech Generation")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection dropdown
                model_dropdown = gr.Dropdown(
                    choices=[
                        "canopylabs/orpheus-3b-0.1-ft",
                        "reubenduncan/orpheus_model_bastila",
                    ],
                    label="Select TTS Model",
                    value="canopylabs/orpheus-3b-0.1-ft"
                )
                
            with gr.Column(scale=2):
                # Text input
                text_input = gr.Textbox(
                    label="Text to Convert",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=5,
                    max_lines=10
                )
                
                # Generate button
                generate_btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Row():
            # Output audio player
            audio_output = gr.Audio(label="Generated Speech", type="filepath")
            
        # Status output
        status_output = gr.Textbox(label="Status", lines=3, interactive=False)
        
        # Connect the generate button to the TTS function
        generate_btn.click(
            fn=lambda text, model: tts_engine.generate_speech(text, model),
            inputs=[text_input, model_dropdown],
            outputs=[audio_output, status_output]
        )

def create_tts_finetuning_interface():
    """Create the TTS finetuning interface."""
    with gr.Column():
        gr.Markdown("## TTS Finetuning")
        gr.Markdown("Upload audio files to finetune a LoRA model, select a TTS model, and enter text to generate speech.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Audio files upload for LoRA finetuning
                audio_upload = gr.File(
                    label="Upload Audio Files for LoRA Training",
                    file_count="multiple",
                    file_types=["audio"],
                    height=150
                )
                
                # Model selection dropdown
                model_dropdown = gr.Dropdown(
                    choices=[
                        "canopylabs/orpheus-3b-0.1-ft",
                        "reubenduncan/orpheus_model_bastila",
                    ],
                    label="Select TTS Model",
                    value="canopylabs/orpheus-3b-0.1-ft"
                )
                
            with gr.Column(scale=2):
                # File selector to select multiple wav files
                file_selector = gr.Files(
                    label="Select WAV Files",
                    file_count="multiple",
                    file_types=["wav"],
                    height=150
                )
                
                # Generate button
                generate_btn = gr.Button("Train Model", variant="primary")
        
        # with gr.Row():
        #     # Output audio player
        #     audio_output = gr.Audio(label="Generated Speech", type="filepath")
            
        # Status output
        status_output = gr.Textbox(label="Status", lines=3, interactive=False)
        
        # Connect the generate button to the TTS function
        generate_btn.click(
            fn=lambda text, model, files: tts_engine.train_model(text, model, files),
            inputs=[text_input, model_dropdown, file_selector],
            outputs=[status_output]
        )

def create_tts_interface():
    with gr.Tabs():
        with gr.Tab("Inference"):
            create_tts_inference_interface()
        with gr.Tab("Finetuning"):
            create_tts_finetuning_interface()
    

def create_music_interface():
    """Create the Music Generation tab interface."""
    with gr.Column():
        gr.Markdown("## Music Generation")
        gr.Markdown("Generate music from text descriptions using AI models.")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Music Description",
                    placeholder="Describe the music you want to generate (e.g., 'upbeat electronic dance music with synthesizers')...",
                    lines=3
                )
                
                with gr.Row():
                    duration_slider = gr.Slider(
                        minimum=5,
                        maximum=120,
                        value=30,
                        step=5,
                        label="Duration (seconds)"
                    )
                    
                    style_dropdown = gr.Dropdown(
                        choices=[
                            "Classical",
                            "Jazz",
                            "Rock",
                            "Electronic",
                            "Ambient",
                            "Pop",
                            "Hip-Hop",
                            "Folk",
                            "Blues"
                        ],
                        label="Music Style",
                        value="Electronic"
                    )
                
                generate_music_btn = gr.Button("Generate Music", variant="primary")
        
        with gr.Row():
            music_output = gr.Audio(label="Generated Music", type="filepath")
            
        music_status = gr.Textbox(label="Status", lines=2, interactive=False)
        
        generate_music_btn.click(
            fn=music_generator.generate_music,
            inputs=[prompt_input, duration_slider, style_dropdown],
            outputs=[music_output, music_status]
        )


def create_sfx_interface():
    """Create the Sound Effects tab interface."""
    with gr.Column():
        gr.Markdown("## Sound Effects Generation")
        gr.Markdown("Generate custom sound effects from text descriptions.")
        
        with gr.Row():
            with gr.Column():
                sfx_description = gr.Textbox(
                    label="Sound Effect Description",
                    placeholder="Describe the sound effect (e.g., 'thunder rumbling in the distance', 'footsteps on gravel')...",
                    lines=3
                )
                
                with gr.Row():
                    sfx_duration = gr.Slider(
                        minimum=0.5,
                        maximum=10,
                        value=3,
                        step=0.5,
                        label="Duration (seconds)"
                    )
                    
                    intensity_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Intensity"
                    )
                
                generate_sfx_btn = gr.Button("Generate Sound Effect", variant="primary")
        
        with gr.Row():
            sfx_output = gr.Audio(label="Generated Sound Effect", type="filepath")
            
        sfx_status = gr.Textbox(label="Status", lines=2, interactive=False)
        
        generate_sfx_btn.click(
            fn=sfx_generator.generate_sfx,
            inputs=[sfx_description, sfx_duration, intensity_slider],
            outputs=[sfx_output, sfx_status]
        )


def create_separation_interface():
    """Create the Audio Separation tab interface."""
    with gr.Column():
        gr.Markdown("## Audio Separation")
        gr.Markdown("Separate audio into different components (vocals, instruments, etc.).")
        
        with gr.Row():
            with gr.Column():
                separation_upload = gr.File(
                    label="Upload Audio Files",
                    file_count="multiple",
                    file_types=["audio"],
                    height=150
                )
                
                separation_type = gr.Dropdown(
                    choices=[
                        "Vocals/Instruments",
                        "Individual Instruments",
                        "Drums/Bass/Other",
                        "Speech/Music/Noise"
                    ],
                    label="Separation Type",
                    value="Vocals/Instruments"
                )
                
                separate_btn = gr.Button("Separate Audio", variant="primary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Separated Audio Files")
                separated_output1 = gr.Audio(label="Component 1", type="filepath")
                separated_output2 = gr.Audio(label="Component 2", type="filepath")
                separated_output3 = gr.Audio(label="Component 3", type="filepath")
            
        separation_status = gr.Textbox(label="Status", lines=2, interactive=False)
        
        def handle_separation(files, sep_type):
            """Handle audio separation and return individual components."""
            result_files, status = audio_separator.separate_audio(files, sep_type)
            
            # Return up to 3 separated files, None for unused outputs
            if result_files:
                outputs = result_files + [None] * (3 - len(result_files))
                return outputs[0], outputs[1], outputs[2], status
            else:
                return None, None, None, status
        
        separate_btn.click(
            fn=handle_separation,
            inputs=[separation_upload, separation_type],
            outputs=[separated_output1, separated_output2, separated_output3, separation_status]
        )


def create_app():
    """Create the main Gradio application with all tabs."""
    with gr.Blocks(
        title="Audio Lab",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as app:
        gr.Markdown(
            """
            # 🎵 Audio Lab
            
            **Your AI-powered audio generation and processing studio**
            
            Generate speech, music, sound effects, and separate audio components using state-of-the-art AI models.
            """
        )
        
        with gr.Tabs():
            with gr.Tab("🗣️ Text-to-Speech"):
                create_tts_interface()
                
            with gr.Tab("🎵 Music Generation"):
                create_music_interface()
                
            with gr.Tab("🔊 Sound Effects"):
                create_sfx_interface()
                
            with gr.Tab("🎛️ Audio Separation"):
                create_separation_interface()
    
    return app
