"""Audio Lab - Main application entry point."""

from tts_engine import tts_engine
from audio_generators import music_generator, sfx_generator, audio_separator
from ui_components import create_app


def initialize_all_models():
    """Initialize all AI models used in the application."""
    print("🚀 Starting Audio Lab...")
    print("📦 Initializing AI models...")
    
    # Initialize TTS engine
    tts_success = tts_engine.initialize()
    
    # Initialize other generators
    music_generator.initialize()
    sfx_generator.initialize()
    audio_separator.initialize()
    
    if tts_success:
        print("✅ All models initialized successfully!")
    else:
        print("⚠️  Warning: Some models failed to load. Functionality may be limited.")
    
    return tts_success


if __name__ == "__main__":
    # Initialize all models
    initialize_all_models()
    
    # Create and launch the app
    app = create_app()
    print("🌐 Launching Gradio app at http://localhost:7860")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
