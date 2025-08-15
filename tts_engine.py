"""
Orpheus TTS Engine - Handles text-to-speech generation using the Orpheus model.
"""

import torch
import torchaudio.transforms as T
from unsloth import FastLanguageModel
from snac import SNAC
import soundfile as sf
import tempfile
import os
from typing import Optional, Tuple


class OrpheusTTSEngine:
    """Orpheus TTS model handler for text-to-speech generation."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_initialized = False
        self.base_model = "unsloth/Llama-3.2-3B-Instruct"
    
    def initialize(self, lora_model_path: str = "./orpheus-lora-model") -> bool:
        """
        Initialize the Orpheus TTS and SNAC models.
        
        Args:
            lora_model_path: Path to the LoRA finetuned model
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.is_initialized:
            return True
            
        try:
            print("Loading Orpheus TTS model...")
            
            # Try to load LoRA model first
            try:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=lora_model_path,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=False,
                )
                
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=64,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=64,
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                    use_rslora=False,
                    loftq_config=None,
                )
                print("✅ LoRA model loaded successfully")
                
            except Exception as e:
                print(f"⚠️  LoRA model not found: {e}")
                print("Loading base model as fallback...")
                
                # Fallback to base model
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.base_model,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                )
                print("✅ Base model loaded successfully")
            
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                print(f"Model loaded on {self.device}")
            
            # Load SNAC model
            print("Loading SNAC model...")
            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
            if torch.cuda.is_available():
                self.snac_model = self.snac_model.to("cuda")
            print("✅ SNAC model loaded successfully")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            return False
    
    def redistribute_codes(self, code_list: list) -> torch.Tensor:
        """
        Convert token list to SNAC codes for audio generation.
        
        Args:
            code_list: List of audio tokens
            
        Returns:
            torch.Tensor: Generated audio tensor
        """
        layer_1 = []
        layer_2 = []
        layer_3 = []
        
        for i in range((len(code_list)+1)//7):
            if 7*i < len(code_list):
                layer_1.append(code_list[7*i])
            if 7*i+1 < len(code_list):
                layer_2.append(code_list[7*i+1]-4096)
            if 7*i+2 < len(code_list):
                layer_3.append(code_list[7*i+2]-(2*4096))
            if 7*i+3 < len(code_list):
                layer_3.append(code_list[7*i+3]-(3*4096))
            if 7*i+4 < len(code_list):
                layer_2.append(code_list[7*i+4]-(4*4096))
            if 7*i+5 < len(code_list):
                layer_3.append(code_list[7*i+5]-(5*4096))
            if 7*i+6 < len(code_list):
                layer_3.append(code_list[7*i+6]-(6*4096))
        
        codes = [torch.tensor(layer_1).unsqueeze(0),
                 torch.tensor(layer_2).unsqueeze(0),
                 torch.tensor(layer_3).unsqueeze(0)]
        
        if torch.cuda.is_available():
            codes = [code.to("cuda") for code in codes]
        
        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
        
        return audio_hat
    
    def generate_speech(self, text: str, model_name: str = "Orpheus TTS") -> Tuple[Optional[str], str]:
        """
        Generate speech from text using the Orpheus TTS model.
        
        Args:
            text: Text to convert to speech
            model_name: Name of the model (for status reporting)
            
        Returns:
            Tuple of (audio_path, status_message)
        """
        if not self.is_initialized:
            if not self.initialize():
                return None, "❌ Error: Failed to initialize TTS models"
        
        if not text.strip():
            return None, "❌ Error: Please provide text to convert to speech"
        
        try:
            # Format the prompt for TTS generation
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant that converts text to speech.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|reserved_special_token_0|>"
            
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")
            
            # Generate tokens
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1200,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    num_return_sequences=1,
                    eos_token_id=128258,
                    use_cache=True
                )
            
            # Post-process generated tokens
            token_to_find = 128257
            token_to_remove = 128258
            
            # Find the last occurrence of the special token
            token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
            if len(token_indices[1]) > 0:
                last_occurrence_idx = token_indices[1][-1].item()
                cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
            else:
                cropped_tensor = generated_ids
            
            # Remove end-of-sequence tokens
            mask = cropped_tensor != token_to_remove
            filtered_tokens = []
            
            for i in range(cropped_tensor.shape[0]):
                row_tokens = cropped_tensor[i][mask[i]].tolist()
                filtered_tokens.extend(row_tokens)
            
            # Convert tokens to audio codes and decode
            if len(filtered_tokens) == 0:
                return None, "❌ Error: No valid audio tokens generated"
            
            # Adjust token values for SNAC decoding
            adjusted_tokens = [token - 128266 for token in filtered_tokens if token >= 128266]
            
            if len(adjusted_tokens) == 0:
                return None, "❌ Error: No valid audio codes after adjustment"
            
            # Generate audio using SNAC
            audio_tensor = self.redistribute_codes(adjusted_tokens)
            
            # Convert to numpy and save as audio file
            if torch.cuda.is_available():
                audio_numpy = audio_tensor.cpu().numpy().squeeze()
            else:
                audio_numpy = audio_tensor.numpy().squeeze()
            
            # Create temporary file for the generated audio
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, "generated_speech.wav")
            
            # Save audio file (SNAC outputs at 24kHz)
            sf.write(audio_path, audio_numpy, 24000)
            
            status = f"✅ TTS generation successful!\n- Text: {text[:100]}{'...' if len(text) > 100 else ''}\n- Model: {model_name}\n- Generated {len(adjusted_tokens)} audio tokens\n- Audio duration: {len(audio_numpy)/24000:.2f} seconds"
            
            return audio_path, status
            
        except Exception as e:
            error_msg = f"❌ Error during TTS generation: {str(e)}"
            print(f"TTS Error: {e}")
            import traceback
            traceback.print_exc()
            return None, error_msg


# Global TTS engine instance
tts_engine = OrpheusTTSEngine()
