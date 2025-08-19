import torch
import torchaudio.transforms as T
from unsloth import FastLanguageModel
from transformers import pipeline, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from snac import SNAC
import soundfile as sf
import tempfile
import os
from typing import Optional, Tuple
from datasets import load_dataset

class TTSEngine:
    """TTS model handler for text-to-speech generation."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_initialized = False
        self.base_model = "canopylabs/orpheus-3b-0.1-ft"
    
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
            print("Loading TTS model...")
            
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
            
            print("✅ Base model loaded successfully")
            
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
    
    def train_model(self, prompt: str, model_name: str = "Orpheus TTS", files: list = []) -> Tuple[Optional[str], str]:
        """
        Train a LoRA model from text and audio files.
        
        Args:
            prompt: Text to convert to speech
            model_name: Name of the model (for status reporting)
            files: List of audio files to use for training
            
        Returns:
            Tuple of (audio_path, status_message)
        """        
        dataset = load_dataset("MrDragonFox/Elise", split="train")
        
        trainer = Trainer(
            model = self.model,
            train_dataset = dataset,
            args = TrainingArguments(
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = 60,
                learning_rate = 2e-4,
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none", # Use this for WandB etc
            ),
        )
        
        trainer_stats = trainer.train()
        
        return trainer_stats
        
    def generate_speech(self, prompt: str, model_name: str = "Orpheus TTS") -> Tuple[Optional[str], str]:
        """
        Generate speech from text using the Orpheus TTS model.
        
        Args:
            prompt: Text to convert to speech
            model_name: Name of the model (for status reporting)
            
        Returns:
            Tuple of (audio_path, status_message)
        """
        if not self.is_initialized:
            if not self.initialize():
                return None, "❌ Error: Failed to initialize TTS models"
        
        if not prompt.strip():
            return None, "❌ Error: Please provide text to convert to speech"
        
        try:
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
            end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human

            input_ids = torch.cat([torch.tensor([[128263]]), start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
            attention_mask = torch.cat([torch.tensor([[0, 1]]), attention_mask, torch.tensor([[1, 1]])], dim=1)
            
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

            new_length = (len(filtered_tokens) // 7) * 7
            filtered_tokens = filtered_tokens[:new_length]

            # Adjust token values for SNAC decoding
            # adjusted_tokens = [token - 128266 for token in filtered_tokens if token >= 128266]
            adjusted_tokens = [token - 128266 for token in filtered_tokens]
            
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
        
    def inference(self, prompts: list[str], model: str = "Orpheus TTS"):
        """
        Generate speech from text using the Orpheus TTS model.
        
        Args:
            prompts: List of prompts to convert to speech
            model: Selected TTS model
            
        Returns:
            Tuple of (audio_path, status_message)
        """
        if not self.is_initialized:
            if not self.initialize():
                return None, "❌ Error: Failed to initialize TTS models"
        
        prompts = [p.strip() for p in prompts if p.strip()]
        
        if len(prompts) == 0:
            return None, "❌ Error: Please provide text to convert to speech"
        
        all_input_ids = []

        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            all_input_ids.append(input_ids)

        start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human

        all_modified_input_ids = []
        for input_ids in all_input_ids:
            modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
            all_modified_input_ids.append(modified_input_ids)

        all_padded_tensors = []
        all_attention_masks = []
        max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
        for modified_input_ids in all_modified_input_ids:
            padding = max_length - modified_input_ids.shape[1]
            padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
            attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)

        all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        if torch.cuda.is_available():
            input_ids = all_padded_tensors.to("cuda")
            attention_mask = all_attention_masks.to("cuda")

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
                use_cache = True
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
        processed_rows = []

        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []

        # Adjust token values for SNAC decoding
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        # Decode audio codes using SNAC
        my_samples = []
        for code_list in code_lists:
            samples = self.redistribute_codes(code_list)
            my_samples.append(samples)
            
        return my_samples
    
    def save_model(self):
        self.model.save_pretrained("./orpheus-lora-model")
        self.tokenizer.save_pretrained("./orpheus-lora-model")

# Global TTS engine instance
tts_engine = TTSEngine()
