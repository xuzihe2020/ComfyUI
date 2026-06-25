import os
import json
import torch
import numpy as np
from PIL import Image
import folder_paths

# ------------------------------------------------------------------
# 1. FLUX HAIRSTYLE SELECTOR
# ------------------------------------------------------------------
class FluxHairstyleSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        female_json = os.path.join(web_dir, "hairstyles_female.json")
        male_json = os.path.join(web_dir, "hairstyles_male.json")
        
        cls.hairstyles_data = {}
        all_hairstyle_names = ["[JSON Load Error]"]

        try:
            with open(female_json, 'r', encoding='utf-8') as f:
                female_data = json.load(f)
                cls.hairstyles_data['Female'] = female_data
            with open(male_json, 'r', encoding='utf-8') as f:
                male_data = json.load(f)
                cls.hairstyles_data['Male'] = male_data
            
            female_names = [item["name"] for item in female_data]
            male_names = [item["name"] for item in male_data]
            all_hairstyle_names = female_names + male_names
        except Exception as e:
            print(f"FluxHairstyle Error: {e}")

        hair_colors = ["Default (Model's)", "Blonde", "Brunette", "Black", "Red", "Ginger", "Auburn", "White", "Silver", "Grey", "Pink", "Blue", "Green", "Purple", "Orange", "Teal", "Rainbow"]

        return {
            "required": {
                "gender": (["Female", "Male"], {"default": "Female"}),
                "hairstyle": (all_hairstyle_names, ), 
                "preserve_identity": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "preserve_hair_color": ("BOOLEAN", {"default": True, "label_on": "Original", "label_off": "Change"}),
                "hair_color": (hair_colors, {"default": "Default (Model's)"}),
                "extra_instruction": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("prompt", "preview", "hairstyle_name")
    FUNCTION = "get_prompt"
    CATEGORY = "Style Loaders"

    def get_prompt(self, gender, hairstyle, preserve_identity, preserve_hair_color, hair_color, extra_instruction):
        clean_name = hairstyle
        selected_data = None
        if gender in self.hairstyles_data:
            found = next((item for item in self.hairstyles_data[gender] if item["name"] == clean_name), None)
            if found: selected_data = found
        
        prompt_text = ""
        preview_image = None

        if selected_data:
            prefix = "Replace the current hairstyle with " if gender == "Female" else "Replace the current hairstyle with "
            desc = selected_data.get("style_description", selected_data.get("description", ""))
            parts = [desc]
            
            if not preserve_hair_color:
                color_text = hair_color.replace("(Model's)", "").strip()
                if color_text: parts.append(f"{color_text} hair")
            else:
                parts.append("keeping original hair color")

            if extra_instruction and extra_instruction.strip(): parts.append(extra_instruction.strip())
            prompt_text = prefix + ", ".join(parts)

            if preserve_identity:
                identity_parts = ["facial features", "expression"]
                if preserve_hair_color: identity_parts.append("original hair color")
                prompt_text += f" while maintaining the same {', '.join(identity_parts)}, clothing details and postures of the characters in the picture unchanged"

            thumb_name = selected_data.get("thumbnail", "")
            if thumb_name:
                thumbs_dir = os.path.join(os.path.dirname(__file__), "web", "thumbnails", gender.lower())
                img_path = os.path.join(thumbs_dir, thumb_name)
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        preview_image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                    except: pass
        
        if preview_image is None: preview_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return (prompt_text, preview_image, clean_name)

# ------------------------------------------------------------------
# 2. FLUX BEARD SELECTOR
# ------------------------------------------------------------------
class FluxBeardSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        beard_json = os.path.join(web_dir, "beards.json")
        
        cls.beards_data = []
        all_beard_names = ["[JSON Load Error]"]

        try:
            if os.path.exists(beard_json):
                with open(beard_json, 'r', encoding='utf-8') as f:
                    cls.beards_data = json.load(f)
                    all_beard_names = [item["name"] for item in cls.beards_data]
        except Exception as e:
            print(f"FluxBeard Error: {e}")

        beard_colors = ["Default (Model's)", "Blonde", "Brunette", "Black", "Red", "Ginger", "Auburn", "White", "Silver", "Grey"]

        return {
            "required": {
                "beard_style": (all_beard_names, {"default": all_beard_names[0] if all_beard_names else ""}),
                "preserve_identity": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "preserve_beard_color": ("BOOLEAN", {"default": True, "label_on": "Original", "label_off": "Change"}),
                "beard_color": (beard_colors, {"default": "Default (Model's)"}),
                "extra_instruction": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("prompt", "preview", "beard_name")
    FUNCTION = "get_prompt"
    CATEGORY = "Style Loaders"

    def get_prompt(self, beard_style, preserve_identity, preserve_beard_color, beard_color, extra_instruction):
        selected_data = next((item for item in self.beards_data if item["name"] == beard_style), None)
        prompt_text = ""
        preview_image = None

        if selected_data:
            prefix = "Change beard style to "
            desc = selected_data.get("description", beard_style.lower())
            parts = [desc]
            
            if not preserve_beard_color:
                color_text = beard_color.replace("(Model's)", "").strip()
                if color_text: parts.append(f"{color_text} facial hair")
            else:
                parts.append("keeping original beard color")

            if extra_instruction and extra_instruction.strip(): parts.append(extra_instruction.strip())
            prompt_text = prefix + ", ".join(parts)

            if preserve_identity:
                prompt_text += " while maintaining the same facial features, eyes, and expression, clothing details and postures of the characters in the picture unchanged"

            thumb_name = selected_data.get("thumbnail", "")
            if thumb_name:
                thumbs_dir = os.path.join(os.path.dirname(__file__), "web", "thumbnails", "beards")
                img_path = os.path.join(thumbs_dir, thumb_name)
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        preview_image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                    except: pass
        
        if preview_image is None: preview_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return (prompt_text, preview_image, beard_style)

# ------------------------------------------------------------------
# РЕГИСТРАЦИЯ
# ------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "FluxHairstyleSelector": FluxHairstyleSelector,
    "FluxBeardSelector": FluxBeardSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxHairstyleSelector": "Flux Hairstyle Selector",
    "FluxBeardSelector": "Flux Beard Selector"
}