# Qwen Image Edit Agent

Comprehensive AI image manipulation agent supporting Qwen models with advanced features for fashion e-commerce, character creation, and image enhancement.

## Features

### 1. **Qwen Image Editing** (`qwen_image_agent.py`)
- Character consistency editing
- Outfit changes and body transformation
- Pose modification
- Background editing
- Single/multiple character support
- Async/await architecture for scalability
- Batch processing capabilities

### 2. **Advanced Text Encoding** (`qwen_text_encode_advanced.py`)
- Vision-Language (VL) processing with Qwen2.5-VL
- Support for multiple reference images (up to 3)
- Configurable megapixels for VL model (0.0-4.0 MP)
- Reference latent encoding via VAE
- Llama template integration for improved results
- Dynamic image scaling for optimal performance

### 3. **Virtual Try-On & Body Shaper** (`virtual_tryon_body_shaper.py`)

#### Virtual Try-On Features:
- Clothing try-on (tops, bottoms, dresses, shoes, jackets, full outfits)
- Multiple body shape support (Petite, Slim, Athletic, Curvy, Plus-Size, Tall)
- 8 camera angle presets:
  - Straight On / Front View
  - Side View / Profile
  - 3/4 View / Three Quarter Angle
  - Back View / From Behind
  - Overhead Shot / From Above
  - Low Angle / Hero Shot
  - Close Up / Macro
  - Full Body / Wide Angle
- Fit adjustment (-1.0 to 1.0: loose to tight)
- Length adjustment (-1.0 to 1.0: short to long)
- Custom prompt support
- CAD-Von & SDXL integration

#### Body Shaper Features:
- Body shape transformation (7 preset shapes + custom)
- Adjustable transformation strength (0.0-1.0)
- Face preservation option
- Natural skin tone and confident expression
- Professional photo quality enhancement

### 4. **SDXL, LoRA & Character Creator** (In Development)

#### SDXL Integration:
- Text-to-image generation with SDXL
- LoRA (Low-Rank Adaptation) model support
- LoRA weight control (0.0-2.0)
- Trigger word integration
- Multi-model inference pipeline

#### Character Creation:
- Consistent character generation
- Multi-pose variations
- Emotional expression control
- Style preservation across generations
- InstantID integration for face consistency

#### Quality Boosters:
- SUPIR upscaling (4x/8x resolution)
- Face detail enhancement
- Real-ESRGAN integration
- Latent upscaling options
- Dynamic quality adjustment (0.0-1.0)

## Installation

```bash
# Clone the repository
git clone https://github.com/tabrezahmed51/qwen-image-edit-agent.git
cd qwen-image-edit-agent

# Install dependencies
pip install -r requirements.txt
```

## Required Dependencies

```
httpx==0.24.1
aiofiles==23.2.1
pydantic==2.5.0
python-dotenv==1.0.0
pillow==10.1.0
numpy==1.26.2
aiohttp==3.9.1
tqdm==4.67.0
loguru==0.7.2
```

## Usage Examples

### Basic Qwen Image Editing

```python
import asyncio
from qwen_image_agent import QwenImageAgent, ImageEditRequest, EditType

async def main():
    agent = QwenImageAgent(comfyui_endpoint="http://localhost:8188")
    
    request = ImageEditRequest(
        image_url="./input.png",
        edit_type=EditType.OUTFIT_CHANGE,
        prompt="Change to blue summer dress",
        steps=20,
        guidance_scale=7.5
    )
    
    result = await agent.process_edit_request(request)
    print(f"Result: {result}")

asyncio.run(main())
```

### Virtual Try-On

```python
from virtual_tryon_body_shaper import VirtualTryOnEngine, TryOnRequest, BodyShape, ClothingType, CameraAngle

engine = VirtualTryOnEngine()

request = TryOnRequest(
    model_image_url="./model.png",
    clothing_image_url="./dress.png",
    body_shape=BodyShape.CURVY,
    clothing_type=ClothingType.DRESS,
    camera_angle=CameraAngle.STRAIGHT_ON,
    adjust_fit=0.2,
    custom_prompt="elegant evening wear"
)

result = await engine.process_try_on(request)
```

### Body Shape Transformation

```python
from virtual_tryon_body_shaper import BodyShapeRequest

request = BodyShapeRequest(
    image_url="./person.png",
    target_body_shape=BodyShape.ATHLETIC,
    adjustment_strength=0.8,
    preserve_face=True
)

result = await engine.process_body_shape_transformation(request)
```

## API Endpoints (Future)

- `POST /api/edit` - Process image editing request
- `POST /api/tryon` - Virtual try-on request
- `POST /api/reshape` - Body shape transformation
- `POST /api/character` - Character creation
- `POST /api/batch` - Batch processing

## Model Requirements

### ComfyUI Setup:
- ComfyUI (latest version)
- CUDA-capable GPU (8GB+ VRAM recommended)

### Model Files:
- Qwen2.5-VL (vision-language model)
- SDXL base model
- LoRA models (optional)
- SUPIR model (for upscaling)
- Real-ESRGAN model (for enhancement)

## Performance Notes

- **Qwen Image Editing**: ~30-60s per image (20 steps)
- **Virtual Try-On**: ~45-90s per image
- **SUPIR Upscaling**: ~60-120s per image
- **Batch Processing**: Scales linearly with number of concurrent requests

## Architecture

```
qwen-image-edit-agent/
├── qwen_image_agent.py                    # Main async agent
├── qwen_text_encode_advanced.py            # Advanced text encoding with VL
├── virtual_tryon_body_shaper.py            # Try-on and body shaping
├── requirements.txt                        # Dependencies
└── README.md                               # This file
```

## Supported Edit Types

1. **CHARACTER_CONSISTENCY** - Maintain character features across edits
2. **OUTFIT_CHANGE** - Change clothing while keeping character
3. **POSE_CHANGE** - Modify body pose and stance
4. **BACKGROUND_CHANGE** - Replace or modify background
5. **MULTI_CHARACTER** - Edit multiple people in one image
6. **FULL_EDIT** - Complete image regeneration with custom prompt

## Supported Body Shapes

- Petite (small frame, shorter height)
- Slim (slender, lean physique)
- Athletic (muscular, toned)
- Curvy (hourglass figure)
- Plus Size (fuller figure)
- Tall (long legs, height emphasis)
- Custom (user-defined proportions)

## Camera Angles

8 professional photography angles optimized for fashion e-commerce:
- Straight On / Front View
- Side View / Profile
- 3/4 View (Three Quarter Angle)
- Back View (From Behind)
- Overhead Shot (From Above)
- Low Angle (Hero Shot)
- Close Up (Macro)
- Full Body (Wide Angle)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file

## Author

tabrezahmed51 - Full-stack developer specializing in AI/ML and ComfyUI workflows

## Support

For issues, questions, or feature requests:
- GitHub Issues: [Create an issue](https://github.com/tabrezahmed51/qwen-image-edit-agent/issues)
- Email: Contact via GitHub

## Roadmap

- [ ] SDXL-specific character creator module
- [ ] LoRA training pipeline
- [ ] SUPIR upscaling integration
- [ ] REST API server
- [ ] Web UI dashboard
- [ ] Model caching optimization
- [ ] GPU memory optimization
- [ ] Multi-GPU support
- [ ] Docker containerization
- [ ] Production deployment guide
