# Ken Burns Reel

A small tool to assemble a video slideshow with a Ken Burns effect and
beat-synchronised transitions. Images are analysed with OCR to extract
captions and OpenCV to determine a focus point.

## Installation

```
pip install -r requirements.txt
```

## Usage

```
python -m ken_burns_reel <input-folder>
```

The input folder should contain a set of images and a single audio file.

## Testing

```
pytest
```
