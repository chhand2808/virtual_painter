# Virtual Painter 🎨✋

## Overview
Virtual Painter is a fun, interactive application that allows you to draw on a digital canvas using hand gestures, powered by **OpenCV** and **Mediapipe**. Simply use your fingers to select colors and start drawing in freehand mode. You can also erase with a special gesture and adjust brush thickness in real time.

## Features 🚀
- **Hand Tracking**: Detects hand movements and finger positions using **Mediapipe**.
- **Color Selection**: Hover over a color box for 3 seconds to change the active color.
- **Freehand Drawing**: Touch your thumb and index finger together to start drawing.
- **Eraser Mode**: Bring your thumb, index, and middle fingers together to erase.
- **Adjustable Brush Size**: Use the slider to change the thickness of your strokes.
- **Save Your Drawing**: Press 'S' to save your artwork with a timestamped filename.

## Installation 📦
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/virtual-painter.git
   cd virtual-painter
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
3. Run the application:
   ```bash
   python virtual_painter.py
   ```

## Controls 🎮
| Gesture | Action |
|---------|--------|
| Hover over color box for 3 sec | Selects that color |
| Thumb & Index Finger Touch | Start drawing |
| Thumb, Index & Middle Finger Touch | Erase |
| Trackbar Slider | Adjust brush thickness |
| Press 'S' | Save the canvas as an image |
| Press 'Q' | Quit the application |

## Requirements 🛠️
- Python 3.x
- OpenCV
- Mediapipe
- NumPy

## Screenshots 📸
*(Add some screenshots here of your application in action!)*

## License 📜
This project is open-source and available under the **MIT License**.

---
Made with ❤️ by Chhand Chaughule

