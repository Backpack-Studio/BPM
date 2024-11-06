# BPM - A C++ Header-Only Math Library

BPM (Backpack Mathematics) is a lightweight, header-only C++ math library built to handle a wide range of common mathematical operations. It provides a set of utilities for working with vectors, matrices, quaternions, and various essential mathematical functions, all designed with simplicity and efficiency in mind.

This library is designed to be used with **C++17** or later.

## Features

- **Vectors**: `Vec2`, `Vec3`, `Vec4` – Essential vector types for 2D, 3D, and 4D spaces, with common vector operations such as addition, subtraction, dot products, and normalization.
- **Matrices**: `Mat2`, `Mat3`, `Mat4` – Matrix types for 2D, 3D, and 4D transformations, including operations like multiplication, inversion, and transposition.
- **Quaternions**: `Quat` – Quaternion representation and operations for rotation and interpolation in 3D space.
- **Mathematical Functions**: A collection of utility functions including easing functions and other basic mathematical operations like trigonometric and exponential functions.

The library is designed to be simple, efficient, and easy to integrate into your C++ project.

## Requirements

- **C++17** or later
- No external dependencies

## Installation

As BPM is a header-only library, integration into your project is straightforward. You can include the library by copying the `include` folder into your project or by setting it up as an external dependency.

### Steps to Install:

1. Clone or download the repository:
   ```sh
   git clone https://github.com/Backpack-Studio/BPM.git
   ```
2. Copy the `include` folder to your project’s directory, or add it as an external dependency.

3. In your source files, you can include the necessary headers individually, like this:
   ```cpp
   #include "BPM/Vec2.hpp"  // For 2D vectors
   #include "BPM/Mat3.hpp"  // For 3x3 matrices
   #include "BPM/Quat.hpp"  // For quaternions
   ```

   Or, for convenience, you can include the **entire library** by including the `BPM.hpp` header:
   ```cpp
   #include "BPM/BPM.hpp"  // Includes all of BPM
   ```

## Usage

Once the library is integrated, you can start using its various features in your code. Here’s a quick example:

```cpp
#include "BPM/Vec3.hpp"
#include "BPM/Mat4.hpp"

int main() {
    bpm::Vec3 v1(1.0f, 2.0f, 3.0f);
    bpm::Vec3 v2(4.0f, 5.0f, 6.0f);

    // Perform vector operations
    bpm::Vec3 result = (v1 + v2) / 2.0f;

    // Create a transformation matrix
    bpm::Mat4 mat = bpm::Mat4::identity();

    return 0;
}
```

## License

This library is released under the **zlib License**. See the [LICENSE](LICENSE) file for more details.
