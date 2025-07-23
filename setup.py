from setuptools import setup, find_packages

setup(
    name="eye_position_interface",
    version="0.1.0",
    description="双眼三维位置捕获接口，基于 RealSense D455 + MediaPipe",
    author="Your Name",
    author_email="youremail@example.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pyrealsense2>=2.50.0",
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.5",
        "numpy>=1.19.0",
    ],
    entry_points={
        "console_scripts": [
            "eye-position=eye_position_interface.interface:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
