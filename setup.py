"""
Minimal setup script for Super Mario Bros AI Training System
"""

from setuptools import setup

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

# Minimal setup configuration
setup(
    name="super-mario-ai",
    version="1.0.0",
    author="Super Mario AI Development Team",
    description="AI Training System for Super Mario Bros using Deep Q-Learning",
    
    # Explicitly list only the python packages
    packages=[
        "python",
        "python.agents",
        "python.capture",
        "python.communication",
        "python.environment",
        "python.mario_logging",
        "python.models",
        "python.training",
        "python.utils"
    ],
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "mario-ai-train=python.main:main",
        ]
    },
    
    # Disable automatic discovery
    zip_safe=False,
)