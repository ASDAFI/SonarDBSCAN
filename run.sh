#!/bin/bash

set -e

echo "🚀 Setting up SonarDBSCAN environment..."

if ! command -v uv &> /dev/null
then
    echo "❌ 'uv' could not be found. Please install it first (e.g., curl -LsSf https://astral.sh/uv/install.sh | sh)"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with uv..."
    uv venv
else
    echo "✅ Virtual environment '.venv' already exists."
fi

source .venv/bin/activate

echo "⚙️  Installing SonarDBSCAN and development dependencies in editable mode..."
uv pip install -e .[dev]

echo "🧪 Running the Pytest test suite..."
pytest tests/

echo "🎉 All tests completed successfully!"