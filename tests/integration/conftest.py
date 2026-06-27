"""Integration test fixtures — heavier setup with real I/O."""

import pytest


@pytest.fixture(autouse=True)
def integration_marker(request):
    """Auto-mark all tests in this folder as integration."""
    request.node.add_marker(pytest.mark.integration)
