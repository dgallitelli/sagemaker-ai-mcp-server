"""Tests for the sagemaker-ai-mcp-server package."""

import importlib
import re


class TestInit:
    """Tests for the __init__.py module."""

    def test_version(self):
        """Test that __version__ is defined and follows semantic versioning."""
        # Import the module
        import sagemaker_ai_mcp_server

        # Check that __version__ is defined
        assert hasattr(sagemaker_ai_mcp_server, '__version__')

        # Check that __version__ is a string
        assert isinstance(sagemaker_ai_mcp_server.__version__, str)

        # Check that __version__ follows semantic versioning (major.minor.patch)
        version_pattern = r'^\d+\.\d+\.\d+$'
        assert re.match(version_pattern, sagemaker_ai_mcp_server.__version__), (
            f"Version '{sagemaker_ai_mcp_server.__version__}' does not follow semantic versioning"
        )

    def test_module_reload(self):
        """Test that the module can be reloaded."""
        # Import the module
        import sagemaker_ai_mcp_server

        # Store the original version
        original_version = sagemaker_ai_mcp_server.__version__

        # Reload the module
        importlib.reload(sagemaker_ai_mcp_server)

        # Check that the version is still the same
        assert sagemaker_ai_mcp_server.__version__ == original_version
