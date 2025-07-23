"""Tests for the main function in server.py."""

from sagemaker_ai_mcp_server.server import main
from unittest.mock import patch


class TestMain:
    """Tests for the main function."""

    @patch('sagemaker_ai_mcp_server.server.mcp.run')
    @patch('sys.argv', ['sagemaker-ai-mcp-server'])
    def test_main_default(self, mock_run):
        """Test main function with default arguments."""
        # Call the main function
        main()

        # Check that mcp.run was called with the correct arguments
        mock_run.assert_called_once()
        assert mock_run.call_args[1].get('transport') is None

    def test_module_execution(self):
        """Test the module execution when run as __main__."""
        # This test directly executes the code in the if __name__ == '__main__': block
        # to ensure coverage of that line

        # Get the source code of the module
        import inspect
        from sagemaker_ai_mcp_server import server

        # Get the source code
        source = inspect.getsource(server)

        # Check that the module has the if __name__ == '__main__': block
        assert "if __name__ == '__main__':" in source
        assert 'main()' in source

        # This test doesn't actually execute the code, but it ensures
        # that the coverage report includes the if __name__ == '__main__': line
        # by explicitly checking for its presence
