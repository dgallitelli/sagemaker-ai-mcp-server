"""Tests for the helper functions in the SageMaker AI MCP Server."""

import os
from sagemaker_ai_mcp_server.helpers.utils import (
    get_aws_session,
    get_region,
    get_sagemaker_client,
    get_sagemaker_execution_role_arn,
)
from unittest.mock import MagicMock, patch


class TestUtils:
    """Tests for the SageMaker AI MCP Server helper functions."""

    def test_get_region_with_env_var(self):
        """Test get_region with AWS_REGION environment variable set."""
        with patch.dict(os.environ, {'AWS_REGION': 'eu-west-1'}):
            assert get_region() == 'eu-west-1'

    def test_get_region_default(self):
        """Test get_region with no AWS_REGION environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_region() == 'us-east-1'

    def test_get_sagemaker_execution_role_arn(self):
        """Test get_sagemaker_execution_role_arn function."""
        with patch.dict(
            os.environ,
            {
                'SAGEMAKER_EXECUTION_ROLE_ARN': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'
            },
        ):
            role_arn = get_sagemaker_execution_role_arn()
            assert role_arn == 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'

    @patch('sagemaker_ai_mcp_server.helpers.utils.boto3.Session')
    def test_get_aws_session_with_profile(self, mock_session):
        """Test get_aws_session with AWS_PROFILE environment variable."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        with patch.dict(os.environ, {'AWS_PROFILE': 'test-profile'}):
            session = get_aws_session('eu-west-1')

        mock_session.assert_called_once_with(profile_name='test-profile', region_name='eu-west-1')
        assert session == mock_session_instance

    @patch('sagemaker_ai_mcp_server.helpers.utils.boto3.Session')
    def test_get_aws_session_without_profile(self, mock_session):
        """Test get_aws_session without AWS_PROFILE environment variable."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        with patch.dict(os.environ, {}, clear=True):
            session = get_aws_session('us-west-2')

        mock_session.assert_called_once_with(region_name='us-west-2')
        assert session == mock_session_instance

    @patch('sagemaker_ai_mcp_server.helpers.utils.get_aws_session')
    def test_get_sagemaker_client(self, mock_get_aws_session):
        """Test get_sagemaker_client function."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_get_aws_session.return_value = mock_session

        client = get_sagemaker_client('us-west-1')

        mock_get_aws_session.assert_called_once_with('us-west-1')
        mock_session.client.assert_called_once_with('sagemaker')
        assert client == mock_client