"""Tests for the helper functions in the SageMaker AI MCP Server."""

import os
import pytest
from sagemaker_ai_mcp_server.helpers import (
    delete_endpoint,
    delete_endpoint_config,
    describe_endpoint,
    describe_endpoint_config,
    describe_processing_job,
    describe_training_job,
    get_aws_session,
    get_region,
    get_sagemaker_client,
    list_endpoint_configs,
    list_endpoints,
    list_processing_jobs,
    list_training_jobs,
    stop_processing_job,
    stop_training_job,
)
from unittest.mock import MagicMock, patch


class TestHelpers:
    """Tests for the SageMaker AI MCP Server helper functions."""

    def test_get_region_with_env_var(self):
        """Test get_region with AWS_REGION environment variable set."""
        with patch.dict(os.environ, {'AWS_REGION': 'eu-west-1'}):
            assert get_region() == 'eu-west-1'

    def test_get_region_default(self):
        """Test get_region with no AWS_REGION environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_region() == 'us-east-1'

    @patch('sagemaker_ai_mcp_server.helpers.boto3.Session')
    def test_get_aws_session_with_profile(self, mock_session):
        """Test get_aws_session with AWS_PROFILE environment variable."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        with patch.dict(os.environ, {'AWS_PROFILE': 'test-profile'}):
            session = get_aws_session('eu-west-1')

        mock_session.assert_called_once_with(profile_name='test-profile', region_name='eu-west-1')
        assert session == mock_session_instance

    @patch('sagemaker_ai_mcp_server.helpers.boto3.Session')
    def test_get_aws_session_without_profile(self, mock_session):
        """Test get_aws_session without AWS_PROFILE environment variable."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        with patch.dict(os.environ, {}, clear=True):
            session = get_aws_session('us-west-2')

        mock_session.assert_called_once_with(region_name='us-west-2')
        assert session == mock_session_instance

    @patch('sagemaker_ai_mcp_server.helpers.get_aws_session')
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

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_endpoints(self, mock_get_sagemaker_client):
        """Test list_endpoints function."""
        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = {
            'Endpoints': [{'EndpointName': 'test-endpoint'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        endpoints = await list_endpoints()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_endpoints.assert_called_once()
        assert endpoints == [{'EndpointName': 'test-endpoint'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_endpoint_configs(self, mock_get_sagemaker_client):
        """Test list_endpoint_configs function."""
        mock_client = MagicMock()
        mock_client.list_endpoint_configs.return_value = {
            'EndpointConfigs': [{'EndpointConfigName': 'test-config'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        configs = await list_endpoint_configs()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_endpoint_configs.assert_called_once()
        assert configs == [{'EndpointConfigName': 'test-config'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_delete_endpoint(self, mock_get_sagemaker_client):
        """Test delete_endpoint function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await delete_endpoint('test-endpoint')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.delete_endpoint.assert_called_once_with(EndpointName='test-endpoint')

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_delete_endpoint_config(self, mock_get_sagemaker_client):
        """Test delete_endpoint_config function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await delete_endpoint_config('test-config')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.delete_endpoint_config.assert_called_once_with(
            EndpointConfigName='test-config'
        )

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_endpoint(self, mock_get_sagemaker_client):
        """Test describe_endpoint function."""
        mock_client = MagicMock()
        expected_response = {'EndpointName': 'test-endpoint', 'Status': 'InService'}
        mock_client.describe_endpoint.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_endpoint('test-endpoint')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_endpoint.assert_called_once_with(EndpointName='test-endpoint')
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_endpoint_config(self, mock_get_sagemaker_client):
        """Test describe_endpoint_config function."""
        mock_client = MagicMock()
        expected_response = {'EndpointConfigName': 'test-config', 'ProductionVariants': []}
        mock_client.describe_endpoint_config.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_endpoint_config('test-config')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_endpoint_config.assert_called_once_with(
            EndpointConfigName='test-config'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_training_job(self, mock_get_sagemaker_client):
        """Test describe_training_job function."""
        mock_client = MagicMock()
        expected_response = {'TrainingJobName': 'test-job', 'TrainingJobStatus': 'Completed'}
        mock_client.describe_training_job.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_training_job('test-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_training_job.assert_called_once_with(TrainingJobName='test-job')
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_training_jobs(self, mock_get_sagemaker_client):
        """Test list_training_jobs function."""
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [{'TrainingJobName': 'test-job'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        jobs = await list_training_jobs()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_training_jobs.assert_called_once()
        assert jobs == [{'TrainingJobName': 'test-job'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_stop_training_job(self, mock_get_sagemaker_client):
        """Test stop_training_job function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await stop_training_job('test-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.stop_training_job.assert_called_once_with(TrainingJobName='test-job')

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_processing_job(self, mock_get_sagemaker_client):
        """Test describe_processing_job function."""
        mock_client = MagicMock()
        expected_response = {
            'ProcessingJobName': 'test-processing-job',
            'ProcessingJobStatus': 'Completed',
        }
        mock_client.describe_processing_job.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_processing_job('test-processing-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_processing_job.assert_called_once_with(
            ProcessingJobName='test-processing-job'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_processing_jobs(self, mock_get_sagemaker_client):
        """Test list_processing_jobs function."""
        mock_client = MagicMock()
        mock_client.list_processing_jobs.return_value = {
            'ProcessingJobSummaries': [{'ProcessingJobName': 'test-processing-job'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        jobs = await list_processing_jobs()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_processing_jobs.assert_called_once()
        assert jobs == [{'ProcessingJobName': 'test-processing-job'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_stop_processing_job(self, mock_get_sagemaker_client):
        """Test stop_processing_job function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await stop_processing_job('test-processing-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.stop_processing_job.assert_called_once_with(
            ProcessingJobName='test-processing-job'
        )
